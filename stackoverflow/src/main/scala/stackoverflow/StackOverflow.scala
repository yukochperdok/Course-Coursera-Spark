package stackoverflow

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import annotation.tailrec

/** A raw stackoverflow posting, either a question or an answer */
case class Posting(postingType: Int, id: Int, acceptedAnswer: Option[Int], parentId: Option[QID], score: Int, tags: Option[String]) extends Serializable


/** The main class */
object StackOverflow extends StackOverflow {

  @transient lazy val conf: SparkConf = new SparkConf().setMaster("local").setAppName("StackOverflow")
  @transient lazy val sc: SparkContext = new SparkContext(conf)

  /** Main function */
  def main(args: Array[String]): Unit = {

    val lines   = sc.textFile("src/main/resources/stackoverflow/stackoverflow.csv")
    val raw     = rawPostings(lines)
    val grouped = groupedPostings(raw)
    val scored  = scoredPostings(grouped)

    // Para probar utilizamos un sample:
    //val scored = scoredPostings(grouped).sample(true, 0.1, 0)

    val vectors = vectorPostings(scored)
    assert(vectors.count() == 2121822, "Incorrect number of vectors: " + vectors.count())

    val means   = kmeans(sampleVectors(vectors), vectors, debug = true)

    // Para probar con una sola iteracion:
    //val means   = kmeans(sampleVectors(vectors), vectors, iter = 1, debug = true)

    val results = clusterResults(means, vectors)
    printResults(results)
  }
}


/** The parsing and kmeans methods */
class StackOverflow extends Serializable {

  /** Languages */
  val langs =
    List(
      "JavaScript", "Java", "PHP", "Python", "C#", "C++", "Ruby", "CSS",
      "Objective-C", "Perl", "Scala", "Haskell", "MATLAB", "Clojure", "Groovy")

  /** K-means parameter: How "far apart" languages should be for the kmeans algorithm? */
  def langSpread = 50000
  assert(langSpread > 0, "If langSpread is zero we can't recover the language from the input data!")

  /** K-means parameter: Number of clusters */
  def kmeansKernels = 45

  /** K-means parameter: Convergence criteria */
  def kmeansEta: Double = 20.0D

  /** K-means parameter: Maximum iterations */
  def kmeansMaxIterations = 120


  //
  //
  // Parsing utilities:
  //
  //

  /** Load postings from the given file */
  def rawPostings(lines: RDD[String]): RDD[Posting] =
    // Cada linea del fichero la transformamos en un elemento Posting
    lines.map(line => {
      val arr = line.split(",")
      Posting(postingType =    arr(0).toInt,
              id =             arr(1).toInt,
              acceptedAnswer = if (arr(2) == "") None else Some(arr(2).toInt),
              parentId =       if (arr(3) == "") None else Some(arr(3).toInt),
              score =          arr(4).toInt,
              tags =           if (arr.length >= 6) Some(arr(5).intern()) else None)
    })


  /** Group the questions and answers together */
  def groupedPostings(postings: RDD[Posting]): RDD[(QID, Iterable[(Question, Answer)])] = {
    // Separamos las preguntas y respuestas en pares (QID,Question) y (QID,Answer) respectivemente
    val questions: RDD[(QID,Question)] = postings.filter(_.postingType == 1).map(p => (p.id,p))
    val answers: RDD[(QID,Answer)] = postings.filter(_.postingType == 2).map(p => (p.parentId.get,p))
    // Hacemos join por el campo QID y obtenemos un RDD[(QID,(Question,Answer))]
    // Y agrupamos por QID y asi tenemos todos los pares (Question,Answer) de una misma pregunta en una lista
    // RDD[(QID, Iterable[(Question,Answer)])]
    questions.join(answers).groupByKey
  }


  /** Compute the maximum score for each posting */
  def scoredPostings(grouped: RDD[(QID, Iterable[(Question, Answer)])]): RDD[(Question, HighScore)] = {

    def answerHighScore(as: Array[Answer]): HighScore = {
      var highScore = 0
          var i = 0
          while (i < as.length) {
            val score = as(i).score
                if (score > highScore)
                  highScore = score
                  i += 1
          }
      highScore
    }
    // Desagrupamos la lista y nos quedamos solo con los pares (Question,Answer)
    // y nos queda un RDD[(Question,Answer)]
    grouped.flatMap(_._2)
      // Agrupamos por Question
      // y por lo tanto nos queda un RDD[(Question, Iterable[Answer])]
      .groupByKey()
      // Con la lista de Answer calculamos el highscore y lo mapeamos
      // Con lo cual nos queda un RDD[(Question,Int)]
      .mapValues(list => answerHighScore(list.toArray))


  }


  /** Compute the vectors for the kmeans */
  def vectorPostings(scored: RDD[(Question, HighScore)]): RDD[(LangIndex, HighScore)] = {
    /** Return optional index of first language that occurs in `tags`. */
    def firstLangInTag(tag: Option[String], ls: List[String]): Option[Int] = {
      if (tag.isEmpty) None
      else if (ls.isEmpty) None
      else if (tag.get == ls.head) Some(0) // index: 0
      else {
        val tmp = firstLangInTag(tag, ls.tail)
        tmp match {
          case None => None
          case Some(i) => Some(i + 1) // index i in ls.tail => index i+1
        }
      }
    }

    // Traducimos la Question a LangIndex
    // Para ello pasamos cada Question.tag a la funcion firstLangIntag
    // Si devuelve None transformamos a 0
    // Si devuelve un Some, multiplicamos el valor devuelto por un factor (langSpread)
    // luego lo emparejamos con x._2 (i.e highscore)
    scored.map(x => {
        val firstLang = firstLangInTag(x._1.tags,langs)
        firstLang match {
          case None => (0, x._2)
          case Some(firstLang) => (firstLangInTag(x._1.tags,langs).get * langSpread, x._2)
        }
    })
      // Persistimos en memoria porque luego lo vamos a utilizar en el kmeans repetidas veces
      // ganamos aprox un 10x
      .persist()
  }


  /** Sample the vectors */
  def sampleVectors(vectors: RDD[(LangIndex, HighScore)]): Array[(Int, Int)] = {

    assert(kmeansKernels % langs.length == 0, "kmeansKernels should be a multiple of the number of languages studied.")
    val perLang = kmeansKernels / langs.length

    // http://en.wikipedia.org/wiki/Reservoir_sampling
    def reservoirSampling(lang: Int, iter: Iterator[Int], size: Int): Array[Int] = {
      val res = new Array[Int](size)
      val rnd = new util.Random(lang)

      for (i <- 0 until size) {
        assert(iter.hasNext, s"iterator must have at least $size elements")
        res(i) = iter.next
      }

      var i = size.toLong
      while (iter.hasNext) {
        val elt = iter.next
        val j = math.abs(rnd.nextLong) % i
        if (j < size)
          res(j.toInt) = elt
        i += 1
      }

      res
    }

    val res =
      if (langSpread < 500)
        // sample the space regardless of the language
        vectors.takeSample(false, kmeansKernels, 42)
      else
        // sample the space uniformly from each language partition
        vectors.groupByKey.flatMap({
          case (lang, vectors) => reservoirSampling(lang, vectors.toIterator, perLang).map((lang, _))
        }).collect()

    assert(res.length == kmeansKernels, res.length)
    res
  }


  //
  //
  //  Kmeans method:
  //
  //

  /** Main kmeans computation */
  @tailrec final def kmeans(means: Array[(Int, Int)], vectors: RDD[(Int, Int)], iter: Int = 1, debug: Boolean = false): Array[(Int, Int)] = {
    val newMeans = means.clone() // you need to compute newMeans

    vectors.map(
      // Cada elemento (langIndex,highscore) lo emparejamos con su mean mas cercano
      vector => (findClosest(vector,means),vector))
      // Agrupamos por mean
      .groupByKey
      // Calculamos la media de los vectores asociado a cada mean
      .mapValues(averageVectors)
      .collect()
      // Para cada elemento actualizamos el newMeans por la mean mejorada.
      .foreach(
        par => {newMeans.update(par._1,par._2)}
      )

    // Calculamos la distancia euclidea total entre la media antigua y la nueva
    val distance = euclideanDistance(means, newMeans)

    if (debug) {
      println(s"""Iteration: $iter
                 |  * current distance: $distance
                 |  * desired distance: $kmeansEta
                 |  * means:""".stripMargin)
      for (idx <- 0 until kmeansKernels)
      println(f"   ${means(idx).toString}%20s ==> ${newMeans(idx).toString}%20s  " +
              f"  distance: ${euclideanDistance(means(idx), newMeans(idx))}%8.0f")
    }

    // Si converge (es optima) devolvemos la nueva media (es la mejor)
    if (converged(distance))
      newMeans
    // Si no converge y no hemos llegado al maximo de iteraciones, calculamos una nueva media
    // que converga mejor
    else if (iter < kmeansMaxIterations)
      kmeans(newMeans, vectors, iter + 1, debug)
    else {
      if (debug) {
        println("Reached max iterations!")
      }
      newMeans
    }
  }




  //
  //
  //  Kmeans utilities:
  //
  //

  /** Decide whether the kmeans clustering converged */
  def converged(distance: Double) =
    distance < kmeansEta


  /** Return the euclidean distance between two points */
  def euclideanDistance(v1: (Int, Int), v2: (Int, Int)): Double = {
    val part1 = (v1._1 - v2._1).toDouble * (v1._1 - v2._1)
    val part2 = (v1._2 - v2._2).toDouble * (v1._2 - v2._2)
    part1 + part2
  }

  /** Return the euclidean distance between two points */
  def euclideanDistance(a1: Array[(Int, Int)], a2: Array[(Int, Int)]): Double = {
    assert(a1.length == a2.length)
    var sum = 0d
    var idx = 0
    while(idx < a1.length) {
      sum += euclideanDistance(a1(idx), a2(idx))
      idx += 1
    }
    sum
  }

  /** Return the closest point */
  def findClosest(p: (Int, Int), centers: Array[(Int, Int)]): Int = {
    var bestIndex = 0
    var closest = Double.PositiveInfinity
    for (i <- 0 until centers.length) {
      val tempDist = euclideanDistance(p, centers(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }
    bestIndex
  }


  /** Average the vectors */
  def averageVectors(ps: Iterable[(Int, Int)]): (Int, Int) = {
    val iter = ps.iterator
    var count = 0
    var comp1: Long = 0
    var comp2: Long = 0
    while (iter.hasNext) {
      val item = iter.next
      comp1 += item._1
      comp2 += item._2
      count += 1
    }
    ((comp1 / count).toInt, (comp2 / count).toInt)
  }




  //
  //
  //  Displaying results:
  //
  //
  def clusterResults(means: Array[(Int, Int)], vectors: RDD[(LangIndex, HighScore)]): Array[(String, Double, Int, Int)] = {
    // Asociamos a cada par (langIndex, HighScore) su media mas cercana y agrupamos por media
    // Por lo tanto tendremos RDD[(langIndexMedian,Iterable[(langIndex,highScore)])]
    // Que en realidad son los clusters
    val closest = vectors.map(p => (findClosest(p, means), p))
    val closestGrouped = closest.groupByKey()

    // Para cada valor de cada media (es decir para cada cluster) trabajamos con el Iterable
    val median = closestGrouped.mapValues { vs =>
      // Cogemos del para (langIndex, highScore) el langIndex y dividimos por el factor
      // y agrupamos por identity. Por lo tanto nos queda algo como Map[langIndex,Iterable[langIndex]]
      // luego transformamos el Iterable en su tamaño, resultando: Map[langIndex,tamanio]
      val groupedLangsCluster = vs.map(_._1/langSpread).groupBy(identity).mapValues(_.size)
      // Nosquedamos con el que mas tamaño tiene y cogemos su langIndex
      val indexMaxLangCluster = groupedLangsCluster.maxBy(_._2)._1

      // Traducimos langIndex a lang
      val langLabel: String   = langs(indexMaxLangCluster)

      // Calculamos el tamanio del cluster
      val clusterSize: Int    = vs.size

      // Relacion (porcentaje %) del lang predominante en el cluster / total del cluster
      val langPercent: Double = groupedLangsCluster(indexMaxLangCluster) * 100d / clusterSize

      // Cogemos la lista de highScore y la ordenamos
      // Para coger el elemento que se queda en la media (dependera del tamanio del cluster)
      val sortedScoreCluster = vs.map(_._2).toList.sorted
      val medianScore: Int    =
        // Si es impar cogemos el elemento medio
        // pero si es par hacemos la media del elemento medio y el medio -1
        if(clusterSize % 2 == 0)
          (sortedScoreCluster((clusterSize / 2)-1) + sortedScoreCluster(clusterSize / 2)) / 2
        else
          sortedScoreCluster(clusterSize / 2)


      (langLabel, langPercent, clusterSize, medianScore)
    }
    // Nos quedamos con el value : (langLabel, langPercent, clusterSize, medianScore)
    // Y lo ordenamos por medianScore
    median.collect().map(_._2).sortBy(_._4)
  }

  def printResults(results: Array[(String, Double, Int, Int)]): Unit = {
    println("Resulting clusters:")
    println("  Score  Dominant language (%percent)  Questions")
    println("================================================")
    for ((lang, percent, size, score) <- results)
      println(f"${score}%7d  ${lang}%-17s (${percent}%-5.1f%%)      ${size}%7d")
  }
}
