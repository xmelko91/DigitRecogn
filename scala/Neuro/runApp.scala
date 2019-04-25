package Neuro

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.file.{Files, Paths}

object runApp extends App {
    import PrepareData._
  import Calculate._


  val batch = 784
  var train = Files.readAllBytes(Paths.get("/home/serge/projexts/Neuro/src/main/scala/train"))
  var names = Files.readAllBytes(Paths.get("/home/serge/projexts/Neuro/src/main/scala/names"))

  def prepare(x : Int) = getImage(x, (train, names), 28, 28, batch)

    var data = prepare(60)
  val neur1 = neuronArray(16)
  val neur2 = neuronArray(16)
  val neur3 = neuronArray(10)

  var w1 = prepareWeights(28 * 28, 16)
  var w2 = prepareWeights(16, 16)
  var w3 = prepareWeights(16, 10)

  def calculateOutputsIteration(): Unit = {
    neur1.foreach(x => {
      x.output = 0.0
      x.error = 0.0
    })
    neur2.foreach(x => {
      x.output = 0.0
      x.error = 0.0
    })
    neur3.foreach(x => {
      x.output = 0.0
      x.error = 0.0
    })

    calculateOuts(data._2, w1, neur1)
    calculateOuts(neurAsDouble(neur1), w2, neur2)
    calculateOuts(neurAsDouble(neur2), w3, neur3)
  }

  def errorIteration(): Unit = {
    outputError(neur3, data._1)
    hiddenError(neur2, w3, neur3)
    hiddenError(neur1, w2, neur2)
  }

  def calculatingWeights():Unit = {
    calcNewWeight(data._2, w1, neur1)
    calcNewWeight(neurAsDouble(neur1), w2, neur2)
    calcNewWeight(neurAsDouble(neur2), w3, neur3)
  }

  def calculatingIteration(nb : Int): Unit = {
    data = prepare(nb)
    //outValues(data._2)

    calculateOutputsIteration()
    errorIteration()
    calculatingWeights()
    saveAsClass(w1, w2, w3)
  }

  var nameP = "temp0"

  def saveAsClass(w11 : Array[Array[Double]], w21 : Array[Array[Double]], w31 : Array[Array[Double]]): Unit = {
    val weights = new Weights {
      override val w1: Array[Array[Double]] = w11
      override val w2: Array[Array[Double]] = w21
      override val w3: Array[Array[Double]] = w31
    }

    val os = new ObjectOutputStream(new FileOutputStream(nameP))
    os.writeObject(weights)
    os.close()
  }

  def openClass(): Unit = {
    val is = new ObjectInputStream(new FileInputStream("temp011111"))
    val weights = is.readObject().asInstanceOf[Weights]
    w1 = weights.w1
    w2 = weights.w2
    w3 = weights.w3
  }

  //openClass()

  var sum = 0.0

  for (z <- 0 to 20) {
    for (x <- 1 until 60000) {
      calculatingIteration(x)
      sum += calcTotalError(neur3)
      if (x % 1000 == 0) {
        println(sum)
        sum = 0.0
        if (x % 10000 == 0) nameP += "1"
        saveAsClass(w1, w2, w3)
      }
    }
    println("----epoch----" + z)
    nameP = "temp0"
  }
  saveAsClass(w1, w2, w3)

}
