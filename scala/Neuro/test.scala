package Neuro

import java.io.{FileInputStream, ObjectInputStream}
import java.nio.file.{Files, Paths}

import Neuro.Calculate._
import Neuro.PrepareData._
import Neuro.runApp._


object test extends App {


  val batch = 784

  ///home/serge/projexts/Neuro/src/main/scala/train
  ///home/serge/projexts/Neuro/src/main/scala/names

  var train = Files.readAllBytes(Paths.get("/home/serge/Рабочий стол/aginandagin/src/main/scala/Neuro/data"))
//  var train = Files.readAllBytes(Paths.get("/home/serge/projexts/Neuro/src/main/scala/train"))
  var names = Files.readAllBytes(Paths.get("/home/serge/Рабочий стол/aginandagin/src/main/scala/Neuro/name"))
//  var names = Files.readAllBytes(Paths.get("/home/serge/projexts/Neuro/src/main/scala/names"))

  def prepare(x : Int) = getImage(x, (train, names), 28, 28, batch)

  var data = prepare(60)


  val neur1 = neuronArray(16)
  val neur2 = neuronArray(16)
  val neur3 = neuronArray(10)

  var w1 = prepareWeights(28 * 28, 16)
  var w2 = prepareWeights(16, 16)
  var w3 = prepareWeights(16, 10)

  def openClass(): Unit = {
    val is = new ObjectInputStream(new FileInputStream("temp011111"))
    val weights = is.readObject().asInstanceOf[Weights]
    w1 = weights.w1
    w2 = weights.w2
    w3 = weights.w3
  }

  def calculateOutputsIteration(): Unit = {
    neur1.foreach(x => {
      x.output = 1.0
      x.error = 0.0
    })
    neur2.foreach(x => {
      x.output = 1.0
      x.error = 0.0
    })
    neur3.foreach(x => {
      x.output = 1.0
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


  openClass()

  var k = 0
  var index = 0
  for (nb <- 1 to 10000) {
   // if (nb % 100 == 0) println("-------" + nb)
    data = prepare(nb)
  //val time = System.currentTimeMillis()
    calculateOutputsIteration()
  //println(System.currentTimeMillis() - time)
    var maxSum = 0.0
    index = 0
    neur3.foreach(x => {
      if (x.output > maxSum) {
        maxSum = x.output
        index = neur3.indexOf(x)
      }
    })
    if (index == data._1) k+=1
  }
  outValues(data._2)
  println(index)
  println((k.doubleValue() / 10000.0) * 100 + " %")
}
