package Neuro

import java.nio.file.{Files, Paths}

import scala.collection.mutable.ArrayBuffer

object PrepareData {

  def prepareData(number: Int) : (Int, Array[Double]) = {
    //Setups - don't touch
    val batch = 784
    var train = Files.readAllBytes(Paths.get("/home/serge/projexts/Neuro/src/main/scala/train"))
    var names = Files.readAllBytes(Paths.get("/home/serge/projexts/Neuro/src/main/scala/names"))
    var data = (train, names)
    //print(math.BigInt.apply(train.slice(0, 4)) + "\n")
    //print(math.BigInt.apply(train.slice(4, 8)) + "\n")
    val rows = math.BigInt.apply(train.slice(8, 12)).intValue()
    val columns = math.BigInt.apply(train.slice(12, 16)).intValue()
    //print(rows + " " + columns + "\n")
    getImage(number, data, rows, columns, batch)

  }


  def getImage(number: Int, train: (Array[Byte], Array[Byte]), rows: Int, columns: Int, batch: Int): (Int, Array[Double]) = {
    (math.BigInt.apply(train._2.slice(7 + number, 8 + number)).intValue()
      , getNewArray(train._1.slice(16 + batch * (number - 1), 16 + batch * number), rows.intValue(), columns.intValue()).toArray)
  }

  private def getNewArray(arr: Array[Byte], rows: Int, columns: Int): ArrayBuffer[Double] = {
    var newArr = new ArrayBuffer[Double]()
    for (x <- arr) {
      if (x.intValue() < 0)
        newArr += ((126 + x.intValue()) + 124.9).doubleValue() / 255.0
      else
        newArr += (x.intValue().doubleValue() + 0.1) / 255.0
    }
    newArr
  }
//printing input image
  def outValues(value: Array[Double], rows: Int = 28) = {
    var z = 1
    for (y <- value) {
      print(y.round + " ")
      if (z % rows == 0 && z != 0)
        print("\n")
      z += 1
    }
  }

  def prepareRandomArr(size : Int):ArrayBuffer[Int] = {
    var out = new ArrayBuffer[Int]()
    for (x <- 0 until size){
      out += x
    }
    out
  }

  def prepareWeights(past : Int, now : Int) : Array[Array[Double]] = {
    val out = new Array[Array[Double]](past)
    for (x <- 0 until past){
      val arr = new Array[Double](now)
      for (y <- 0 until now){
        arr(y) =math.random() * ( 1 / math.sqrt(past))
      }
      out(x) = arr
    }
    out
  }

  def neuronArray(pastNodes: Int): Array[Neuron] = {
    val out = new Array[Neuron](pastNodes)
    for (x <- 0 until pastNodes)
      out(x) = new Neuron
    out
  }

}
