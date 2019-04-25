package Neuro

object Calculate {

  val learn = 0.3

  def calculateOuts(in : Array[Double], arr: Array[Array[Double]], out: Array[Neuron]): Unit = {
    for (x <- arr.indices){
      for (y <- out.indices){
        out(y).output += in(x) * arr(x)(y)
      }
    }
    out.foreach(x => x.output = sigmoid(x.output))
  }

  def sigmoid(x : Double): Double = 1 / (1 + math.exp(-x))

  def neurAsDouble(in: Array[Neuron]) : Array[Double] = {
    val out = new Array[Double](in.length)
    for (x <- in.indices)
      out(x) = in(x).output
    out
  }

  def outputError(out : Array[Neuron], pref : Int): Unit = {
    for (x <- out.indices){
      var need = 0.01
      if (x == pref) {need = 0.99}
      out(x).error = need - out(x).output
    }
  }

  def hiddenError(past : Array[Neuron], arr: Array[Array[Double]], now: Array[Neuron]): Unit = {
    for (x <- past.indices){
      for (y <- now.indices){
        past(x).error += now(y).error * arr(x)(y)
      }
    }
  }

  def calcNewWeight(past : Array[Double], arr : Array[Array[Double]], now : Array[Neuron]): Unit = {
    for (x <- past.indices){
      for (y <- now.indices){
        arr(x)(y) += learn * now(y).error * now(y).output * (1 - now(y).output) * past(x)
      }
    }
  }

  def calcTotalError(out : Array[Neuron]): Double = {
    out.map(x => x.error * x.error).sum
  }

}
