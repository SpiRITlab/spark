/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml_fhe.linalg

import java.lang.{Double => JavaDouble, Integer => JavaInteger, Iterable => JavaIterable}
import java.util

import scala.annotation.varargs
import scala.collection.JavaConverters._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

import org.apache.spark.annotation.Since

/**
 * Represents a numeric vector, whose index type is Int and value type is Double.
 *
 * @note Users should not implement this interface.
 */
@Since("2.0.0")
sealed trait CtxtVector extends Serializable {

  /**
   * Size of the vector.
   */
  @Since("2.0.0")
  def size: Int

  /**
   * Converts the instance to a double array.
   */
  @Since("2.0.0")
  def toArray: Array[Double]

  override def equals(other: Any): Boolean = {
    other match {
      case v2: CtxtVector =>
        if (this.size != v2.size) return false
        (this, v2) match {
          case (_, _) => util.Arrays.equals(this.toArray, v2.toArray)
        }
      case _ => false
    }
  }

  /**
   * Returns a hash code value for the vector. The hash code is based on its size and its first 128
   * nonzero entries, using a hash algorithm similar to `java.util.Arrays.hashCode`.
   */
  override def hashCode(): Int = {
    // This is a reference implementation. It calls return in foreachActive, which is slow.
    // Subclasses should override it with optimized implementation.
    var result: Int = 31 + size
    var nnz = 0
    this.foreachActive { (index, value) =>
      if (nnz < CtxtVectors.MAX_HASH_NNZ) {
        // ignore explicit 0 for comparison between sparse and dense
        if (value != 0) {
          result = 31 * result + index
          val bits = java.lang.Double.doubleToLongBits(value)
          result = 31 * result + (bits ^ (bits >>> 32)).toInt
          nnz += 1
        }
      } else {
        return result
      }
    }
    result
  }

  /**
   * Converts the instance to a breeze vector.
   */
  private[spark] def asBreeze: BV[Double]

  /**
   * Gets the value of the ith element.
   * @param i index
   */
  @Since("2.0.0")
  def apply(i: Int): Double = asBreeze(i)

  /**
   * Makes a deep copy of this vector.
   */
  @Since("2.0.0")
  def copy: CtxtVector = {
    throw new NotImplementedError(s"copy is not implemented for ${this.getClass}.")
  }

  /**
   * Applies a function `f` to all the active elements of dense and sparse vector.
   *
   * @param f the function takes two parameters where the first parameter is the index of
   *          the vector with type `Int`, and the second parameter is the corresponding value
   *          with type `Double`.
   */
  @Since("2.0.0")
  def foreachActive(f: (Int, Double) => Unit): Unit

  /**
   * Number of active entries.  An "active entry" is an element which is explicitly stored,
   * regardless of its value.  Note that inactive entries have value 0.
   */
  @Since("2.0.0")
  def numActives: Int

  /**
   * Number of nonzero elements. This scans all active values and count nonzeros.
   */
  @Since("2.0.0")
  def numNonzeros: Int

  /**
   * Converts this vector to a dense vector.
   */
  @Since("2.0.0")
  def toDense: CtxtDenseVector = new CtxtDenseVector(this.toArray)

  /**
   * Returns a vector in either dense or sparse format, whichever uses less storage.
   */
  @Since("2.0.0")
  def compressed: CtxtVector = {
    val nnz = numNonzeros
    // A dense vector needs 8 * size + 8 bytes, while a sparse vector needs 12 * nnz + 20 bytes.
    toDense
  }

  /**
   * Find the index of a maximal element.  Returns the first maximal element in case of a tie.
   * Returns -1 if vector has length 0.
   */
  @Since("2.0.0")
  def argmax: Int
}

/**
 * Factory methods for [[org.apache.spark.ml.linalg.Vector]].
 * We don't use the name `Vector` because Scala imports
 * `scala.collection.immutable.Vector` by default.
 */
@Since("2.0.0")
object CtxtVectors {

  /**
   * Creates a dense vector from its values.
   */
  @varargs
  @Since("2.0.0")
  def dense(firstValue: Double, otherValues: Double*): CtxtVector =
    new CtxtDenseVector((firstValue +: otherValues).toArray)

  // A dummy implicit is used to avoid signature collision with the one generated by @varargs.
  /**
   * Creates a dense vector from a double array.
   */
  @Since("2.0.0")
  def dense(values: Array[Double]): CtxtVector = new CtxtDenseVector(values)

  /**
   * Creates a vector of all zeros.
   *
   * @param size vector size
   * @return a zero vector
   */
  @Since("2.0.0")
  def zeros(size: Int): CtxtVector = {
    new CtxtDenseVector(new Array[Double](size))
  }

  /**
   * Creates a vector instance from a breeze vector.
   */
  private[spark] def fromBreeze(breezeVector: BV[Double]): CtxtVector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new CtxtDenseVector(v.data)
        } else {
          new CtxtDenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  /**
   * Returns the p-norm of this vector.
   * @param vector input vector.
   * @param p norm.
   * @return norm in L^p^ space.
   */
  @Since("2.0.0")
  def norm(vector: CtxtVector, p: Double): Double = {
    require(p >= 1.0, "To compute the p-norm of the vector, we require that you specify a p>=1. " +
      s"You specified p=$p.")
    val values = vector match {
      case CtxtDenseVector(vs) => vs
      case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
    }
    val size = values.length

    if (p == 1) {
      var sum = 0.0
      var i = 0
      while (i < size) {
        sum += math.abs(values(i))
        i += 1
      }
      sum
    } else if (p == 2) {
      var sum = 0.0
      var i = 0
      while (i < size) {
        sum += values(i) * values(i)
        i += 1
      }
      math.sqrt(sum)
    } else if (p == Double.PositiveInfinity) {
      var max = 0.0
      var i = 0
      while (i < size) {
        val value = math.abs(values(i))
        if (value > max) max = value
        i += 1
      }
      max
    } else {
      var sum = 0.0
      var i = 0
      while (i < size) {
        sum += math.pow(math.abs(values(i)), p)
        i += 1
      }
      math.pow(sum, 1.0 / p)
    }
  }

  /**
   * Returns the squared distance between two Vectors.
   * @param v1 first Vector.
   * @param v2 second Vector.
   * @return squared distance between two Vectors.
   */
  @Since("2.0.0")
  def sqdist(v1: CtxtVector, v2: CtxtVector): Double = {
    require(v1.size == v2.size, s"Vector dimensions do not match: Dim(v1)=${v1.size} and Dim(v2)" +
      s"=${v2.size}.")
    var squaredDistance = 0.0
    (v1, v2) match {
      case (CtxtDenseVector(vv1), CtxtDenseVector(vv2)) =>
        var kv = 0
        val sz = vv1.length
        while (kv < sz) {
          val score = vv1(kv) - vv2(kv)
          squaredDistance += score * score
          kv += 1
        }
      case _ =>
        throw new IllegalArgumentException("Do not support vector type " + v1.getClass +
          " and " + v2.getClass)
    }
    squaredDistance
  }

  /**
   * Check equality between sparse/dense vectors
   */
  private[ml_fhe] def equals(
      v1Indices: IndexedSeq[Int],
      v1Values: Array[Double],
      v2Indices: IndexedSeq[Int],
      v2Values: Array[Double]): Boolean = {
    val v1Size = v1Values.length
    val v2Size = v2Values.length
    var k1 = 0
    var k2 = 0
    var allEqual = true
    while (allEqual) {
      while (k1 < v1Size && v1Values(k1) == 0) k1 += 1
      while (k2 < v2Size && v2Values(k2) == 0) k2 += 1

      if (k1 >= v1Size || k2 >= v2Size) {
        return k1 >= v1Size && k2 >= v2Size // check end alignment
      }
      allEqual = v1Indices(k1) == v2Indices(k2) && v1Values(k1) == v2Values(k2)
      k1 += 1
      k2 += 1
    }
    allEqual
  }

  /** Max number of nonzero entries used in computing hash code. */
  private[linalg] val MAX_HASH_NNZ = 128
}

/**
 * A dense vector represented by a value array.
 */
@Since("2.0.0")
class CtxtDenseVector @Since("2.0.0") ( @Since("2.0.0") val values: Array[Double])
  extends CtxtVector {

  override def size: Int = values.length

  override def toString: String = values.mkString("[", ",", "]")

  override def toArray: Array[Double] = values

  private[spark] override def asBreeze: BV[Double] = new BDV[Double](values)

  override def apply(i: Int): Double = values(i)

  override def copy: CtxtDenseVector = {
    new CtxtDenseVector(values.clone())
  }

  override def foreachActive(f: (Int, Double) => Unit): Unit = {
    var i = 0
    val localValuesSize = values.length
    val localValues = values

    while (i < localValuesSize) {
      f(i, localValues(i))
      i += 1
    }
  }

  override def equals(other: Any): Boolean = super.equals(other)

  override def hashCode(): Int = {
    var result: Int = 31 + size
    var i = 0
    val end = values.length
    var nnz = 0
    while (i < end && nnz < CtxtVectors.MAX_HASH_NNZ) {
      val v = values(i)
      if (v != 0.0) {
        result = 31 * result + i
        val bits = java.lang.Double.doubleToLongBits(values(i))
        result = 31 * result + (bits ^ (bits >>> 32)).toInt
        nnz += 1
      }
      i += 1
    }
    result
  }

  override def numActives: Int = size

  override def numNonzeros: Int = {
    // same as values.count(_ != 0.0) but faster
    var nnz = 0
    values.foreach { v =>
      if (v != 0.0) {
        nnz += 1
      }
    }
    nnz
  }

  override def argmax: Int = {
    if (size == 0) {
      -1
    } else {
      var maxIdx = 0
      var maxValue = values(0)
      var i = 1
      while (i < size) {
        if (values(i) > maxValue) {
          maxIdx = i
          maxValue = values(i)
        }
        i += 1
      }
      maxIdx
    }
  }
}

@Since("2.0.0")
object CtxtDenseVector {

  /** Extracts the value array from a dense vector. */
  @Since("2.0.0")
  def unapply(dv: CtxtDenseVector): Option[Array[Double]] = Some(dv.values)
}

