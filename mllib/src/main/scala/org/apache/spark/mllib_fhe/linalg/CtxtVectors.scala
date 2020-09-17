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

package org.apache.spark.mllib_fhe.linalg

import java.math.BigInteger
import java.util

import scala.annotation.varargs
import scala.language.implicitConversions

import spiritlab.sparkfhe.api.{SparkFHE, SparkFHEConstants, StringVector}

import org.apache.spark .annotation.{AlphaComponent, Since}
import org.apache.spark.ml_fhe.{linalg => newlinalg}
import org.apache.spark.mllib.util.NumericParser
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{GenericInternalRow, UnsafeArrayData}
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String


/**
 * Represents a numeric vector, whose index type is Int and value type is Double.
 *
 * @note Users should not implement this interface.
 */
@SQLUserDefinedType(udt = classOf[CtxtVectorUDT])
@Since("1.0.0")
sealed trait CtxtVector extends Serializable {

  /**
   * Size of the vector.
   */
  @Since("1.0.0")
  def size: Int

  /**
   * Converts the instance to a double array.
   */
  @Since("1.0.0")
  def toArray: Array[String]

  override def equals(other: Any): Boolean = {
    other match {
      case v2: CtxtVector =>
        if (this.size != v2.size) return false
        (this, v2) match {
          case (_, _) => this.toArray.deep == v2.toArray.deep
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
        if (value != "") {
          result = 31 * result + index
          val bits = java.lang.Long.parseLong(new BigInteger(value.getBytes()).toString(2))
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
  // private[spark] def asBreeze: BV[Double]

  /**
   * Gets the value of the ith element.
   * @param i index
   */
  @Since("1.1.0")
  def apply(i: Int): String = this.toArray(i)

  /**
   * Makes a deep copy of this vector.
   */
  @Since("1.1.0")
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
  @Since("1.6.0")
  def foreachActive(f: (Int, String) => Unit): Unit

  /**
   * Number of active entries.  An "active entry" is an element which is explicitly stored,
   * regardless of its value.
   *
   * @note Inactive entries have value 0.
   */
  /* @Since("1.4.0")
  def numActives: Int */

  /**
   * Number of nonzero elements. This scans all active values and count nonzeros.
   */
  @Since("1.4.0")
  def numNonzeros: String

  /**
   * Converts this vector to a dense vector.
   */
  @Since("1.4.0")
  def toDense: CtxtDenseVector = new CtxtDenseVector(this.toArray)

  /**
   * Returns a vector in either dense or sparse format, whichever uses less storage.
   */
  /* @Since("1.4.0")
  def compressed: CtxtVector = {
    val nnz = numNonzeros
    // A dense vector needs 8 * size + 8 bytes, while a sparse vector needs 12 * nnz + 20 bytes.
    toDense
  } */

  /**
   * Find the index of a maximal element.  Returns the first maximal element in case of a tie.
   * Returns -1 if vector has length 0.
   */
  /* @Since("1.5.0")
  def argmax: Int */

  /**
   * Converts the vector to a JSON string.
   */
  /* @Since("1.6.0")
  def toJson: String */

  /**
   * Convert this vector to the new mllib-local representation.
   * This does NOT copy the data; it copies references.
   */
  @Since("2.0.0")
  def asML: newlinalg.CtxtVector
}

/**
 * :: AlphaComponent ::
 *
 * User-defined type for [[CtxtVector]] which allows easy interaction with SQL
 * via [[org.apache.spark.sql.Dataset]].
 */
@AlphaComponent
class CtxtVectorUDT extends UserDefinedType[CtxtVector] {

  override def sqlType: StructType = {
    // type: 0 = sparse, 1 = dense
    // We only use "values" for dense vectors, and "size", "indices", and "values" for sparse
    // vectors. The "values" field is nullable because we might want to add binary vectors later,
    // which uses "size" and "indices", but not "values".
    StructType(Seq(
      StructField("type", ByteType, nullable = false),
      StructField("size", IntegerType, nullable = true),
      StructField("indices", ArrayType(IntegerType, containsNull = false), nullable = true),
      StructField("values", ArrayType(StringType, containsNull = false), nullable = true)))
  }

  override def serialize(obj: CtxtVector): InternalRow = {
    obj match {
      case CtxtDenseVector(values) =>
        val row = new GenericInternalRow(4)
        row.setByte(0, 1)
        row.setNullAt(1)
        row.setNullAt(2)
        row.update(3, ArrayData.toArrayData(values.map { x => UTF8String.fromString(x) }))
        row
    }
  }

  override def deserialize(datum: Any): CtxtVector = {
    datum match {
      case row: InternalRow =>
        require(row.numFields == 4,
          s"CtxtVectorUDT.deserialize given row with length ${row.numFields} but requires length" +
            s" == 4")
        val tpe = row.getByte(0)
        tpe match {
          case 1 =>
            val values = row.getArray(3).toArray[UTF8String](StringType).map { x => x.toString}
            new CtxtDenseVector(values)
        }
    }
  }

  override def pyUDT: String = "pyspark.mllib.linalg.CtxtVectorUDT"

  override def userClass: Class[CtxtVector] = classOf[CtxtVector]

  override def equals(o: Any): Boolean = {
    o match {
      case v: CtxtVectorUDT => true
      case _ => false
    }
  }

  // see [SPARK-8647], this achieves the needed constant hash code without constant no.
  override def hashCode(): Int = classOf[CtxtVectorUDT].getName.hashCode()

  override def typeName: String = "ctxtvector"

  private[spark] override def asNullable: CtxtVectorUDT = this
}

/**
 * Factory methods for [[org.apache.spark.mllib_fhe.linalg.CtxtVector]].
 * We don't use the name `Vector` because Scala imports
 * `scala.collection.immutable.Vector` by default.
 */
@Since("1.0.0")
object CtxtVectors {

  /**
   * Creates a dense vector from its values.
   */
  @Since("1.0.0")
  @varargs
  def dense(firstValue: String, otherValues: String*): CtxtVector =
    new CtxtDenseVector((firstValue +: otherValues).toArray)

  // A dummy implicit is used to avoid signature collision with the one generated by @varargs.
  /**
   * Creates a dense vector from a double array.
   */
  @Since("1.0.0")
  def dense(values: Array[String]): CtxtVector = new CtxtDenseVector(values)

  /**
   * Creates a vector of all zeros.
   *
   * @param size vector size
   * @return a zero vector
   */
  @Since("1.1.0")
  def zeros(size: Int): CtxtVector = {
    new CtxtDenseVector(new Array[String](size))
  }

  /**
   * Parses a string resulted from `CtxtVector.toString` into a [[CtxtVector]].
   */
  /* @Since("1.1.0")
  def parse(s: String): CtxtVector = {
    parseNumeric(NumericParser.parse(s))
  } */

  /**
   * Parses the JSON representation of a vector into a [[CtxtVector]].
   */
  /* @Since("1.6.0")
  def fromJson(json: String): CtxtVector = {
    implicit val formats = DefaultFormats
    val jValue = parseJson(json)
    (jValue \ "type").extract[Int] match {
      case 1 => // dense
        val values = (jValue \ "values").extract[Seq[Double]].toArray
        dense(values)
      case _ =>
        throw new IllegalArgumentException(s"Cannot parse $json into a vector.")
    }
  } */

  /* private[mllib_fhe] def parseNumeric(any: Any): CtxtVector = {
    any match {
      case values: Array[Double] =>
        CtxtVectors.dense(values)
      case other =>
        throw new SparkException(s"Cannot parse $other.")
    }
  } */

  /**
   * Creates a vector instance from a breeze vector.
   */
  /* private[spark] def fromBreeze(breezeVector: BV[Double]): CtxtVector = {
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
  } */

//  /**
//   * Returns the p-norm of this vector.
//   * @param vector input vector.
//   * @param p norm.
//   * @return norm in L^p^ space.
//   */
  /* @Since("1.3.0")
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
  } */

  /**
   * Returns the squared distance between two Vectors.
   * @param v1 first Vector.
   * @param v2 second Vector.
   * @return squared distance between two Vectors.
   */
  @Since("1.3.0")
  def sqdist(v1: CtxtVector, v2: CtxtVector): String = {
    require(v1.size == v2.size, s"Vector dimensions do not match: Dim(v1)=${v1.size} and Dim(v2)" +
      s"=${v2.size}.")
    var squaredDistance = SparkFHE.getInstance().encrypt(0)
    (v1, v2) match {
      case (CtxtDenseVector(vv1), CtxtDenseVector(vv2)) =>
        var kv = 0
        val sz = vv1.length
        while (kv < sz) {
          val score = SparkFHE.getInstance().do_FHE_basic_op(vv1(kv), vv2(kv),
            SparkFHEConstants.FHE_SUBTRACT)
          squaredDistance = SparkFHE.getInstance().do_FHE_basic_op(squaredDistance,
            SparkFHE.getInstance().do_FHE_basic_op(score, score,
              SparkFHEConstants.FHE_MULTIPLY),
            SparkFHEConstants.FHE_ADD)
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
  private[mllib_fhe] def equals(
      v1Indices: IndexedSeq[Int],
      v1Values: Array[String],
      v2Indices: IndexedSeq[Int],
      v2Values: Array[String]): Boolean = {
    val v1Size = v1Values.length
    val v2Size = v2Values.length
    var k1 = 0
    var k2 = 0
    var allEqual = true
    while (allEqual) {
      while (k1 < v1Size && v1Values(k1) == "") k1 += 1
      while (k2 < v2Size && v2Values(k2) == "") k2 += 1

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

  /**
   * Convert new linalg type to spark.mllib type.  Light copy; only copies references
   */
  @Since("2.0.0")
  def fromML(v: newlinalg.CtxtVector): CtxtVector = v match {
    case dv: newlinalg.CtxtDenseVector =>
      CtxtDenseVector.fromML(dv)
  }
}

/**
 * A dense vector represented by a value array.
 */
@Since("1.0.0")
@SQLUserDefinedType(udt = classOf[CtxtVectorUDT])
class CtxtDenseVector @Since("1.0.0") (
    @Since("1.0.0") val values: Array[String]) extends CtxtVector {

  @Since("1.0.0")
  override def size: Int = values.length

  override def toString: String = values.mkString("[", ",", "]")

  @Since("1.0.0")
  override def toArray: Array[String] = values

  // private[spark] override def asBreeze: BV[Double] = new BDV[Double](values)

  @Since("1.0.0")
  override def apply(i: Int): String = values(i)

  @Since("1.1.0")
  override def copy: CtxtDenseVector = {
    new CtxtDenseVector(values.clone())
  }

  @Since("1.6.0")
  override def foreachActive(f: (Int, String) => Unit): Unit = {
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
      if (v != "") {
        result = 31 * result + i
        val bits = java.lang.Long.parseLong(new BigInteger(values(i).getBytes()).toString(2))
        result = 31 * result + (bits ^ (bits >>> 32)).toInt
        nnz += 1
      }
      i += 1
    }
    result
  }

  /* @Since("1.4.0")
  override def numActives: Int = size */

  @Since("1.4.0")
  override def numNonzeros: String = {
    // same as values.count(_ != 0.0) but faster
    val sv: StringVector = new StringVector()
    values.foreach { v =>
      sv.add(v)
    }
    val x = SparkFHE.getInstance().numNonzeros(sv)
    x
    /* var nnz = 0
    values.foreach { v =>
      if (v != 0.0) {
        nnz += 1
      }
    }
    nnz */
  }

  /* @Since("1.5.0")
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
  } */

  /* @Since("1.6.0")
  override def toJson: String = {
    val jValue = ("type" -> 1) ~ ("values" -> values.toSeq)
    compact(render(jValue))
  } */

  @Since("2.0.0")
  override def asML: newlinalg.CtxtDenseVector = {
    new newlinalg.CtxtDenseVector(values)
  }
}

@Since("1.3.0")
object CtxtDenseVector {

  /** Extracts the value array from a dense vector. */
  @Since("1.3.0")
  def unapply(dv: CtxtDenseVector): Option[Array[String]] = Some(dv.values)

  /**
   * Convert new linalg type to spark.mllib type.  Light copy; only copies references
   */
  @Since("2.0.0")
  def fromML(v: newlinalg.CtxtDenseVector): CtxtDenseVector = {
    new CtxtDenseVector(v.values)
  }
}

/**
 * Implicit methods available in Scala for converting
 * [[org.apache.spark.mllib_fhe.linalg.CtxtVector]]
 * to [[org.apache.spark.ml_fhe.linalg.CtxtVector]] and vice versa.
 */
private[spark] object CtxtVectorImplicits {

  implicit def mllibVectorToMLVector(v: CtxtVector): newlinalg.CtxtVector = v.asML

  implicit def mllibDenseVectorToMLDenseVector(v: CtxtDenseVector): newlinalg.CtxtDenseVector =
    v.asML

  implicit def mlVectorToMLlibVector(v: newlinalg.CtxtVector): CtxtVector = CtxtVectors.fromML(v)

  implicit def mlDenseVectorToMLlibDenseVector(v: newlinalg.CtxtDenseVector): CtxtDenseVector =
    CtxtVectors.fromML(v).asInstanceOf[CtxtDenseVector]

}
