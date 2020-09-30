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

import java.util.Random

import spiritlab.sparkfhe.api.{SparkFHE, StringVector}

import org.apache.spark.annotation.{AlphaComponent, Since}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

sealed trait CtxtMatrix extends Serializable {

  /** Number of rows. */
  @Since("1.0.0")
  def numRows: Int

  /** Number of columns. */
  @Since("1.0.0")
  def numCols: Int

  /** Flag that keeps track whether the matrix is transposed or not. False by default. */
  @Since("1.3.0")
  val isTransposed: Boolean = false

  /** Converts to a dense array in column major. */
  @Since("1.0.0")
  def toArray: Array[String] = {
    val newArray = new Array[String](numRows*numCols)
    foreachActive((i, j, v) => {
      newArray(j*numRows + i) = v
    })
    newArray
  }

  /** Gets the (i, j)-th element. */
  @Since("1.3.0")
  def apply(i: Int, j: Int): String

  /** Return the index for the (i, j)-th element in the backing array. */
  private[mllib_fhe] def index(i: Int, j: Int): Int

  /** Update element at (i, j) */
  private[mllib_fhe] def update(i: Int, j: Int, v: String): Unit


  /**
   * Update all the values of this matrix using the function f. Performed in-place on the
   * backing array. For example, an operation such as addition or subtraction will only be
   * performed on the non-zero values in a `SparseMatrix`.
   */
  private[mllib_fhe] def update(f: String => String): CtxtMatrix

  /** Get a deep copy of the matrix. */
  @Since("1.2.0")
  def copy: CtxtMatrix

  /**
   * Transpose the Matrix. Returns a new `Matrix` instance sharing the same underlying data.
   */
  @Since("1.3.0")
  def transpose: CtxtMatrix

  /**
   * Applies a function `f` to all the active elements of dense and sparse matrix. The ordering
   * of the elements are not defined.
   *
   * @param f the function takes three parameters where the first two parameters are the row
   *          and column indices respectively with the type `Int`, and the final parameter is the
   *          corresponding value in the matrix with type `Double`.
   */
  private[spark] def foreachActive(f: (Int, Int, String) => Unit)

  /**
   * Method for matrix - dense matrix multiplication
   * @param y: Ctxt Dense Matrix
   * @return resulting dense matrix
   */
  def multiply(y: CtxtDenseMatrix): CtxtDenseMatrix = {
    require(numCols == y.numRows, s"The columns of this matrix doesn't match rows of y. " +
      s"this: ${numCols}, y: ${y.numRows}")

    val C = CtxtDenseMatrix.zeros(numRows, y.numCols)
    BLAS_FHE.gemm(1.0, this, y, 0.0, C)
    C
  }

  private[spark] def map(f: String => String): CtxtMatrix

  /**
   * Find the number of non-zero active values.
   */
  @Since("1.5.0")
  def numNonzeros: String

  /**
   * Find the number of values stored explicitly. These values can be zero as well.
   */
  @Since("1.5.0")
  def numActives: Int

}

@AlphaComponent
private[spark] class CtxtMatrixUDT extends UserDefinedType[CtxtMatrix] {
  /** Underlying storage type for this UDT */
  override def sqlType: StructType = {
    StructType(Seq(
      StructField("type", ByteType, nullable = false),
      StructField("numRows", IntegerType, nullable = false),
      StructField("numCols", IntegerType, nullable = false),
      StructField("isTransposed", BooleanType, nullable = false),
      StructField("values", ArrayType(IntegerType, containsNull = false)),
      StructField("colPtrs", ArrayType(IntegerType, containsNull = false)),
      StructField("rowIndices", ArrayType(IntegerType, containsNull = false))
    ))
  }

  /**
   * Convert the user type to a SQL datum
   */
  override def serialize(obj: CtxtMatrix): InternalRow = {
    obj match {
      case CtxtDenseMatrix(numRows, numCols, values: Array[String], isTransposed) =>
        val row = new GenericInternalRow(size = 7)
        row.setByte(0, 1)
        row.setInt(1, numRows)
        row.setInt(2, numCols)
        row.setBoolean(3, isTransposed)
        row.update(4, ArrayData.toArrayData(values.map {UTF8String.fromString}))
        row.setNullAt(5)
        row.setNullAt(6)
        row
    }
  }

  /** Convert a SQL datum to the user type */
  override def deserialize(datum: Any): CtxtMatrix = {
    datum match {
      case row: InternalRow =>
        require(row.numFields == 7, s"CtxtMatrixUDT.deserialize given row with " +
          s"length ${row.numFields} but requires length == 7")
        row.getByte(0) match {
          case 1 =>
            val numRows = row.getInt(1)
            val numCols = row.getInt(2)
            val isTransposed = row.getBoolean(3)
            val values = row.getArray(4).toArray[UTF8String](StringType).map(x => x.toString)
            CtxtMatrices.dense(numRows, numCols, values, isTransposed)
        }
    }
  }

  /**
   * Class object for the UserType
   */
  override def userClass: Class[CtxtMatrix] = classOf[CtxtMatrix]
}

@SQLUserDefinedType(udt = classOf[CtxtMatrixUDT])
class CtxtDenseMatrix @Since("1.3.0") (
        @Since("1.0.0") val numRows: Int,
        @Since("1.0.0") val numCols: Int,
        @Since("1.0.0") val values: Array[String],
        @Since("1.3.0") override val isTransposed: Boolean) extends CtxtMatrix {

  require(values.length == numRows * numCols, "The number of values supplied doesn't match the " +
    s"size of the matrix! values.length: ${values.length}, numRows * numCols: ${numRows * numCols}")

  @Since("1.0.0")
  def this(numRows: Int, numCols: Int, values: Array[String]) =
    this(numRows, numCols, values, false)

  /** Converts to a dense array in column major. */
  override def toArray: Array[String] = values

  /** Gets the (i, j)-th element. */
  override def apply(i: Int, j: Int): String = values(index(i, j))

  /** Return the index for the (i, j)-th element in the backing array. */
  override private[mllib_fhe] def index(i: Int, j: Int): Int = {
    require(i >= 0 && i < numRows, s"Expected 0 <= i < $numRows, got i = $i.")
    require(j >= 0 && j < numCols, s"Expected 0 <= j < $numCols, got j = $j.")
    if (!isTransposed) i + numRows * j else j + numCols * i
  }

  /** Update element at (i, j) */
  override private[mllib_fhe] def update(i: Int, j: Int, v: String): Unit = {
    values(index(i, j)) = v
  }

  /**
   * Applies a function `f` to all the active elements of dense and sparse matrix. The ordering
   * of the elements are not defined.
   *
   * @param f the function takes three parameters where the first two parameters are the row
   *          and column indices respectively with the type `Int`, and the final parameter is the
   *          corresponding value in the matrix with type `Double`.
   */
  override private[spark] def foreachActive(f: (Int, Int, String) => Unit): Unit = {
    if (!isTransposed) {
      for (j <- 0 until numCols) {
        for (i <- 0 until numRows) {
          f(i, j, values(j*numRows + i))
        }
      }
    } else {
      for (i <- 0 until numRows) {
        for (j <- 0 until numCols) {
          f(i, j, values(i*numCols + j))
        }
      }
    }
  }

  /** Get a deep copy of the matrix. */
  override def copy: CtxtMatrix = {
    new CtxtDenseMatrix(numRows, numCols, values.clone(), isTransposed)
  }

  /**
   * Transpose the Matrix. Returns a new `Matrix` instance sharing the same underlying data.
   */
  override def transpose: CtxtMatrix = {
    new CtxtDenseMatrix(numCols, numRows, values, !isTransposed)
  }

  override private[spark] def map(f: String => String) = {
    new CtxtDenseMatrix(numRows, numCols, values.map(f), isTransposed)
  }

  /**
   * Update all the values of this matrix using the function f. Performed in-place on the
   * backing array. For example, an operation such as addition or subtraction will only be
   * performed on the non-zero values in a `SparseMatrix`.
   */
  override private[mllib_fhe] def update(f: String => String): CtxtDenseMatrix = {
    for (i <- values.indices) {
      values(i) = f(values(i))
    }
    this
  }

  /**
   * Find the number of non-zero active values.
   */
  override def numNonzeros: String = {
    val vector = new StringVector(values)
    SparkFHE.getInstance().numNonzeros(vector)
  }

  /**
   * Find the number of values stored explicitly. These values can be zero as well.
   */
  override def numActives: Int = values.length
}

object CtxtDenseMatrix {
  def unapply(arg: CtxtDenseMatrix): Option[(Int, Int, Array[String], Boolean)] = {
    Some((arg.numRows, arg.numCols, arg.values, arg.isTransposed))
  }

  def zeros(numRows: Int, numCols: Int): CtxtDenseMatrix = {
    val zero = SparkFHE.getInstance().encrypt(SparkFHE.getInstance().encode("0")).toString
    val values = new Array[String](numRows*numCols)
    for (i <- values.indices) {
      values(i) = zero
    }
    CtxtMatrices.dense(numRows, numCols, values)
  }

  /**
   * Generate a `DenseMatrix` consisting of ones.
   * @param numRows number of rows of the matrix
   * @param numCols number of columns of the matrix
   * @return `DenseMatrix` with size `numRows` x `numCols` and values of ones
   */
  @Since("1.3.0")
  def ones(numRows: Int, numCols: Int): CtxtDenseMatrix = {
    val one = SparkFHE.getInstance().encrypt(SparkFHE.getInstance().encode("1")).toString
    new CtxtDenseMatrix(numRows, numCols, Array.fill(numRows*numCols)(one))
  }

  /**
   * Generate an Identity Matrix in `DenseMatrix` format.
   * @param n number of rows and columns of the matrix
   * @return `DenseMatrix` with size `n` x `n` and values of ones on the diagonal
   */
  @Since("1.3.0")
  def eye(n: Int): CtxtDenseMatrix = {
    val identity = CtxtDenseMatrix.zeros(n, n)
    val one = SparkFHE.getInstance().encrypt(SparkFHE.getInstance().encode("1")).toString

    for (i <- 0 until n) {
      identity.update(i, i, one)
    }

    identity
  }

  /**
   * Generate a `DenseMatrix` consisting of `i.i.d.` uniform random numbers.
   * @param numRows number of rows of the matrix
   * @param numCols number of columns of the matrix
   * @param rng a random number generator
   * @return `DenseMatrix` with size `numRows` x `numCols` and values in U(0, 1)
   */
  @Since("1.3.0")
  def rand(numRows: Int, numCols: Int, rng: Random): CtxtDenseMatrix = {
    new CtxtDenseMatrix(numRows, numCols,
      Array.fill(numRows*numCols)
      (SparkFHE.getInstance().encrypt(SparkFHE.getInstance().encode(rng.nextDouble())).toString))
  }

  /**
   * Generate a `DenseMatrix` consisting of `i.i.d.` gaussian random numbers.
   * @param numRows number of rows of the matrix
   * @param numCols number of columns of the matrix
   * @param rng a random number generator
   * @return `DenseMatrix` with size `numRows` x `numCols` and values in N(0, 1)
   */
  @Since("1.3.0")
  def randn(numRows: Int, numCols: Int, rng: Random): CtxtDenseMatrix = {
    new CtxtDenseMatrix(numRows, numCols,
      Array.fill(numRows*numCols)
      (SparkFHE.getInstance().encrypt(SparkFHE.getInstance().encode(rng.nextGaussian())).toString))
  }

  /**
   * Generate a diagonal matrix in `DenseMatrix` format from the supplied values.
   * @param vector a `Vector` that will form the values on the diagonal of the matrix
   * @return Square `DenseMatrix` with size `values.length` x `values.length` and `values`
   *         on the diagonal
   */
  @Since("1.3.0")
  def diag(vector: CtxtVector): CtxtDenseMatrix = {
    val n = vector.size
    val matrix = CtxtDenseMatrix.zeros(n, n)
    val values = vector.toArray
    for (i <- 0 until n) {
      matrix.update(i, i, values(i))
    }
    matrix
  }
}

object CtxtMatrices {

  def dense(numRows: Int, numCols: Int, values: Array[String]): CtxtDenseMatrix =
    new CtxtDenseMatrix(numRows, numCols, values)

  def dense(numRows: Int, numCols: Int, values: Array[String],
            isTransposed: Boolean): CtxtDenseMatrix = {
    new CtxtDenseMatrix(numRows, numCols, values, isTransposed)
  }

  /**
   * Generate a `Matrix` consisting of zeros.
   * @param numRows number of rows of the matrix
   * @param numCols number of columns of the matrix
   * @return `Matrix` with size `numRows` x `numCols` and values of zeros
   */
  @Since("1.2.0")
  def zeros(numRows: Int, numCols: Int): CtxtMatrix = CtxtDenseMatrix.zeros(numRows, numCols)

  /**
   * Generate a `DenseMatrix` consisting of ones.
   * @param numRows number of rows of the matrix
   * @param numCols number of columns of the matrix
   * @return `Matrix` with size `numRows` x `numCols` and values of ones
   */
  @Since("1.2.0")
  def ones(numRows: Int, numCols: Int): CtxtMatrix = CtxtDenseMatrix.ones(numRows, numCols)

  /**
   * Generate a dense Identity Matrix in `Matrix` format.
   * @param n number of rows and columns of the matrix
   * @return `Matrix` with size `n` x `n` and values of ones on the diagonal
   */
  @Since("1.2.0")
  def eye(n: Int): CtxtMatrix = CtxtDenseMatrix.eye(n)

  /**
   * Generate a `DenseMatrix` consisting of `i.i.d.` uniform random numbers.
   * @param numRows number of rows of the matrix
   * @param numCols number of columns of the matrix
   * @param rng a random number generator
   * @return `Matrix` with size `numRows` x `numCols` and values in U(0, 1)
   */
  @Since("1.2.0")
  def rand(numRows: Int, numCols: Int, rng: Random): CtxtMatrix =
    CtxtDenseMatrix.rand(numRows, numCols, rng)

  /**
   * Generate a `DenseMatrix` consisting of `i.i.d.` gaussian random numbers.
   * @param numRows number of rows of the matrix
   * @param numCols number of columns of the matrix
   * @param rng a random number generator
   * @return `Matrix` with size `numRows` x `numCols` and values in N(0, 1)
   */
  @Since("1.2.0")
  def randn(numRows: Int, numCols: Int, rng: Random): CtxtMatrix =
    CtxtDenseMatrix.randn(numRows, numCols, rng)

  /**
   * Generate a diagonal matrix in `Matrix` format from the supplied values.
   * @param vector a `Vector` that will form the values on the diagonal of the matrix
   * @return Square `Matrix` with size `values.length` x `values.length` and `values`
   *         on the diagonal
   */
  @Since("1.2.0")
  def diag(vector: CtxtVector): CtxtMatrix = CtxtDenseMatrix.diag(vector)
}
