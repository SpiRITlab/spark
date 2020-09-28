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

import spiritlab.sparkfhe.api.SparkFHE

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
  def toArray: Array[String]

  /** Gets the (i, j)-th element. */
  @Since("1.3.0")
  def apply(i: Int, j: Int): String

  /** Return the index for the (i, j)-th element in the backing array. */
  private[mllib_fhe] def index(i: Int, j: Int): Int

  /** Update element at (i, j) */
  private[mllib_fhe] def update(i: Int, j: Int, v: String): Unit

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

  def multiply(y: CtxtDenseMatrix): CtxtDenseMatrix = {
    require(numCols == y.numRows, s"The columns of this matrix doesn't match rows of y. " +
      s"this: ${numCols}, y: ${y.numRows}")

    val result = new Array[String](numRows * y.numCols)
    // SparkFHE.getInstance().dense_matrix_multiply(values, y.values)
    for (i <- 0 until numRows; j <- 0 until y.numCols; k <- 0 until numCols) {
      // result[i, j] += values[i, k] * y[k, j]
      // TODO: Using pairwise multiply could be more efficient
      result(index(i, j)) = SparkFHE.getInstance().fhe_add(result(index(i, j)),
        SparkFHE.getInstance().fhe_multiply(values(index(i, k)), y(k, j)))
    }
    CtxtMatrices.dense(numRows, y.numCols, result, false)
  }
}

object CtxtDenseMatrix {
  def unapply(arg: CtxtDenseMatrix): Option[(Int, Int, Array[String], Boolean)] = {
    Some((arg.numRows, arg.numCols, arg.values, arg.isTransposed))
  }
}

object CtxtMatrices {

  def dense(numRows: Int, numCols: Int, values: Array[String],
            isTransposed: Boolean): CtxtDenseMatrix =
    new CtxtDenseMatrix(numRows, numCols, values, isTransposed)
}
