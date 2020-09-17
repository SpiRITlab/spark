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

import org.apache.spark.annotation.Since
import spiritlab.sparkfhe.api.{SparkFHE, SparkFHEConstants}

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
    for (i <- 0 until numRows; j <- 0 until y.numCols; k <- 0 until numCols) {
      result(index(i, j)) = SparkFHE.getInstance().do_FHE_basic_op(result(index(i, j)),
      SparkFHE.getInstance().do_FHE_basic_op(values(index(i, k)), y(k, j),
        SparkFHEConstants.FHE_MULTIPLY),
        SparkFHEConstants.FHE_ADD)
    }
    CtxtMatrices.dense(numRows, y.numCols, result, false)
  }
}

object CtxtMatrices {

  def dense(numRows: Int, numCols: Int, values: Array[String],
            isTransposed: Boolean): CtxtDenseMatrix =
    new CtxtDenseMatrix(numRows, numCols, values, isTransposed)
}
