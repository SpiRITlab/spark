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

import org.apache.spark.internal.Logging
import spiritlab.sparkfhe.api.{SparkFHE, StringVector}

private[spark] object BLAS_FHE extends Serializable with Logging{
  /**
   * C := alpha * A * B + beta * C
   * @param alpha a scalar to scale the multiplication A * B.
   * @param A the matrix A that will be left multiplied to B. Size of m x k.
   * @param B the matrix B that will be left multiplied by A. Size of k x n.
   * @param beta a scalar that can be used to scale matrix C.
   * @param C the resulting matrix C. Size of m x n. C.isTransposed must be false.
   */
  def gemm(alpha: Double,
            A: CtxtMatrix,
            B: CtxtDenseMatrix,
            beta: Double,
            C: CtxtDenseMatrix): Unit = {
    require(!C.isTransposed,
      "The matrix C cannot be the product of a transpose() call. C.isTransposed must be false.")
    if (alpha == 0.0 && beta == 1.0) {
      logDebug("gemm: alpha is equal to 0 and beta is equal to 1. Returning C.")
    } else if (alpha == 0.0) {
//      f2jBLAS.dscal(C.values.length, beta, C.values, 1)
    } else {
      A match {
//        case sparse: SparseMatrix => gemm(alpha, sparse, B, beta, C)
        case dense: CtxtDenseMatrix => gemm(alpha, dense, B, beta, C)
        case _ =>
          throw new IllegalArgumentException(s"gemm doesn't support matrix type ${A.getClass}.")
      }
    }
  }

  /**
   * C := alpha * A * B + beta * C
   * For `DenseMatrix` A.
   */
  private def gemm(alpha: Double,
                   A: CtxtDenseMatrix,
                   B: CtxtDenseMatrix,
                   beta: Double,
                   C: CtxtDenseMatrix): Unit = {
    val tAstr = if (A.isTransposed) "T" else "N"
    val tBstr = if (B.isTransposed) "T" else "N"
    val lda = if (!A.isTransposed) A.numRows else A.numCols
    val ldb = if (!B.isTransposed) B.numRows else B.numCols

    require(A.numCols == B.numRows,
      s"The columns of A don't match the rows of B. A: ${A.numCols}, B: ${B.numRows}")
    require(A.numRows == C.numRows,
      s"The rows of C don't match the rows of A. C: ${C.numRows}, A: ${A.numRows}")
    require(B.numCols == C.numCols,
      s"The columns of C don't match the columns of B. C: ${C.numCols}, A: ${B.numCols}")
    val resultStringVector = new StringVector(C.values)

    SparkFHE.getInstance().fhe_dgemm(tAstr, tBstr, A.numRows, B.numCols, A.numCols, alpha,
      new StringVector(A.values), lda, new StringVector(B.values), ldb, beta,
      resultStringVector, C.numRows)

    for (i <- C.values.indices) {
      C.values(i) = resultStringVector.get(i)
    }
  }
}
