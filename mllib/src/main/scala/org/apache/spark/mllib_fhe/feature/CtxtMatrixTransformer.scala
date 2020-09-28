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

package org.apache.spark.mllib_fhe.feature

import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib_fhe.linalg.CtxtMatrix
import org.apache.spark.rdd.RDD

trait CtxtMatrixTransformer {
  /**
   * Applies transformation on a vector.
   *
   * @param matrix vector to be transformed.
   * @return transformed vector.
   */
  @Since("1.1.0")
  def transform(matrix: CtxtMatrix): CtxtMatrix

  /**
   * Applies transformation on an RDD[Vector].
   *
   * @param data RDD[Vector] to be transformed.
   * @return transformed RDD[Vector].
   */
  @Since("1.1.0")
  def transform(data: RDD[CtxtMatrix]): RDD[CtxtMatrix] = {
    // Later in #1498 , all RDD objects are sent via broadcasting instead of RPC.
    // So it should be no longer necessary to explicitly broadcast `this` object.
    data.map(x => this.transform(x))
  }

  /**
   * Applies transformation on a JavaRDD[Vector].
   *
   * @param data JavaRDD[Vector] to be transformed.
   * @return transformed JavaRDD[Vector].
   */
  @Since("1.1.0")
  def transform(data: JavaRDD[CtxtMatrix]): JavaRDD[CtxtMatrix] = {
    transform(data.rdd)
  }
}
