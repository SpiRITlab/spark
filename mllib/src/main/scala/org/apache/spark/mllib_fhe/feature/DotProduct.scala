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

import spiritlab.sparkfhe.api.{SparkFHE, SparkFHEConstants}

import org.apache.spark.annotation.Since
import org.apache.spark.mllib_fhe.linalg._

/**
 * Outputs the Dot product of the input vector with a provided second vector.
 * In other words, it applies element-wise product then aggregates all elements in the result vector.
 *
 * @param secondVec The values used to scale the reference vector's individual components.
 */
@Since("1.4.0")
class DotProduct @Since("1.4.0")
  (@Since("1.4.0") val secondVec: CtxtVector) extends CtxtVectorTransformer {

  /**
   * Does the dot product transformation.
   *
   * @param vector vector to be transformed.
   * @return transformed vector, where the last element is the dot product result.
   */
  @Since("1.4.0")
  override def transform(vector: CtxtVector): CtxtVector = {
    require(vector.size == secondVec.size,
      s"vector sizes do not match: Expected ${secondVec.size} but found ${vector.size}")
    vector match {
      case dv: CtxtDenseVector =>
        val values: Array[String] = dv.values.clone()
        val dim = secondVec.size
        var i = 0
        while (i < dim) {
          values(i) = SparkFHE.getInstance().do_FHE_basic_op(values(i), secondVec(i),
            SparkFHEConstants.FHE_MULTIPLY)
          if (i > 0) { // add the current result to the sum of total results
            values(i) = SparkFHE.getInstance().do_FHE_basic_op(values(i), values(i - 1),
              SparkFHEConstants.FHE_ADD)
          }
          i += 1
        }
        CtxtVectors.dense(values)
      case v => throw new IllegalArgumentException("Does not support vector type " + v.getClass)
    }
  }

}
