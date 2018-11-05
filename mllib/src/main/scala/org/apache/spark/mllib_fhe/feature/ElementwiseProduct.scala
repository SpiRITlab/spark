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

import sparkfhe.{SparkFHE, SparkFHEConstants}

import org.apache.spark.annotation.Since
import org.apache.spark.mllib_fhe.linalg._

/**
 * Outputs the Hadamard product (i.e., the element-wise product) of each input vector with a
 * provided "weight" vector. In other words, it scales each column of the dataset by a scalar
 * multiplier.
 * @param scalingVec The values used to scale the reference vector's individual components.
 */
@Since("1.4.0")
class ElementwiseProduct @Since("1.4.0")(
    @Since("1.4.0") val scalingVec: CtxtVector) extends CtxtVectorTransformer {

  /**
   * Does the hadamard product transformation.
   *
   * @param vector vector to be transformed.
   * @return transformed vector.
   */
  @Since("1.4.0")
  override def transform(vector: CtxtVector): CtxtVector = {
    require(vector.size == scalingVec.size,
      s"vector sizes do not match: Expected ${scalingVec.size} but found ${vector.size}")

    // scalastyle:off println
    System.out.println(SparkFHE.do_basic_op(2, 6, SparkFHEConstants.ADD))
    // scalastyle:on println
    vector match {
      case dv: CtxtDenseVector =>
        val values: Array[String] = dv.values.clone()
        val dim = scalingVec.size
        var i = 0
        while (i < dim) {
          values(i) = (Integer.parseInt(values(i)) * Integer.parseInt(scalingVec(i))).toString
          i += 1
        }
        CtxtVectors.dense(values)
      case v => throw new IllegalArgumentException("Does not support vector type " + v.getClass)
    }
  }


}
