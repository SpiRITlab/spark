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

package org.apache.spark.ml_fhe.feature

import org.apache.spark.annotation.Since
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib_fhe.feature
import org.apache.spark.mllib_fhe.feature.ElementWiseProductMatrix
import org.apache.spark.mllib_fhe.linalg.{CtxtMatrix, CtxtMatrixUDT}
import org.apache.spark.sql.types.DataType

@Since("1.4.0")
class ElementWiseProductCtxtMatrix @Since("1.4.0") (@Since("1.4.0") override val uid: String)
  extends UnaryTransformer[CtxtMatrix, CtxtMatrix,
    ElementWiseProductCtxtMatrix] with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("elemProd"))

  /**
   * the matrix to multiply with input matrix
   * @group param
   */
  @Since("2.0.0")
  val scalingMatrix: Param[CtxtMatrix] = {
    new Param(this, "scalingVec", "vector for hadamard product")
  }

  /** @group setParam */
  @Since("2.0.0")
  def setScalingMatrix(value: CtxtMatrix): this.type = set(scalingMatrix, value)

  /** @group getParam */
  @Since("2.0.0")
  def getScalingMatrix: CtxtMatrix = getOrDefault(scalingMatrix)
  /**
   * Creates the transform function using the given param map. The input param map already takes
   * account of the embedded param map. So the param values should be determined solely by the input
   * param map.
   */
  override protected def createTransformFunc: CtxtMatrix => CtxtMatrix = {
    require(params.contains(scalingMatrix), s"transformation requires a weight matrix")
    val elemScaler = new ElementWiseProductMatrix($(scalingMatrix))
    v => elemScaler.transform(v)
  }

  /**
   * Returns the data type of the output column.
   */
  override protected def outputDataType: DataType = new CtxtMatrixUDT()
}

@Since("2.0.0")
object ElementWiseProductCtxtMatrix extends DefaultParamsReadable[ElementWiseProductCtxtMatrix] {

  @Since("2.0.0")
  override def load(path: String): ElementWiseProductCtxtMatrix = super.load(path)
}

