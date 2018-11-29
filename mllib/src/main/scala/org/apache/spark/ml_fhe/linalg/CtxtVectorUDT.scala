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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{GenericInternalRow, UnsafeArrayData}
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

/**
 * User-defined type for [[CtxtVector]] in [[org.apache.spark.mllib-local]] which allows
 * easy interaction with SQL via [[org.apache.spark.sql.Dataset]].
 */
private[spark] class CtxtVectorUDT extends UserDefinedType[CtxtVector] {

  override final def sqlType: StructType = _sqlType

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
          s"VectorUDT.deserialize given row with length ${row.numFields} but requires length == 4")
        val tpe = row.getByte(0)
        tpe match {
          case 1 =>
            val values = row.getArray(3).toArray[UTF8String](StringType).map { x => x.toString }
            new CtxtDenseVector(values)
        }
    }
  }

  override def pyUDT: String = "pyspark.ml.linalg.CtxtVectorUDT"

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

  private[this] val _sqlType = {
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
}
