// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
// http://kunzmi.github.io/managedCuda
//
// This file is part of ManagedCuda.
//
// Commercial License Usage
//  Licensees holding valid commercial ManagedCuda licenses may use this
//  file in accordance with the commercial license agreement provided with
//  the Software or, alternatively, in accordance with the terms contained
//  in a written agreement between you and Artic Imaging SARL. For further
//  information contact us at managedcuda@articimaging.eu.
//  
// GNU General Public License Usage
//  Alternatively, this file may be used under the terms of the GNU General
//  Public License as published by the Free Software Foundation, either 
//  version 3 of the License, or (at your option) any later version.
//  
//  ManagedCuda is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <http://www.gnu.org/licenses/>.


using System;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.BasicTypes
{

    /// <summary>
    /// Translates from CudaDataType to .net type and vice versa
    /// </summary>	
    public static class CudaDataTypeTranslator
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="cudaType"></param>
        /// <returns></returns>
		public static Type GetType(cudaDataType cudaType)
        {
            switch (cudaType)
            {
                case cudaDataType.CUDA_R_16F:
                    return typeof(half);
                case cudaDataType.CUDA_C_16F:
                    return typeof(half2);
                case cudaDataType.CUDA_R_32F:
                    return typeof(float);
                case cudaDataType.CUDA_C_32F:
                    return typeof(cuFloatComplex);
                case cudaDataType.CUDA_R_64F:
                    return typeof(double);
                case cudaDataType.CUDA_C_64F:
                    return typeof(cuDoubleComplex);
                case cudaDataType.CUDA_R_8I:
                    return typeof(sbyte);
                case cudaDataType.CUDA_C_8I:
                    return typeof(char2);
                case cudaDataType.CUDA_R_8U:
                    return typeof(byte);
                case cudaDataType.CUDA_C_8U:
                    return typeof(uchar2);
                case cudaDataType.CUDA_R_16BF:
                    return typeof(bfloat16);
                case cudaDataType.CUDA_C_16BF:
                    return typeof(bfloat162);
                //case cudaDataType.CUDA_R_4I:
                //    return typeof(int);
                //case cudaDataType.CUDA_C_4I:
                //    return typeof(int2);
                //case cudaDataType.CUDA_R_4U:
                //    return typeof(uint);
                //case cudaDataType.CUDA_C_4U:
                //    return typeof(uint2);
                case cudaDataType.CUDA_R_16I:
                    return typeof(short);
                case cudaDataType.CUDA_C_16I:
                    return typeof(short2);
                case cudaDataType.CUDA_R_16U:
                    return typeof(ushort);
                case cudaDataType.CUDA_C_16U:
                    return typeof(ushort2);
                case cudaDataType.CUDA_R_32I:
                    return typeof(int);
                case cudaDataType.CUDA_C_32I:
                    return typeof(int2);
                case cudaDataType.CUDA_R_32U:
                    return typeof(uint);
                case cudaDataType.CUDA_C_32U:
                    return typeof(uint2);
                case cudaDataType.CUDA_R_64I:
                    return typeof(long);
                case cudaDataType.CUDA_C_64I:
                    return typeof(long2);
                case cudaDataType.CUDA_R_64U:
                    return typeof(ulong);
                case cudaDataType.CUDA_C_64U:
                    return typeof(ulong2);
                default:
                    throw new ArgumentException("Unsupported cuda type: " + cudaType.ToString());
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="cudaType"></param>
        /// <returns></returns>
		public static int GetSize(cudaDataType cudaType)
        {
            switch (cudaType)
            {
                case cudaDataType.CUDA_R_16F:
                    return 2;
                case cudaDataType.CUDA_C_16F:
                    return 4;
                case cudaDataType.CUDA_R_32F:
                    return 4;
                case cudaDataType.CUDA_C_32F:
                    return 8;
                case cudaDataType.CUDA_R_64F:
                    return 8;
                case cudaDataType.CUDA_C_64F:
                    return 16;
                case cudaDataType.CUDA_R_8I:
                    return 1;
                case cudaDataType.CUDA_C_8I:
                    return 2;
                case cudaDataType.CUDA_R_8U:
                    return 1;
                case cudaDataType.CUDA_C_8U:
                    return 2;
                case cudaDataType.CUDA_R_16BF:
                    return 2;
                case cudaDataType.CUDA_C_16BF:
                    return 4;
                //case cudaDataType.CUDA_R_4I:
                //    return typeof(int);
                //case cudaDataType.CUDA_C_4I:
                //    return typeof(int2);
                //case cudaDataType.CUDA_R_4U:
                //    return typeof(uint);
                //case cudaDataType.CUDA_C_4U:
                //    return typeof(uint2);
                case cudaDataType.CUDA_R_16I:
                    return 2;
                case cudaDataType.CUDA_C_16I:
                    return 4;
                case cudaDataType.CUDA_R_16U:
                    return 2;
                case cudaDataType.CUDA_C_16U:
                    return 4;
                case cudaDataType.CUDA_R_32I:
                    return 4;
                case cudaDataType.CUDA_C_32I:
                    return 8;
                case cudaDataType.CUDA_R_32U:
                    return 4;
                case cudaDataType.CUDA_C_32U:
                    return 8;
                case cudaDataType.CUDA_R_64I:
                    return 8;
                case cudaDataType.CUDA_C_64I:
                    return 16;
                case cudaDataType.CUDA_R_64U:
                    return 8;
                case cudaDataType.CUDA_C_64U:
                    return 16;
                default:
                    throw new ArgumentException("Unsupported cuda type: " + cudaType.ToString());
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static cudaDataType GetType(Type type)
        {
            if (type == typeof(half))
                return cudaDataType.CUDA_R_16F;
            if (type == typeof(half2))
                return cudaDataType.CUDA_C_16F;
            if (type == typeof(float))
                return cudaDataType.CUDA_R_32F;
            if (type == typeof(cuFloatComplex))
                return cudaDataType.CUDA_C_32F;
            if (type == typeof(double))
                return cudaDataType.CUDA_R_64F;
            if (type == typeof(cuDoubleComplex))
                return cudaDataType.CUDA_C_64F;
            if (type == typeof(sbyte))
                return cudaDataType.CUDA_R_8I;
            if (type == typeof(char2))
                return cudaDataType.CUDA_C_8I;
            if (type == typeof(byte))
                return cudaDataType.CUDA_R_8U;
            if (type == typeof(uchar2))
                return cudaDataType.CUDA_C_8U;
            if (type == typeof(bfloat16))
                return cudaDataType.CUDA_R_16BF;
            if (type == typeof(bfloat162))
                return cudaDataType.CUDA_C_16BF;
            //if (type == typeof(int))
            //return cudaDataType.CUDA_R_4I;
            //if (type == typeof(int2))
            //return cudaDataType.CUDA_C_4I;
            //if (type == typeof(uint))
            //return cudaDataType.CUDA_R_4U;
            //if (type == typeof(uint2))
            //return cudaDataType.CUDA_C_4U;
            if (type == typeof(short))
                return cudaDataType.CUDA_R_16I;
            if (type == typeof(short2))
                return cudaDataType.CUDA_C_16I;
            if (type == typeof(ushort))
                return cudaDataType.CUDA_R_16U;
            if (type == typeof(ushort2))
                return cudaDataType.CUDA_C_16U;
            if (type == typeof(int))
                return cudaDataType.CUDA_R_32I;
            if (type == typeof(int2))
                return cudaDataType.CUDA_C_32I;
            if (type == typeof(uint))
                return cudaDataType.CUDA_R_32U;
            if (type == typeof(uint2))
                return cudaDataType.CUDA_C_32U;
            if (type == typeof(long))
                return cudaDataType.CUDA_R_64I;
            if (type == typeof(long2))
                return cudaDataType.CUDA_C_64I;
            if (type == typeof(ulong))
                return cudaDataType.CUDA_R_64U;
            if (type == typeof(ulong2))
                return cudaDataType.CUDA_C_64U;

            throw new ArgumentException("Cannot translate " + type.ToString() + " into a cudaDataType");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static int GetSize(Type type)
        {
            if (type == typeof(half))
                return 2;
            if (type == typeof(half2))
                return 4;
            if (type == typeof(float))
                return 4;
            if (type == typeof(cuFloatComplex))
                return 8;
            if (type == typeof(double))
                return 8;
            if (type == typeof(cuDoubleComplex))
                return 16;
            if (type == typeof(sbyte))
                return 1;
            if (type == typeof(char2))
                return 2;
            if (type == typeof(byte))
                return 1;
            if (type == typeof(uchar2))
                return 2;
            if (type == typeof(bfloat16))
                return 2;
            if (type == typeof(bfloat162))
                return 4;
            //if (type == typeof(int))
            //return cudaDataType.CUDA_R_4I;
            //if (type == typeof(int2))
            //return cudaDataType.CUDA_C_4I;
            //if (type == typeof(uint))
            //return cudaDataType.CUDA_R_4U;
            //if (type == typeof(uint2))
            //return cudaDataType.CUDA_C_4U;
            if (type == typeof(short))
                return 2;
            if (type == typeof(short2))
                return 4;
            if (type == typeof(ushort))
                return 2;
            if (type == typeof(ushort2))
                return 4;
            if (type == typeof(int))
                return 4;
            if (type == typeof(int2))
                return 8;
            if (type == typeof(uint))
                return 4;
            if (type == typeof(uint2))
                return 8;
            if (type == typeof(long))
                return 8;
            if (type == typeof(long2))
                return 16;
            if (type == typeof(ulong))
                return 8;
            if (type == typeof(ulong2))
                return 16;

            throw new ArgumentException("Cannot translate " + type.ToString() + " into a cudaDataType");
        }

    }

}
