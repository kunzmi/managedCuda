//	Copyright (c) 2012, Michael Kunz. All rights reserved.
//	http://kunzmi.github.io/managedCuda
//
//	This file is part of ManagedCuda.
//
//	ManagedCuda is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as 
//	published by the Free Software Foundation, either version 2.1 of the 
//	License, or (at your option) any later version.
//
//	ManagedCuda is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.


using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda
{
    /// <summary>
    /// Helper methods used in the wrapper framework
    /// </summary>
    public static class CudaHelperMethods
    {
        /// <summary>
        /// Returns the number of channels used in textures depending on the given type.
        /// </summary>
        /// <param name="value">Type</param>
        /// <returns>Number of channels</returns>
        public static int GetNumChannels(System.Type value)
        {
            if (value == typeof(char4) || value == typeof(uchar4) || value == typeof(short4) || value == typeof(ushort4) || value == typeof(int4) ||
                value == typeof(uint4) || value == typeof(long4) || value == typeof(ulong4) || value == typeof(float4))
                return 4;

            if (value == typeof(char2) || value == typeof(uchar2) || value == typeof(short2) || value == typeof(ushort2) || value == typeof(int2) ||
                value == typeof(uint2) || value == typeof(long2) || value == typeof(ulong2) || value == typeof(float2) || value == typeof(double2) ||
                value == typeof(cuFloatComplex) || value == typeof(cuDoubleComplex))
                return 2;
            
            if (value == typeof(byte) || value == typeof(sbyte) || value == typeof(short) || value == typeof(ushort) || value == typeof(int) || value == typeof(uint) ||
                value == typeof(long) || value == typeof(ulong) || value == typeof(float) || value == typeof(double) ||
                value == typeof(char1) || value == typeof(uchar1) || value == typeof(short1) || value == typeof(ushort1) || value == typeof(int1) ||
                value == typeof(uint1) || value == typeof(long1) || value == typeof(ulong1) || value == typeof(float1) || value == typeof(double1) ||
                value == typeof(cuFloatReal) || value == typeof(cuDoubleReal))
                return 1;

            throw new ArgumentException("Argument type must either be of 1, 2 or 4 channels. E.g. float1, float2, float4");
        }


        /// <summary>
        /// Returns the channel size of an CUDA array in bytes.
        /// </summary>
        /// <param name="format">Channel format</param>
        /// <returns>Size in bytes</returns>
        public static uint GetChannelSize(CUArrayFormat format)
        {
            uint result = 0;
            switch (format)
            {
                case CUArrayFormat.Float:
                    result = sizeof(float);
                    break;
                case CUArrayFormat.Half:
                    result = sizeof(short);
                    break;
                case CUArrayFormat.UnsignedInt8:
                    result = sizeof(byte);
                    break;
                case CUArrayFormat.UnsignedInt16:
                    result = sizeof(ushort);
                    break;
                case CUArrayFormat.UnsignedInt32:
                    result = sizeof(uint);
                    break;
                case CUArrayFormat.SignedInt8:
                    result = sizeof(sbyte);
                    break;
                case CUArrayFormat.SignedInt16:
                    result = sizeof(short);
                    break;
                case CUArrayFormat.SignedInt32:
                    result = sizeof(int);
                    break;
                default:
                    throw new CudaException(CUResult.ErrorInvalidValue, "Unknown CUArrayFormat format", null);

            }

            return result;
        }
    }
}
