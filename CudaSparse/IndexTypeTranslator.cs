//	Copyright (c) 2020, Michael Kunz. All rights reserved.
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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda.CudaSparse
{

    /// <summary>
    /// Translates from IndexType to .net type and vice versa
    /// </summary>	
    public static class IndexTypeTranslator
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="indexType"></param>
        /// <returns></returns>
		public static Type GetType(IndexType indexType)
        {
            switch (indexType)
            {
                case IndexType.Index16U:
                    return typeof(ushort);
                case IndexType.Index32I:
                    return typeof(int);
                case IndexType.Index64I:
                    return typeof(long);
                default:
                    throw new ArgumentException("Unsupported IndexType: " + indexType.ToString());
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static IndexType GetType(Type type)
        {
            if (type == typeof(ushort))
                return IndexType.Index16U;
            if (type == typeof(int))
                return IndexType.Index32I;
            if (type == typeof(long))
                return IndexType.Index64I;

            throw new ArgumentException("Cannot translate " + type.ToString() + " into an indexType");
        }

    }

}
