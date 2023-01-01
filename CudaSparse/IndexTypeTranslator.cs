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
