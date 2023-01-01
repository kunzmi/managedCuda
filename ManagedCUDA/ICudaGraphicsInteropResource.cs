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
using ManagedCuda.BasicTypes;

namespace ManagedCuda
{
    /// <summary>
    /// Common interface for OpenGL and DirectX graphics interop resources
    /// </summary>
    public interface ICudaGraphicsInteropResource : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="stream"></param>
        void Map(CUstream stream);

        /// <summary>
        /// 
        /// </summary>
        void Map();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="stream"></param>
        void UnMap(CUstream stream);

        /// <summary>
        /// 
        /// </summary>
        void UnMap();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="flags"></param>
        void SetMapFlags(CUGraphicsMapResourceFlags flags);

        /// <summary>
        /// 
        /// </summary>
        void Unregister();

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        CudaDeviceVariable<T> GetMappedPointer<T>() where T : struct;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="devicePtr"></param>
        /// <param name="size"></param>
        void GetMappedPointer(out CUdeviceptr devicePtr, out SizeT size);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        CUdeviceptr GetMappedPointer();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        CudaArray1D GetMappedArray1D(uint arrayIndex, uint mipLevel);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        CudaArray2D GetMappedArray2D(uint arrayIndex, uint mipLevel);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        CudaArray3D GetMappedArray3D(uint arrayIndex, uint mipLevel);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        CudaMipmappedArray GetMappedMipmappedArray(CUArrayFormat format, CudaMipmappedArrayNumChannels numChannels);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="arrayIndex"></param>
        /// <param name="mipLevel"></param>
        /// <returns></returns>
        CUarray GetMappedCUArray(uint arrayIndex, uint mipLevel);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
		CUmipmappedArray GetMappedCUMipmappedArray();

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        CUgraphicsResource GetCUgraphicsResource();

        /// <summary>
        /// 
        /// </summary>
        void SetIsMapped();

        /// <summary>
        /// 
        /// </summary>
        void SetIsUnmapped();
    }

}
