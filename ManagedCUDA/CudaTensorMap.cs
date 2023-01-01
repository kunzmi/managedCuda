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
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Tensor map descriptor.
    /// </summary>
    public class CudaTensorMap
    {
        private CUtensorMap _tensormap;
        private CUResult res;

        #region Constructors
        /// <summary>
        /// Create a tensor map descriptor object representing tiled memory region<para/>
        /// Creates a descriptor for Tensor Memory Access(TMA) object specified by the parameters describing a tiled region and returns it in \p tensorMap.<para/>
        /// Tensor map objects are only supported on devices of compute capability 9.0 or higher.
        /// Additionally, a tensor map object is an opaque value, and, as such, should only be accessed through CUDA API calls.
        /// </summary>
        /// <param name="tensorDataType">Tensor data type</param>
        /// <param name="tensorRank">Dimensionality of tensor</param>
        /// <param name="globalAddress">Starting address of memory region described by tensor</param>
        /// <param name="globalDim">Array containing tensor size (number of elements) along each of the \p tensorRank dimensions</param>
        /// <param name="globalStrides">Array containing stride size (in bytes) along each of the \p tensorRank - 1 dimensions</param>
        /// <param name="boxDim">Array containing traversal box size (number of elments) along each of the \p tensorRank dimensions. Specifies how many elements to be traversed along each tensor dimension.</param>
        /// <param name="elementStrides">Array containing traversal stride in each of the \p tensorRank dimensions</param>
        /// <param name="interleave">Type of interleaved layout the tensor addresses</param>
        /// <param name="swizzle">Bank swizzling pattern inside shared memory</param>
        /// <param name="l2Promotion">L2 promotion size</param>
        /// <param name="oobFill">Indicate whether zero or special NaN constant must be used to fill out-of-bound elements</param>
        public CudaTensorMap(CUtensorMapDataType tensorDataType, uint tensorRank, CUdeviceptr globalAddress, ulong[] globalDim, ulong[] globalStrides, uint[] boxDim, uint[] elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
        {
            _tensormap = new CUtensorMap();
            res = DriverAPINativeMethods.TensorCoreManagment.cuTensorMapEncodeTiled(ref _tensormap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTensorMapEncodeTiled", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Create a tensor map descriptor object representing im2col memory region<para/>
        /// Creates a descriptor for Tensor Memory Access (TMA) object specified
        /// by the parameters describing a im2col memory layout and returns it in \p tensorMap.<para/>
        /// Tensor map objects are only supported on devices of compute capability 9.0 or higher.
        /// Additionally, a tensor map object is an opaque value, and, as such, should only be
        /// accessed through CUDA API calls.
        /// </summary>
        /// <param name="tensorDataType">Tensor data type</param>
        /// <param name="tensorRank">Dimensionality of tensor, needs to be at least of dimension 3</param>
        /// <param name="globalAddress">Starting address of memory region described by tensor</param>
        /// <param name="globalDim">Array containing tensor size (number of elements) along each of the \p tensorRank dimensions</param>
        /// <param name="globalStrides">Array containing stride size (in bytes) along each of the \p tensorRank - 1 dimensions</param>
        /// <param name="pixelBoxLowerCorner">Array containing DHW dimentions of lower box corner</param>
        /// <param name="pixelBoxUpperCorner">Array containing DHW dimentions of upper box corner</param>
        /// <param name="channelsPerPixel">Number of channels per pixel</param>
        /// <param name="pixelsPerColumn">Number of pixels per column</param>
        /// <param name="elementStrides">Array containing traversal stride in each of the \p tensorRank dimensions</param>
        /// <param name="interleave">Type of interleaved layout the tensor addresses</param>
        /// <param name="swizzle">Bank swizzling pattern inside shared memory</param>
        /// <param name="l2Promotion">L2 promotion size</param>
        /// <param name="oobFill">Indicate whether zero or special NaN constant must be used to fill out-of-bound elements</param>
        public CudaTensorMap(CUtensorMapDataType tensorDataType, uint tensorRank, CUdeviceptr globalAddress, ulong[] globalDim, ulong[] globalStrides, int[] pixelBoxLowerCorner, int[] pixelBoxUpperCorner, uint channelsPerPixel, uint pixelsPerColumn, uint[] elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
        {
            _tensormap = new CUtensorMap();
            res = DriverAPINativeMethods.TensorCoreManagment.cuTensorMapEncodeIm2col(ref _tensormap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTensorMapEncodeIm2col", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Modify an existing tensor map descriptor with an updated global address<para/>
        /// Modifies the descriptor for Tensor Memory Access (TMA) object passed in \p tensorMap with an updated \p globalAddress.<para/>
        /// Tensor map objects are only supported on devices of compute capability 9.0 or higher.
        /// Additionally, a tensor map object is an opaque value, and, as such, should only be
        /// accessed through CUDA API calls.
        /// </summary>
        /// <param name="globalAddress">Starting address of memory region described by tensor, must follow previous alignment requirements</param>
        public CudaTensorMap(CUdeviceptr globalAddress)
        {
            _tensormap = new CUtensorMap();
            res = DriverAPINativeMethods.TensorCoreManagment.cuTensorMapReplaceAddress(ref _tensormap, globalAddress);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTensorMapReplaceAddress", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion

        #region Properties
        /// <summary>
        /// Return inner handle
        /// </summary>
        public CUtensorMap CUTensorMap
        {
            get { return _tensormap; }
        }
        #endregion
    }
}
