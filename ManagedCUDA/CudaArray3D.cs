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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Number of channels in array
    /// </summary>
    public enum CudaArray3DNumChannels
    {
        /// <summary>
        /// One channel, e.g. float1, int1, float, int
        /// </summary>
        One = 1,
        /// <summary>
        /// Two channels, e.g. float2, int2
        /// </summary>
        Two = 2,
        /// <summary>
        /// Four channels, e.g. float4, int4
        /// </summary>
        Four = 4
    }

    /// <summary>
    /// A three dimensional CUDA array
    /// </summary>
    public class CudaArray3D : IDisposable
    {
        private CUDAArray3DDescriptor _array3DDescriptor;
        private CUarray _cuArray;
        private CUResult res;
        private bool disposed;
        private bool _isOwner;

        #region Constructors
        /// <summary>
        /// Creates a new CUDA array. 
        /// </summary>
        /// <param name="format"></param>
        /// <param name="width">In elements</param>
        /// <param name="height">In elements</param>
        /// <param name="depth">In elements</param>
        /// <param name="numChannels"></param>
        /// <param name="flags"></param>
        public CudaArray3D(CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaArray3DNumChannels numChannels, CUDAArray3DFlags flags)
        {
            _array3DDescriptor = new CUDAArray3DDescriptor();
            _array3DDescriptor.Format = format;
            _array3DDescriptor.Height = height;
            _array3DDescriptor.Width = width;
            _array3DDescriptor.Depth = depth;
            _array3DDescriptor.Flags = flags;
            _array3DDescriptor.NumChannels = (uint)numChannels;

            _cuArray = new CUarray();

            res = DriverAPINativeMethods.ArrayManagement.cuArray3DCreate_v2(ref _cuArray, ref _array3DDescriptor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArray3DCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;
        }

        /// <summary>
        /// Creates a new CUDA array from an existing CUarray. 
        /// The CUarray won't be destroyed when disposing.
        /// Array properties are obtained by cuArrayGetDescriptor
        /// </summary>
        /// <param name="cuArray"></param>
        public CudaArray3D(CUarray cuArray)
            : this(cuArray, false)
        {

        }

        /// <summary>
        /// Creates a new CUDA array from an existing CUarray. 
        /// Array properties are obtained by cuArrayGetDescriptor
        /// </summary>
        /// <param name="cuArray"></param>
        /// <param name="isOwner">The cuArray will be destroyed while disposing, if the CudaArray is the owner</param>
        public CudaArray3D(CUarray cuArray, bool isOwner)
        {
            _cuArray = cuArray;
            _array3DDescriptor = new CUDAArray3DDescriptor();

            res = DriverAPINativeMethods.ArrayManagement.cuArray3DGetDescriptor_v2(ref _array3DDescriptor, _cuArray);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArray3DGetDescriptor", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = isOwner;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaArray3D()
        {
            Dispose(false);
        }
        #endregion

        #region Dispose
        /// <summary>
        /// Dispose
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// For IDisposable
        /// </summary>
        /// <param name="fDisposing"></param>
        protected virtual void Dispose(bool fDisposing)
        {
            if (fDisposing && !disposed)
            {
                if (_isOwner)
                {
                    res = DriverAPINativeMethods.ArrayManagement.cuArrayDestroy(_cuArray);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArrayDestroy", res));
                }
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods
        /// <summary>
        /// A raw data copy method
        /// </summary>
        /// <param name="aCopyParameters">3D copy paramters</param>
        public void CopyData(CUDAMemCpy3D aCopyParameters)
        {
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref aCopyParameters);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to this array
        /// </summary>
        /// <param name="aHostSrc">Source</param>
        /// <param name="aElementSizeInBytes"></param>
        public void CopyFromHostToThis(IntPtr aHostSrc, SizeT aElementSizeInBytes)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();

            copyParams.srcHost = aHostSrc;
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * aElementSizeInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from this array to host
        /// </summary>
        /// <param name="aHostDest">IntPtr to destination in host memory</param>
        /// <param name="aElementSizeInBytes"></param>
        public void CopyFromThisToHost(IntPtr aHostDest, SizeT aElementSizeInBytes)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.dstHost = aHostDest;
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * aElementSizeInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to this array
        /// </summary>
        /// <typeparam name="T">Host array base type</typeparam>
        /// <param name="aHostSrc">Source</param>
        public void CopyFromHostToThis<T>(T[] aHostSrc)
        {
            GCHandle handle = GCHandle.Alloc(aHostSrc, GCHandleType.Pinned);
            try
            {
                CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
                copyParams.srcHost = handle.AddrOfPinnedObject();
                copyParams.srcMemoryType = CUMemoryType.Host;
                copyParams.dstArray = _cuArray;
                copyParams.dstMemoryType = CUMemoryType.Array;
                copyParams.Depth = _array3DDescriptor.Depth;
                copyParams.Height = _array3DDescriptor.Height;
                copyParams.WidthInBytes = _array3DDescriptor.Width * Marshal.SizeOf(typeof(T));

                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            }
            finally
            {
                handle.Free();
            }
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from this array to host
        /// </summary>
        /// <typeparam name="T">Host array base type</typeparam>
        /// <param name="aHostDest">Destination</param>
        public void CopyFromThisToHost<T>(T[] aHostDest)
        {
            GCHandle handle = GCHandle.Alloc(aHostDest, GCHandleType.Pinned);
            try
            {
                CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
                copyParams.dstHost = handle.AddrOfPinnedObject();
                copyParams.dstMemoryType = CUMemoryType.Host;
                copyParams.srcArray = _cuArray;
                copyParams.srcMemoryType = CUMemoryType.Array;
                copyParams.Depth = _array3DDescriptor.Depth;
                copyParams.Height = _array3DDescriptor.Height;
                copyParams.WidthInBytes = _array3DDescriptor.Width * Marshal.SizeOf(typeof(T));

                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            }
            finally
            {
                handle.Free();
            }
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from a pitched device variable to this array
        /// </summary>
        /// <param name="aDeviceVariable">Source</param>
        /// <param name="aElementSizeInBytes"></param>
        public void CopyFromDeviceToThis(CUdeviceptr aDeviceVariable, SizeT aElementSizeInBytes)
        {
            CopyFromDeviceToThis(aDeviceVariable, aElementSizeInBytes, 0);
        }

        /// <summary>
        /// Copy from a pitched device variable to this array
        /// </summary>
        /// <param name="aDeviceVariable">Source</param>
        /// <param name="aElementSizeInBytes"></param>
        /// <param name="pitch">Pitch in bytes</param>
        public void CopyFromDeviceToThis(CUdeviceptr aDeviceVariable, SizeT aElementSizeInBytes, SizeT pitch)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.srcDevice = aDeviceVariable;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = pitch;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * aElementSizeInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from this array to a pitched device variable
        /// </summary>
        /// <param name="aDeviceVariable">Destination</param>
        /// <param name="aElementSizeInBytes"></param>
        public void CopyFromThisToDevice(CUdeviceptr aDeviceVariable, SizeT aElementSizeInBytes)
        {
            CopyFromThisToDevice(aDeviceVariable, aElementSizeInBytes, 0);
        }

        /// <summary>
        /// Copy from this array to a pitched device variable
        /// </summary>
        /// <param name="aDeviceVariable">Destination</param>
        /// <param name="aElementSizeInBytes"></param>
        /// <param name="pitch">Pitch in bytes</param>
        public void CopyFromThisToDevice(CUdeviceptr aDeviceVariable, SizeT aElementSizeInBytes, SizeT pitch)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.dstDevice = aDeviceVariable;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = pitch;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * aElementSizeInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="aSourceArray"></param>
        public void CopyFromArrayToThis(CudaArray3D aSourceArray)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.srcArray = aSourceArray.CUArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = aSourceArray._array3DDescriptor.Depth;
            copyParams.Height = aSourceArray.Array3DDescriptor.Height;
            copyParams.WidthInBytes = aSourceArray.Array3DDescriptor.Width * CudaHelperMethods.GetChannelSize(aSourceArray.Array3DDescriptor.Format) * aSourceArray.Array3DDescriptor.NumChannels;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="aDestArray"></param>
        public void CopyFromThisToArray(CudaArray3D aDestArray)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = aDestArray.CUArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = aDestArray._array3DDescriptor.Depth;
            copyParams.Height = aDestArray.Array3DDescriptor.Height;
            copyParams.WidthInBytes = aDestArray.Array3DDescriptor.Width * CudaHelperMethods.GetChannelSize(aDestArray.Array3DDescriptor.Format) * aDestArray.Array3DDescriptor.NumChannels;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>        /// 
        /// Returns the layout properties of a sparse CUDA array
        /// Returns the layout properties of a sparse CUDA array in \p sparseProperties
        /// If the CUDA array is not allocated with flag ::CUDA_ARRAY3D_SPARSE ::CUDA_ERROR_INVALID_VALUE will be returned.
        /// If the returned value in ::CUDA_ARRAY_SPARSE_PROPERTIES::flags contains ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL,
        /// then::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize represents the total size of the array.Otherwise, it will be zero.
        /// Also, the returned value in ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel is always zero.
        /// Note that the \p array must have been allocated using ::cuArrayCreate or::cuArray3DCreate.For CUDA arrays obtained
        /// using ::cuMipmappedArrayGetLevel, ::CUDA_ERROR_INVALID_VALUE will be returned.Instead, ::cuMipmappedArrayGetSparseProperties
        /// must be used to obtain the sparse properties of the entire CUDA mipmapped array to which \p array belongs to.
        /// </summary>
        public CudaArraySparseProperties GetSparseProperties()
        {
            CudaArraySparseProperties sparseProperties = new CudaArraySparseProperties();

            res = DriverAPINativeMethods.ArrayManagement.cuArrayGetSparseProperties(ref sparseProperties, _cuArray);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArrayGetSparseProperties", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            return sparseProperties;
        }

        /// <summary>  
        /// Gets a CUDA array plane from a CUDA array<para/>
        /// Returns a CUDA array that represents a single format plane
        /// of the CUDA array \p hArray.<para/>
        /// If planeIdx is greater than the maximum number of planes in this array or if the array does
        /// not have a multi-planar format e.g: ::CU_AD_FORMAT_NV12, then::CUDA_ERROR_INVALID_VALUE is returned.<para/>
        /// Note that if the \p hArray has format ::CU_AD_FORMAT_NV12, then passing in 0 for \p planeIdx returns
        /// a CUDA array of the same size as \p hArray but with one channel and::CU_AD_FORMAT_UNSIGNED_INT8 as its format.
        /// If 1 is passed for \p planeIdx, then the returned CUDA array has half the height and width
        /// of \p hArray with two channels and ::CU_AD_FORMAT_UNSIGNED_INT8 as its format.
        /// </summary>
        public CudaArray3D GetPlane(uint planeIdx)
        {
            CUarray arrayPlane = new CUarray();

            res = DriverAPINativeMethods.ArrayManagement.cuArrayGetPlane(ref arrayPlane, _cuArray, planeIdx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArrayGetPlane", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            return new CudaArray3D(arrayPlane, true);
        }

        /// <summary>
        /// Returns the memory requirements of a CUDA array
        /// </summary>
        public CudaArrayMemoryRequirements GetMemoryRequirements(CUdevice device)
        {
            return _cuArray.GetMemoryRequirements(device);
        }
        #endregion

        #region Properties
        /// <summary>
        /// Returns the wrapped CUarray
        /// </summary>
        public CUarray CUArray
        {
            get { return _cuArray; }
        }

        /// <summary>
        /// Returns the wrapped CUDAArray3DDescriptor
        /// </summary>
        public CUDAArray3DDescriptor Array3DDescriptor
        {
            get { return _array3DDescriptor; }
        }

        /// <summary>
        /// Returns the Depth of the array
        /// </summary>
        public SizeT Depth
        {
            get { return _array3DDescriptor.Depth; }
        }

        /// <summary>
        /// Returns the Height of the array
        /// </summary>
        public SizeT Height
        {
            get { return _array3DDescriptor.Height; }
        }

        /// <summary>
        /// Returns the array width in elements
        /// </summary>
        public SizeT Width
        {
            get { return _array3DDescriptor.Width; }
        }

        /// <summary>
        /// Returns the array width in bytes
        /// </summary>
        public SizeT WidthInBytes
        {
            get { return _array3DDescriptor.Width * _array3DDescriptor.NumChannels * CudaHelperMethods.GetChannelSize(_array3DDescriptor.Format); }
        }

        /// <summary>
        /// If the wrapper class instance is the owner of a CUDA handle, it will be destroyed while disposing.
        /// </summary>
        public bool IsOwner
        {
            get { return _isOwner; }
        }
        #endregion
    }
}
