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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Number of channels in array
    /// </summary>
    public enum CudaArray2DNumChannels 
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
    /// A two dimansional CUDA array
    /// </summary>
    public class CudaArray2D : IDisposable
    {
        private CUDAArrayDescriptor _arrayDescriptor;
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
        /// <param name="numChannels"></param>
        public CudaArray2D(CUArrayFormat format, SizeT width, SizeT height, CudaArray2DNumChannels numChannels)
        {
            _arrayDescriptor = new CUDAArrayDescriptor();
            _arrayDescriptor.Format = format;
            _arrayDescriptor.Height = height;
            _arrayDescriptor.Width = width;
            _arrayDescriptor.NumChannels = (uint)numChannels;

            _cuArray = new CUarray();

            res = DriverAPINativeMethods.ArrayManagement.cuArrayCreate_v2(ref _cuArray, ref _arrayDescriptor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArrayCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;
        }

        /// <summary>
        /// Creates a new CUDA array from an existing CUarray. 
        /// The CUarray won't be destroyed when disposing.
        /// Array properties are obtained by cuArrayGetDescriptor
        /// </summary>
        /// <param name="cuArray"></param>
        public CudaArray2D(CUarray cuArray)
            : this(cuArray, false)
        {
            
        }

        /// <summary>
        /// Creates a new CUDA array from an existing CUarray. 
        /// Array properties are obtained by cuArrayGetDescriptor
        /// </summary>
        /// <param name="cuArray"></param>
        /// <param name="isOwner">The cuArray will be destroyed while disposing if the CudaArray is the owner</param>
        public CudaArray2D(CUarray cuArray, bool isOwner)
        {
            _cuArray = cuArray;
            _arrayDescriptor = new CUDAArrayDescriptor();

            res = DriverAPINativeMethods.ArrayManagement.cuArrayGetDescriptor_v2(ref _arrayDescriptor, _cuArray);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArrayGetDescriptor", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = isOwner;
        }
        
        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaArray2D()
        {
            Dispose (false);            
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
        protected virtual void Dispose (bool fDisposing)
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
        /// <param name="aCopyParameters">2D copy paramters</param>
        public void CopyData(CUDAMemCpy2D aCopyParameters)
        {
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref aCopyParameters);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// A raw unaligned copy method
        /// </summary>
        /// <param name="aCopyParameters"></param>
        public void CopyDataUnaligned(CUDAMemCpy2D aCopyParameters)
        {
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2DUnaligned_v2(ref aCopyParameters);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DUnaligned", res));
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
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcHost = aHostSrc;
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = _arrayDescriptor.Height;
            copyParams.WidthInBytes = _arrayDescriptor.Width * aElementSizeInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
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
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstHost = aHostDest;
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Height = _arrayDescriptor.Height;
            copyParams.WidthInBytes = _arrayDescriptor.Width * aElementSizeInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to this array
        /// </summary>
        /// <typeparam name="T">Host array base type</typeparam>
        /// <param name="aHostSrc">Source</param>
        public void CopyFromHostToThis<T>(T[] aHostSrc) where T : struct
        {
            GCHandle handle = GCHandle.Alloc(aHostSrc, GCHandleType.Pinned);
            try
            {
                CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
                copyParams.srcHost = handle.AddrOfPinnedObject();
                copyParams.srcMemoryType = CUMemoryType.Host;
                copyParams.dstArray = _cuArray;
                copyParams.dstMemoryType = CUMemoryType.Array;
                copyParams.Height = _arrayDescriptor.Height;
                copyParams.WidthInBytes = _arrayDescriptor.Width * Marshal.SizeOf(typeof(T));

                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
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
        public void CopyFromThisToHost<T>(T[] aHostDest) where T : struct
        {
            GCHandle handle = GCHandle.Alloc(aHostDest, GCHandleType.Pinned);
            try
            {
                CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
                copyParams.dstHost = handle.AddrOfPinnedObject();
                copyParams.dstMemoryType = CUMemoryType.Host;
                copyParams.srcArray = _cuArray;
                copyParams.srcMemoryType = CUMemoryType.Array;
                copyParams.Height = _arrayDescriptor.Height;
                copyParams.WidthInBytes = _arrayDescriptor.Width * Marshal.SizeOf(typeof(T));

                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
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
        /// <typeparam name="T">device variable base type</typeparam>
        /// <param name="aDeviceVariable">Source</param>
        public void CopyFromDeviceToThis<T>(CudaPitchedDeviceVariable<T> aDeviceVariable) where T:struct
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcDevice = aDeviceVariable.DevicePointer;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = aDeviceVariable.Pitch;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = aDeviceVariable.Height;
            copyParams.WidthInBytes = aDeviceVariable.WidthInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from this array to a pitched device variable
        /// </summary>
        /// <typeparam name="T">device variable base type</typeparam>
        /// <param name="aDeviceVariable">Destination</param>
        public void CopyFromThisToDevice<T>(CudaPitchedDeviceVariable<T> aDeviceVariable) where T : struct
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstDevice = aDeviceVariable.DevicePointer;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = aDeviceVariable.Pitch;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Height = aDeviceVariable.Height;
            copyParams.WidthInBytes = aDeviceVariable.WidthInBytes;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="aSourceArray"></param>
        public void CopyFromArrayToThis(CudaArray2D aSourceArray)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcArray = aSourceArray.CUArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = aSourceArray.ArrayDescriptor.Height;
            copyParams.WidthInBytes = aSourceArray.ArrayDescriptor.Width * CudaHelperMethods.GetChannelSize(aSourceArray.ArrayDescriptor.Format) * aSourceArray.ArrayDescriptor.NumChannels;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="aDestArray"></param>
        public void CopyFromThisToArray(CudaArray2D aDestArray)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = aDestArray.CUArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = aDestArray.ArrayDescriptor.Height;
            copyParams.WidthInBytes = aDestArray.ArrayDescriptor.Width * CudaHelperMethods.GetChannelSize(aDestArray.ArrayDescriptor.Format) * aDestArray.ArrayDescriptor.NumChannels;

            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
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
        /// Returns the wrapped CUDAArrayDescriptor
        /// </summary>
        public CUDAArrayDescriptor ArrayDescriptor
        {
            get { return _arrayDescriptor; }
        }

        /// <summary>
        /// Returns the Height of the array
        /// </summary>
        public SizeT Height
        {
            get { return _arrayDescriptor.Height; }
        }

        /// <summary>
        /// Returns the array width in elements
        /// </summary>
        public SizeT Width
        {
            get { return _arrayDescriptor.Width; }
        }

        /// <summary>
        /// Returns the array width in bytes
        /// </summary>
        public SizeT WidthInBytes
        {
            get { return _arrayDescriptor.Width * _arrayDescriptor.NumChannels * CudaHelperMethods.GetChannelSize(_arrayDescriptor.Format); }
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
