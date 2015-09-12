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
    public enum CudaArray1DNumChannels 
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
    /// An one dimensional CUDA array
    /// </summary>
    public class CudaArray1D : IDisposable
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
        /// <param name="size"></param>
        /// <param name="numChannels"></param>
        public CudaArray1D(CUArrayFormat format, SizeT size, CudaArray1DNumChannels numChannels)
        {
            _arrayDescriptor = new CUDAArrayDescriptor();
            _arrayDescriptor.Format = format;
            _arrayDescriptor.Height = 1;
            _arrayDescriptor.Width = size;
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
        public CudaArray1D(CUarray cuArray)
            : this(cuArray, false)
        {

        }

        /// <summary>
        /// Creates a new CUDA array from an existing CUarray. 
        /// Array properties are obtained by cuArrayGetDescriptor
        /// </summary>
        /// <param name="cuArray"></param>
        /// <param name="isOwner">The cuArray will be destroyed while disposing, if the CudaArray is the owner</param>
        public CudaArray1D(CUarray cuArray, bool isOwner)
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
        ~CudaArray1D()
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
        #region CopyFromHostToArray
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="source">source pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromHostToArray1D<T>(T[] source, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = (source.LongLength * Marshal.SizeOf(typeof(T)));
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            CUResult res;
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, ptr, sizeInBytes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            }
            finally
            {
                handle.Free();
            }
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="source">source pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromHostToArray1D<T>(T source, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = Marshal.SizeOf(typeof(T));
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            CUResult res;
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, ptr, sizeInBytes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            }
            finally
            {
                handle.Free();
            }
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="source">Pointer to source data</param>
        /// <param name="sizeInBytes">Number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromHostToArray1D(IntPtr source, SizeT sizeInBytes, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(byte[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, source.LongLength);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(double[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(double)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(float[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(float)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(int[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(int)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(long[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(long)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(sbyte[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(sbyte)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(short[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(short)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(uint[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(uint)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(ulong[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(ulong)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(ushort[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(ushort)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        #region VectorTypesArray
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.dim3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.double1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.double2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuDoubleComplex[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuDoubleReal[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuFloatComplex[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuFloatReal[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion
        #endregion

        #region CopyFromArrayToHost
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromArray1DToHost<T>(T[] dest, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = (dest.Length * Marshal.SizeOf(typeof(T)));
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            CUResult res;
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(ptr, _cuArray, offsetInBytes, sizeInBytes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            }
            finally
            {
                handle.Free();
            }
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromArray1DToHost<T>(ref T dest, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = Marshal.SizeOf(typeof(T));
            // T is a struct and therefor a value type. GCHandle will pin a copy of dest, not dest itself
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            CUResult res;
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(ptr, _cuArray, offsetInBytes, sizeInBytes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
                //Copy Data from pinned copy to original dest
                dest = (T)Marshal.PtrToStructure(ptr, typeof(T)); 
            }
            finally
            {
                handle.Free();
            }
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="dest">Pointer to Destination data</param>
        /// <param name="sizeInBytes">Number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromArray1DToHost(IntPtr dest, uint sizeInBytes, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(byte[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, dest.Length);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(double[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(double)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(float[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(float)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(int[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(int)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(long[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(long)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(sbyte[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(sbyte)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(short[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(short)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(uint[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(uint)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(ulong[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(ulong)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(ushort[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * sizeof(ushort)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        #region VectorTypesArray
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.dim3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.char1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.char2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.char3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.char4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.short1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.short2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.short3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.short4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.int1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.int2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.int3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.int4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.long1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.long2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.long3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.long4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.float1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.float2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.float3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.float4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.double1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.double1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.double2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.double2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuDoubleComplex[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuDoubleReal[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuFloatComplex[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuFloatReal[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.Length * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion
        #endregion
        
        #region InterDeviceCopy
        /// <summary>
        /// Copy data from array to array
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="source">source array</param>
        /// <param name="aBytesToCopy">Size of memory copy in bytes</param>
        /// <param name="destOffset">Offset in bytes of destination array</param>
        /// <param name="sourceOffset">Offset in bytes of source array</param>
        public static void CopyFromArray1DToArray1D(CudaArray1D dest, CudaArray1D source, SizeT aBytesToCopy, SizeT destOffset, SizeT sourceOffset)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoA_v2(dest.CUArray, destOffset, source.CUArray, sourceOffset, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to array
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="aBytesToCopy">Size of memory copy in bytes</param>
        /// <param name="destOffset">Offset in bytes of destination array</param>
        /// <param name="sourceOffset">Offset in bytes of source array</param>
        public void CopyFromThisToArray1D(CudaArray1D dest, SizeT aBytesToCopy, SizeT destOffset, SizeT sourceOffset)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoA_v2(dest.CUArray, destOffset, this.CUArray, sourceOffset, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to array
        /// </summary>
        /// <param name="source">Destination array</param>
        /// <param name="aBytesToCopy">Size of memory copy in bytes</param>
        /// <param name="destOffset">Offset in bytes of destination array</param>
        /// <param name="sourceOffset">Offset in bytes of source array</param>
        public void CopyFromArray1DToThis(CudaArray1D source, SizeT aBytesToCopy, SizeT destOffset, SizeT sourceOffset)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoA_v2(this.CUArray, destOffset, source.CUArray, sourceOffset, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to device
        /// </summary>
        /// <param name="dest">DevicePointer to copy data to</param>
        /// <param name="aBytesToCopy">number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of source array</param>
        public void CopyFromArray1DToDevice(CUdeviceptr dest, SizeT aBytesToCopy, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoD_v2(dest, _cuArray, offsetInBytes, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoD", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to array
        /// </summary>
        /// <param name="source">DevicePointer to copy data from</param>
        /// <param name="aBytesToCopy">number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of source array</param>
        public void CopyFromDeviceToArray1D(CUdeviceptr source, SizeT aBytesToCopy, SizeT offsetInBytes)
        {
            CUResult res;
            res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoA_v2(_cuArray, offsetInBytes, source, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoA", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion
        #endregion

        #region Properties
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
        /// If the wrapper class instance is the owner of a CUDA handle, it will be destroyed while disposing.
        /// </summary>
        public bool IsOwner
        {
            get { return _isOwner; }
        }
        #endregion
    }
}
