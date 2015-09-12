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
	/// A variable located in CUDA device memory
	/// </summary>
	/// <typeparam name="T">variable base type</typeparam>
	public class CudaDeviceVariable<T> : IDisposable where T : struct
	{
		CUdeviceptr _devPtr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructors
		/// <summary>
		/// Creates a new CudaDeviceVariable and allocates the memory on the device
		/// </summary>
		/// <param name="size">In elements</param>
		public CudaDeviceVariable(SizeT size)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (uint)Marshal.SizeOf(typeof(T));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAlloc_v2(ref _devPtr, _typeSize * size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAlloc", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaDeviceVariable from an existing CUdeviceptr. The allocated size is gethered via the CUDA API.
		/// devPtr won't be freed while disposing.
		/// </summary>
		/// <param name="devPtr"></param>
		public CudaDeviceVariable(CUdeviceptr devPtr)
			: this (devPtr, false)
		{

		}

		/// <summary>
		/// Creates a new CudaDeviceVariable from an existing CUdeviceptr. The allocated size is gethered via the CUDA API.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="isOwner">The CUdeviceptr will be freed while disposing, if the CudaDeviceVariable is the owner</param>
		public CudaDeviceVariable(CUdeviceptr devPtr, bool isOwner)
		{
			_devPtr = devPtr;
			CUdeviceptr NULL = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemGetAddressRange_v2(ref NULL, ref _size, devPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemGetAddressRange", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_typeSize = (uint)Marshal.SizeOf(typeof(T));
			SizeT sizeInBytes = _size;
			_size = sizeInBytes / _typeSize;
			if (sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");
			_isOwner = isOwner;
		}

		/// <summary>
		/// Creates a new CudaDeviceVariable from an existing CUdeviceptr.
		/// devPtr won't be freed while disposing.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="size">Size in Bytes</param>
		public CudaDeviceVariable(CUdeviceptr devPtr, SizeT size)
			: this (devPtr, false, size)
		{

		}

		/// <summary>
		/// Creates a new CudaDeviceVariable from an existing CUdeviceptr.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="isOwner">The CUdeviceptr will be freed while disposing, if the CudaDeviceVariable is the owner</param>
		/// <param name="size">Size in Bytes</param>
		public CudaDeviceVariable(CUdeviceptr devPtr, bool isOwner, SizeT size)
		{
			_devPtr = devPtr;
			_typeSize = (uint)Marshal.SizeOf(typeof(T));
			_size = size / _typeSize;
			if (size != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");
			_isOwner = isOwner;
		}

		/// <summary>
		/// Creates a new CudaDeviceVariable from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaDeviceVariable(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(T));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaDeviceVariable from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaDeviceVariable(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaDeviceVariable()
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
					//Ignore if failing
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Copy sync
		private SizeT MinSize(SizeT value)
		{
			return (ulong)value < (ulong)_size ? value : _size;
		}
		private SizeT MinSizeWithType(SizeT value)
		{
			return (ulong)value < (ulong)_size * _typeSize? value : _size * _typeSize;
		}

		/// <summary>
		/// Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		public void CopyToDevice(CUdeviceptr source)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2(_devPtr, source, aSizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		public void CopyToDevice(CUdeviceptr source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2(_devPtr + offsetDest, source + offsetSrc, sizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source</param>
		public void CopyToDevice(CudaDeviceVariable<T> source)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = MinSize(source.Size) * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2(_devPtr, source.DevicePointer, aSizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		public void CopyToDevice(CudaDeviceVariable<T> source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2(_devPtr + offsetDest, source.DevicePointer + offsetSrc, sizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDevice(CudaPitchedDeviceVariable<T> deviceSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = deviceSrc.Height;
			copyParams.WidthInBytes = deviceSrc.WidthInBytes;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="widthInBytes">Width of 2D memory to copy in bytes</param>
		/// <param name="height">Height in elements</param>
		public void CopyToDevice(CudaPitchedDeviceVariable<T> deviceSrc, SizeT offsetSrc, SizeT offsetDest, SizeT widthInBytes, SizeT height)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer + offsetSrc;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr + offsetDest;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = height;
			copyParams.WidthInBytes = widthInBytes;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		public void CopyToDevice(T[] source)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = MinSize(source.LongLength) * _typeSize;
			GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr, ptr, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		public void CopyToDevice(T[] source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = new IntPtr(handle.AddrOfPinnedObject().ToInt64() + (long)offsetSrc);
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr + offsetDest, ptr, sizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		public void CopyToDevice(T source)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _typeSize;
			GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr, ptr, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		public void CopyToDevice(T source, SizeT offsetDest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _typeSize;
			GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr + offsetDest, ptr, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		public void CopyToDevice(IntPtr source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());

			CUResult res;
			IntPtr ptr = new IntPtr(source.ToInt64() + (long)offsetSrc);
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr + offsetDest, ptr, sizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		public void CopyToDevice(IntPtr source, SizeT offsetDest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());

			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr + offsetDest, source, _size * _typeSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		public void CopyToDevice(IntPtr source)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr, source, _size * _typeSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from Host to device memory. Array elements can be of any (value)type, but total size in bytes must match to allocated device memory.
		/// </summary>
		/// <param name="hostSrc">Source</param>
		public void CopyToDevice(Array hostSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_devPtr, ptr, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="dest">Destination pointer to host memory</param>
		public void CopyToHost(T[] dest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = MinSize(dest.LongLength) * _typeSize;
			GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, _devPtr, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			}
			finally
			{
				handle.Free();
			}
			
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="dest">Destination pointer to host memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		public void CopyToHost(T[] dest, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = new IntPtr(handle.AddrOfPinnedObject().ToInt64() + (long)offsetDest);
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, _devPtr + offsetSrc, sizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			}
			finally
			{
				handle.Free();
			}

			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="dest">Destination data in host memory</param>
		public void CopyToHost(ref T dest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _typeSize;
			// T is a struct and therefor a value type. GCHandle will pin a copy of dest, not dest itself
			GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, _devPtr, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
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
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="dest">Destination data in host memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		public void CopyToHost(ref T dest, SizeT offsetSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _typeSize;
			// T is a struct and therefor a value type. GCHandle will pin a copy of dest, not dest itself
			GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, _devPtr + offsetSrc, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
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
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="dest">Destination pointer to host memory</param>
		public void CopyToHost(IntPtr dest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());

			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, _devPtr, _size * _typeSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="dest">Destination data in host memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		public void CopyToHost(IntPtr dest, SizeT offsetSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dest, _devPtr + offsetSrc, _size * _typeSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="dest">Destination pointer to host memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		public void CopyToHost(IntPtr dest, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());

			CUResult res;
			IntPtr ptr = new IntPtr(dest.ToInt64() + (long)offsetDest);
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, _devPtr + offsetSrc, sizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from this device to host memory. Array elements can be of any (value)type, but total size in bytes must match to allocated device memory.
		/// </summary>
		/// <param name="hostDest">Destination</param>
		public void CopyToHost(Array hostDest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, _devPtr, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region AsyncDeviceToDevice
		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr source, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr, source, aSizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<T> source, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = MinSize(source.Size) * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr, source.DevicePointer, aSizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaPitchedDeviceVariable<T> deviceSrc, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = deviceSrc.Height;
			copyParams.WidthInBytes = deviceSrc.WidthInBytes;
			
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr source, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr, source, aSizeInBytes, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<T> source, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = MinSize(source.Size) * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr, source.DevicePointer, aSizeInBytes, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaPitchedDeviceVariable<T> deviceSrc, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = deviceSrc.Height;
			copyParams.WidthInBytes = deviceSrc.WidthInBytes;

			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}


		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr + offsetDest, source + offsetSrc, sizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<T> source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr + offsetDest, source.DevicePointer + offsetSrc, sizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr + offsetDest, source + offsetSrc, sizeInBytes, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="sizeInBytes">Size to copy in bytes</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<T> source, SizeT offsetSrc, SizeT offsetDest, SizeT sizeInBytes, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr + offsetDest, source.DevicePointer + offsetSrc, sizeInBytes, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="width">Width of 2D memory to copy in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaPitchedDeviceVariable<T> deviceSrc, SizeT offsetSrc, SizeT offsetDest, SizeT width, SizeT height, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer + offsetSrc;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr + offsetDest;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = height;
			copyParams.WidthInBytes = width;

			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="width">Width of 2D memory to copy in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaPitchedDeviceVariable<T> deviceSrc, SizeT offsetSrc, SizeT offsetDest, SizeT width, SizeT height, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer + offsetSrc;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr + offsetDest;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = height;
			copyParams.WidthInBytes = width;

			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region Memset
		/// <summary>
		/// Memset
		/// </summary>
		/// <param name="aValue"></param>
		public void Memset(byte aValue)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.Memset.cuMemsetD8_v2(_devPtr, aValue, _size * _typeSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD8", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Memset
		/// </summary>
		/// <param name="aValue"></param>
		public void Memset(ushort aValue)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.Memset.cuMemsetD16_v2(_devPtr, aValue, _size * _typeSize / sizeof(ushort));
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD16", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Memset
		/// </summary>
		/// <param name="aValue"></param>
		public void Memset(uint aValue)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.Memset.cuMemsetD32_v2(_devPtr, aValue, _size * _typeSize / sizeof(uint));
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD32", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region Memset async
		/// <summary>
		/// Memset
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="stream"></param>
		public void MemsetAsync(byte aValue, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.MemsetAsync.cuMemsetD8Async(_devPtr, aValue, _size * _typeSize, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD8Async", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Memset
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="stream"></param>
		public void MemsetAsync(ushort aValue, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.MemsetAsync.cuMemsetD16Async(_devPtr, aValue, _size * _typeSize / sizeof(ushort), stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD16Async", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Memset
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="stream"></param>
		public void MemsetAsync(uint aValue, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(_devPtr, aValue, _size * _typeSize / sizeof(uint), stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD32Async", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region PeerCopy
		/// <summary>
		/// Copies from device memory in one context to device memory in another context
		/// </summary>
		/// <param name="destCtx">Destination context</param>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="sourceCtx">Source context</param>
		public void PeerCopyToDevice(CudaContext destCtx, CUdeviceptr source, CudaContext sourceCtx)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyPeer(_devPtr, destCtx.Context, source, sourceCtx.Context, aSizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyPeer", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copies from device memory in one context to device memory in another context
		/// </summary>
		/// <param name="destCtx">Destination context</param>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="sourceCtx">Source context</param>
		public void PeerCopyToDevice(CudaContext destCtx, CudaDeviceVariable<T> source, CudaContext sourceCtx)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyPeer(_devPtr, destCtx.Context, source.DevicePointer, sourceCtx.Context, aSizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyPeer", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async-Copies from device memory in one context to device memory in another context
		/// </summary>
		/// <param name="destCtx">Destination context</param>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="sourceCtx">Source context</param>
		/// <param name="stream"></param>
		public void PeerCopyToDevice(CudaContext destCtx, CUdeviceptr source, CudaContext sourceCtx, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyPeerAsync(_devPtr, destCtx.Context, source, sourceCtx.Context, aSizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyPeerAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async-Copies from device memory in one context to device memory in another context
		/// </summary>
		/// <param name="destCtx">Destination context</param>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="sourceCtx">Source context</param>
		/// <param name="stream"></param>
		public void PeerCopyToDevice(CudaContext destCtx, CudaDeviceVariable<T> source, CudaContext sourceCtx, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _size * _typeSize;
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyPeerAsync(_devPtr, destCtx.Context, source.DevicePointer, sourceCtx.Context, aSizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyPeerAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region Properties
		/// <summary>
		/// Access array elements directly from host.<para/>
		/// Each single access invokes a device to host or host to device copy. Access is therefor rather slow.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public T this[SizeT index]
		{
			get
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());

				CUdeviceptr position = _devPtr + index * _typeSize;
				T dest = new T();

				SizeT aSizeInBytes = _typeSize;
				// T is a struct and therefor a value type. GCHandle will pin a copy of dest, not dest itself
				GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
				CUResult res;
				try
				{
					IntPtr ptr = handle.AddrOfPinnedObject();
					res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(ptr, position, aSizeInBytes);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
					//Copy Data from pinned copy to original dest
					dest = (T)Marshal.PtrToStructure(ptr, typeof(T));
				}
				finally
				{
					handle.Free();
				}
				if (res != CUResult.Success)
					throw new CudaException(res);
				return dest;
			}

			set
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				CUdeviceptr position = _devPtr + index * _typeSize;

				SizeT aSizeInBytes = _typeSize;
				GCHandle handle = GCHandle.Alloc(value, GCHandleType.Pinned);
				CUResult res;
				try
				{
					IntPtr ptr = handle.AddrOfPinnedObject();
					res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(position, ptr, aSizeInBytes);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
				}
				finally
				{
					handle.Free();
				}
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
		}

		/// <summary>
		/// Device pointer
		/// </summary>
		public CUdeviceptr DevicePointer
		{
			get { return _devPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Type size in bytes
		/// </summary>
		public SizeT TypeSize
		{
			get { return _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// If the wrapper class instance is the owner of a CUDA handle, it will be destroyed while disposing.
		/// </summary>
		public bool IsOwner
		{
			get { return _isOwner; }
		}
		#endregion

		#region Converter operators
		/// <summary>
		/// Converts a device variable to a host array
		/// </summary>
		/// <param name="d">device variable</param>
		/// <returns>newly allocated host array with values from device memory</returns>
		public static implicit operator T[](CudaDeviceVariable<T> d)
		{
			T[] ret = new T[(long)d.Size];
			d.CopyToHost(ret);
			return ret;
		}

		/// <summary>
		/// Converts a device variable to a host value. In case of multiple device values, only the first value is copied.
		/// </summary>
		/// <param name="d">device variable</param>
		/// <returns>newly allocated host variable with value from device memory</returns>
		public static implicit operator T(CudaDeviceVariable<T> d)
		{
			T ret = new T();
			d.CopyToHost(ref ret);
			return ret;
		}

		/// <summary>
		/// Converts a host array to a newly allocated device variable.
		/// </summary>
		/// <param name="d">host array</param>
		/// <returns>newly allocated device variable with values from host memory</returns>
		public static implicit operator CudaDeviceVariable<T>(T[] d)
		{
			CudaDeviceVariable<T> ret = new CudaDeviceVariable<T>(d.LongLength);
			ret.CopyToDevice(d);
			return ret;
		}

		/// <summary>
		/// Converts a host array to a newly allocated device variable.
		/// </summary>
		/// <param name="d">host array</param>
		/// <returns>newly allocated device variable with values from host memory</returns>
		public static implicit operator CudaDeviceVariable<T>(T d)
		{
			CudaDeviceVariable<T> ret = new CudaDeviceVariable<T>(1);
			ret.CopyToDevice(d);
			return ret;
		}
		#endregion

		#region NULL
		/// <summary>
		/// Gets a null-pointer equivalent
		/// </summary>
		public static CudaDeviceVariable<T> Null
		{
			get 
			{ 
				return new CudaDeviceVariable<T>(new CUdeviceptr(), false, 0); 
			}
		}
		#endregion
	}
}
