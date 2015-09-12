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
using System.Collections;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: byte
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_byte : IDisposable
	{
		IntPtr _intPtr;
		byte* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_byte from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_byte(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(byte));
			_ptr = (byte*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_byte()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public byte this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<byte> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<byte> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<byte> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<byte> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uchar1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uchar1 : IDisposable
	{
		IntPtr _intPtr;
		uchar1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uchar1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uchar1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar1));
			_ptr = (uchar1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uchar1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uchar1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uchar1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uchar1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uchar1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uchar1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uchar2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uchar2 : IDisposable
	{
		IntPtr _intPtr;
		uchar2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uchar2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uchar2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar2));
			_ptr = (uchar2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uchar2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uchar2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uchar2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uchar2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uchar2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uchar2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uchar3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uchar3 : IDisposable
	{
		IntPtr _intPtr;
		uchar3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uchar3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uchar3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar3));
			_ptr = (uchar3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uchar3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uchar3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uchar3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uchar3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uchar3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uchar3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uchar4
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uchar4 : IDisposable
	{
		IntPtr _intPtr;
		uchar4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uchar4 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uchar4(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar4));
			_ptr = (uchar4*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uchar4()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uchar4 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uchar4> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uchar4> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uchar4> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uchar4> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: sbyte
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_sbyte : IDisposable
	{
		IntPtr _intPtr;
		sbyte* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_sbyte from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_sbyte(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(sbyte));
			_ptr = (sbyte*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_sbyte()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public sbyte this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<sbyte> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<sbyte> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<sbyte> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<sbyte> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: char1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_char1 : IDisposable
	{
		IntPtr _intPtr;
		char1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_char1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_char1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char1));
			_ptr = (char1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_char1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public char1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<char1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<char1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<char1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<char1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: char2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_char2 : IDisposable
	{
		IntPtr _intPtr;
		char2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_char2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_char2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char2));
			_ptr = (char2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_char2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public char2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<char2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<char2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<char2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<char2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: char3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_char3 : IDisposable
	{
		IntPtr _intPtr;
		char3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_char3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_char3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char3));
			_ptr = (char3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_char3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public char3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<char3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<char3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<char3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<char3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: char4
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_char4 : IDisposable
	{
		IntPtr _intPtr;
		char4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_char4 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_char4(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char4));
			_ptr = (char4*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_char4()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public char4 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<char4> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<char4> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<char4> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<char4> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: short
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_short : IDisposable
	{
		IntPtr _intPtr;
		short* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_short from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_short(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short));
			_ptr = (short*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_short()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public short this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<short> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<short> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<short> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<short> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: short1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_short1 : IDisposable
	{
		IntPtr _intPtr;
		short1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_short1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_short1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short1));
			_ptr = (short1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_short1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public short1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<short1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<short1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<short1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<short1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: short2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_short2 : IDisposable
	{
		IntPtr _intPtr;
		short2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_short2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_short2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short2));
			_ptr = (short2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_short2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public short2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<short2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<short2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<short2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<short2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: short3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_short3 : IDisposable
	{
		IntPtr _intPtr;
		short3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_short3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_short3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short3));
			_ptr = (short3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_short3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public short3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<short3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<short3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<short3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<short3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: short4
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_short4 : IDisposable
	{
		IntPtr _intPtr;
		short4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_short4 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_short4(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short4));
			_ptr = (short4*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_short4()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public short4 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<short4> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<short4> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<short4> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<short4> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ushort
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ushort : IDisposable
	{
		IntPtr _intPtr;
		ushort* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ushort from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ushort(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort));
			_ptr = (ushort*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ushort()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ushort this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ushort> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ushort> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ushort> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ushort> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ushort1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ushort1 : IDisposable
	{
		IntPtr _intPtr;
		ushort1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ushort1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ushort1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort1));
			_ptr = (ushort1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ushort1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ushort1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ushort1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ushort1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ushort1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ushort1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ushort2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ushort2 : IDisposable
	{
		IntPtr _intPtr;
		ushort2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ushort2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ushort2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort2));
			_ptr = (ushort2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ushort2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ushort2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ushort2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ushort2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ushort2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ushort2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ushort3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ushort3 : IDisposable
	{
		IntPtr _intPtr;
		ushort3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ushort3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ushort3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort3));
			_ptr = (ushort3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ushort3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ushort3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ushort3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ushort3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ushort3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ushort3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ushort4
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ushort4 : IDisposable
	{
		IntPtr _intPtr;
		ushort4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ushort4 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ushort4(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort4));
			_ptr = (ushort4*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ushort4()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ushort4 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ushort4> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ushort4> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ushort4> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ushort4> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: int
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_int : IDisposable
	{
		IntPtr _intPtr;
		int* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_int from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_int(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int));
			_ptr = (int*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_int()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public int this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<int> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<int> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<int> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<int> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: int1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_int1 : IDisposable
	{
		IntPtr _intPtr;
		int1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_int1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_int1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int1));
			_ptr = (int1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_int1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public int1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<int1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<int1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<int1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<int1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: int2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_int2 : IDisposable
	{
		IntPtr _intPtr;
		int2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_int2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_int2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int2));
			_ptr = (int2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_int2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public int2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<int2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<int2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<int2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<int2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: int3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_int3 : IDisposable
	{
		IntPtr _intPtr;
		int3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_int3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_int3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int3));
			_ptr = (int3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_int3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public int3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<int3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<int3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<int3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<int3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: int4
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_int4 : IDisposable
	{
		IntPtr _intPtr;
		int4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_int4 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_int4(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int4));
			_ptr = (int4*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_int4()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public int4 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<int4> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<int4> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<int4> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<int4> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uint
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uint : IDisposable
	{
		IntPtr _intPtr;
		uint* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uint from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uint(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint));
			_ptr = (uint*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uint()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uint this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uint> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uint> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uint> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uint> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uint1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uint1 : IDisposable
	{
		IntPtr _intPtr;
		uint1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uint1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uint1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint1));
			_ptr = (uint1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uint1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uint1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uint1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uint1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uint1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uint1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uint2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uint2 : IDisposable
	{
		IntPtr _intPtr;
		uint2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uint2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uint2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint2));
			_ptr = (uint2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uint2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uint2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uint2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uint2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uint2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uint2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uint3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uint3 : IDisposable
	{
		IntPtr _intPtr;
		uint3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uint3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uint3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint3));
			_ptr = (uint3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uint3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uint3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uint3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uint3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uint3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uint3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: uint4
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_uint4 : IDisposable
	{
		IntPtr _intPtr;
		uint4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_uint4 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_uint4(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint4));
			_ptr = (uint4*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_uint4()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public uint4 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<uint4> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<uint4> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<uint4> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<uint4> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: long
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_long : IDisposable
	{
		IntPtr _intPtr;
		long* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_long from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_long(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(long));
			_ptr = (long*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_long()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public long this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<long> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<long> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<long> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<long> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: long1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_long1 : IDisposable
	{
		IntPtr _intPtr;
		long1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_long1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_long1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(long1));
			_ptr = (long1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_long1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public long1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<long1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<long1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<long1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<long1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: long2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_long2 : IDisposable
	{
		IntPtr _intPtr;
		long2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_long2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_long2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(long2));
			_ptr = (long2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_long2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public long2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<long2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<long2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<long2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<long2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ulong
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ulong : IDisposable
	{
		IntPtr _intPtr;
		ulong* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ulong from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ulong(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong));
			_ptr = (ulong*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ulong()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ulong this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ulong> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ulong> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ulong> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ulong> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ulong1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ulong1 : IDisposable
	{
		IntPtr _intPtr;
		ulong1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ulong1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ulong1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong1));
			_ptr = (ulong1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ulong1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ulong1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ulong1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ulong1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ulong1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ulong1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: ulong2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_ulong2 : IDisposable
	{
		IntPtr _intPtr;
		ulong2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_ulong2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_ulong2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong2));
			_ptr = (ulong2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_ulong2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public ulong2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<ulong2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<ulong2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<ulong2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<ulong2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: float
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_float : IDisposable
	{
		IntPtr _intPtr;
		float* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_float from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_float(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float));
			_ptr = (float*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_float()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public float this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<float> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<float> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<float> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<float> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: float1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_float1 : IDisposable
	{
		IntPtr _intPtr;
		float1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_float1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_float1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float1));
			_ptr = (float1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_float1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public float1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<float1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<float1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<float1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<float1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: float2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_float2 : IDisposable
	{
		IntPtr _intPtr;
		float2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_float2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_float2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float2));
			_ptr = (float2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_float2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public float2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<float2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<float2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<float2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<float2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: float3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_float3 : IDisposable
	{
		IntPtr _intPtr;
		float3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_float3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_float3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float3));
			_ptr = (float3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_float3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public float3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<float3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<float3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<float3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<float3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: float4
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_float4 : IDisposable
	{
		IntPtr _intPtr;
		float4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_float4 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_float4(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float4));
			_ptr = (float4*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_float4()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public float4 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<float4> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<float4> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<float4> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<float4> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: double
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_double : IDisposable
	{
		IntPtr _intPtr;
		double* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_double from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_double(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(double));
			_ptr = (double*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_double()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public double this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<double> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<double> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<double> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<double> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: double1
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_double1 : IDisposable
	{
		IntPtr _intPtr;
		double1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_double1 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_double1(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(double1));
			_ptr = (double1*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_double1()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public double1 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<double1> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<double1> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<double1> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<double1> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: double2
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_double2 : IDisposable
	{
		IntPtr _intPtr;
		double2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_double2 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_double2(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(double2));
			_ptr = (double2*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_double2()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public double2 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<double2> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<double2> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<double2> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<double2> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: cuDoubleComplex
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_cuDoubleComplex : IDisposable
	{
		IntPtr _intPtr;
		cuDoubleComplex* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_cuDoubleComplex from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_cuDoubleComplex(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuDoubleComplex));
			_ptr = (cuDoubleComplex*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_cuDoubleComplex()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public cuDoubleComplex this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<cuDoubleComplex> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<cuDoubleComplex> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<cuDoubleComplex> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<cuDoubleComplex> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: cuDoubleReal
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_cuDoubleReal : IDisposable
	{
		IntPtr _intPtr;
		cuDoubleReal* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_cuDoubleReal from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_cuDoubleReal(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuDoubleReal));
			_ptr = (cuDoubleReal*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_cuDoubleReal()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public cuDoubleReal this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<cuDoubleReal> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<cuDoubleReal> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<cuDoubleReal> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<cuDoubleReal> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: cuFloatComplex
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_cuFloatComplex : IDisposable
	{
		IntPtr _intPtr;
		cuFloatComplex* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_cuFloatComplex from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_cuFloatComplex(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuFloatComplex));
			_ptr = (cuFloatComplex*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_cuFloatComplex()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public cuFloatComplex this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<cuFloatComplex> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<cuFloatComplex> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<cuFloatComplex> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<cuFloatComplex> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: cuFloatReal
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_cuFloatReal : IDisposable
	{
		IntPtr _intPtr;
		cuFloatReal* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_cuFloatReal from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_cuFloatReal(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuFloatReal));
			_ptr = (cuFloatReal*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_cuFloatReal()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public cuFloatReal this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<cuFloatReal> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<cuFloatReal> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<cuFloatReal> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<cuFloatReal> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).<para/>
	/// Type: dim3
	/// </summary>
	public unsafe class CudaRegisteredHostMemory_dim3 : IDisposable
	{
		IntPtr _intPtr;
		dim3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory_dim3 from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory_dim3(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(dim3));
			_ptr = (dim3*)_intPtr;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory_dim3()
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
				if (_registered)//Unregister memory if it is registered
				{
					//Ignore possible errors
					res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
					_registered = false;
				}

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Pointer to pinned host memory.
		/// </summary>
		public IntPtr PinnedHostPointer
		{
			get { return _intPtr; }
		}

		/// <summary>
		/// Size in bytes
		/// </summary>
		public SizeT SizeInBytes
		{
			get { return _size * _typeSize; }
		}

		/// <summary>
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
		}

		/// <summary>
		/// Returns register status
		/// </summary>
		public bool IsRegisterd
		{
			get { return _registered; }
		}

		/// <summary>
		/// Access array per element.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public dim3 this[SizeT index]
		{
			get
			{
				return _ptr[index];
			}
			set
			{
				_ptr[index] = value;
			}
		}
		#endregion

		#region Synchron Copy Methods
		#region Array
		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(deviceArray, offset, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyToArray1D(CUarray deviceArray)
		{
			SynchronCopyToArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyToArray1D(CudaArray1D array)
		{
			SynchronCopyToArray1D(array.CUArray, 0);
		}

		/// <summary>
		/// Synchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyToArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyToArray1D(array.CUArray, offset);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(this._intPtr, deviceArray, offset, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		public void SynchronCopyFromArray1D(CUarray deviceArray)
		{
			SynchronCopyFromArray1D(deviceArray, 0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array)
		{
			SynchronCopyFromArray1D(array.CUArray,0);
		}

		/// <summary>
		/// Synchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="offset"></param>
		public void SynchronCopyFromArray1D(CudaArray1D array, SizeT offset)
		{
			SynchronCopyFromArray1D(array.CUArray, offset);
		}
		#endregion
		#region devicePtr
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr, this._intPtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToDevice(CudaDeviceVariable<dim3> devicePtr)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(this._intPtr, devicePtr, SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		public void SynchronCopyToHost(CudaDeviceVariable<dim3> devicePtr)
		{
			SynchronCopyToHost(devicePtr.DevicePointer);
		}
		#endregion
		#endregion

		#region Asynchron Copy Methods
		#region Array
		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoAAsync_v2(deviceArray, offset, this._intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoAAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyToArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyToArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy host to 1D Array
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream, SizeT offset)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAtoHAsync_v2(this._intPtr, deviceArray, offset, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CUarray deviceArray, CUstream stream)
		{
			AsyncCopyFromArray1D(deviceArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, 0);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="array"></param>
		/// <param name="stream"></param>
		/// <param name="offset"></param>
		public void AsyncCopyFromArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyFromArray1D(array.CUArray, stream, offset);
		}
		#endregion
		#region DevicePtr
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr, _intPtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<dim3> deviceVar, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(_intPtr, devicePtr, SizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<dim3> deviceVar, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, stream);
		}
		#endregion



		#endregion

		#region Methods
		/// <summary>
		/// Returns the CUdeviceptr for pinned host memory mapped to device memory space. Only valid if context is created with flag <see cref="CUCtxFlags.MapHost"/>
		/// </summary>
		/// <returns>Device Pointer</returns>
		public CUdeviceptr GetDevicePointer()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUdeviceptr ptr = new CUdeviceptr();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2(ref ptr, _intPtr, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return ptr;
		}

		/// <summary>
		/// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
		/// for the device(s) as specified by <c>Flags</c>. This memory range also is added
		/// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
		/// calls to functions such as <see cref="DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
		/// directly by the device, it can be read or written with much higher bandwidth 
		/// than pageable memory that has not been registered.  Page-locking excessive
		/// amounts of memory may degrade system performance, since it reduces the amount
		/// of memory available to the system for paging. As a result, this function is
		/// best used sparingly to register staging areas for data exchange between
		/// host and device.<para/>
		/// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
		/// The memory page-locked by this function must be unregistered with <see cref="Unregister"/>
		/// </summary>
		/// <param name="flags"></param>
		public void Register(CUMemHostRegisterFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostRegister(_intPtr, _size, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostRegister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = true;
		}

		/// <summary> 
		/// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
		/// The base address must be the same one specified to <see cref="Register"/>.
		/// </summary>
		public void Unregister()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostUnregister(_intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostUnregister", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_registered = false;
		}
		#endregion
	}

	
}
