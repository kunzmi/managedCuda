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
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// cuMemHostRegister doesn't work with managed memory (e.g. normal C# arrays). But you can use cuMemHostRegister for
	/// natively allocated memory (Marshal.AllocHGlobal, or a native dll).
	/// </summary>
	/// <typeparam name="T">variable base type</typeparam>
	public class CudaRegisteredHostMemory<T> : IDisposable where T : struct
	{
		IntPtr _intPtr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool _registered;
		bool disposed;

		#region Constructor
		/// <summary>
		/// Creates a new CudaRegisteredHostMemory from an existing IntPtr. IntPtr must be page size aligned (4KBytes)!
		/// </summary>
		/// <param name="hostPointer">must be page size aligned (4KBytes)</param>
		/// <param name="size">In elements</param>
		public CudaRegisteredHostMemory(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(T));

		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRegisteredHostMemory()
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
		/// Access array per element.<para/>
		/// Each single access hast to trespass the managed/unmanged memory barrier. Access is therefor rather slow.
		/// </summary>
		/// <param name="index">index in elements</param>
		/// <returns></returns>
		public T this[SizeT index]
		{
			get 
			{
				IntPtr position = new IntPtr((long)index * (long)_typeSize + _intPtr.ToInt64());
				T ret = (T)Marshal.PtrToStructure(position, typeof(T));
				return ret;
			}
			set
			{
				IntPtr position = new IntPtr((long)index * (long)_typeSize + _intPtr.ToInt64());
				Marshal.StructureToPtr(value, position, false);
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
		public void SynchronCopyToDevice(CudaDeviceVariable<T> devicePtr)
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
		public void SynchronCopyToHost(CudaDeviceVariable<T> devicePtr)
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
		public void AsyncCopyToDevice(CudaDeviceVariable<T> deviceVar, CUstream stream)
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
		public void AsyncCopyFromDevice(CudaDeviceVariable<T> deviceVar, CUstream stream)
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
