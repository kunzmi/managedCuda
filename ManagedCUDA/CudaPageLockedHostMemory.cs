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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.
	/// </summary>
	/// <typeparam name="T">variable base type</typeparam>
	public class CudaPageLockedHostMemory<T> : IDisposable, IEnumerable<T> where T : struct
	{
		IntPtr _intPtr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaPageLockedHostMemory and allocates the memory on host. Using cuMemHostAlloc
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="allocFlags"></param>
		public CudaPageLockedHostMemory(SizeT size, CUMemHostAllocFlags allocFlags)
		{
			_intPtr = new IntPtr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(T));

			res = DriverAPINativeMethods.MemoryManagement.cuMemHostAlloc(ref _intPtr, _typeSize * size, allocFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostAlloc", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaPageLockedHostMemory and allocates the memory on host. Using cuMemAllocHost
		/// </summary>
		/// <param name="size">In elements</param>
		public CudaPageLockedHostMemory(SizeT size)
		{
			_intPtr = new IntPtr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(T));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocHost_v2(ref _intPtr, _typeSize * size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostAlloc", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaPageLockedHostMemory from an existing IntPtr. IntPtr must point to page locked memory!
		/// hostPointer won't be freed while disposing.
		/// </summary>
		/// <param name="hostPointer"></param>
		/// <param name="size">In elements</param>
		public CudaPageLockedHostMemory(IntPtr hostPointer, SizeT size)
		{
			_intPtr = hostPointer;
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(T));
			_isOwner = false;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaPageLockedHostMemory()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFreeHost(_intPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFreeHost", res));
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

		/// <summary>
		/// If the wrapper class instance is the owner of a CUDA handle, it will be destroyed while disposing.
		/// </summary>
		public bool IsOwner
		{
			get { return _isOwner; }
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
			SynchronCopyFromArray1D(array.CUArray, 0);
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
		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr">Pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		public void SynchronCopyToDevice(CUdeviceptr devicePtr, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(devicePtr + offsetDest, new IntPtr(this._intPtr.ToInt64() + (long)offsetSrc), aSizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy host to device
		/// </summary>
		/// <param name="devicePtr">Pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		public void SynchronCopyToDevice(CudaDeviceVariable<T> devicePtr, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes)
		{
			SynchronCopyToDevice(devicePtr.DevicePointer, offsetSrc, offsetDest, aSizeInBytes);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr">Pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		public void SynchronCopyToHost(CUdeviceptr devicePtr, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(new IntPtr(this._intPtr.ToInt64() + (long)offsetDest), devicePtr + offsetSrc, aSizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Synchron copy device to host
		/// </summary>
		/// <param name="devicePtr">Pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		public void SynchronCopyToHost(CudaDeviceVariable<T> devicePtr, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes)
		{
			SynchronCopyToHost(devicePtr.DevicePointer, offsetSrc, offsetDest, aSizeInBytes);
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
		/// <param name="offset">in bytes</param>
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
		/// <param name="offset">in bytes</param>
		public void AsyncCopyToArray1D(CudaArray1D array, CUstream stream, SizeT offset)
		{
			AsyncCopyToArray1D(array.CUArray, stream, offset);
		}

		/// <summary>
		/// Asynchron copy 1D Array to host
		/// </summary>
		/// <param name="deviceArray"></param>
		/// <param name="stream"></param>
		/// <param name="offset">bytes</param>
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
		/// <param name="offset">bytes</param>
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
		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="devicePtr">Pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr devicePtr, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(devicePtr + offsetDest, new IntPtr(_intPtr.ToInt64() + (long)offsetSrc), aSizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoDAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron Copy host to device
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<T> deviceVar, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes, CUstream stream)
		{
			AsyncCopyToDevice(deviceVar.DevicePointer, offsetSrc, offsetDest, aSizeInBytes, stream);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="devicePtr">Pointer to device memory</param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CUdeviceptr devicePtr, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoHAsync_v2(new IntPtr(_intPtr.ToInt64() + (long)offsetDest), devicePtr + offsetSrc, aSizeInBytes, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoHAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Asynchron copy device to host
		/// </summary>
		/// <param name="deviceVar"></param>
		/// <param name="offsetSrc">Offset to source pointer in bytes</param>
		/// <param name="offsetDest">Offset to destination pointer in bytes</param>
		/// <param name="aSizeInBytes">Bytes to copy</param>
		/// <param name="stream"></param>
		public void AsyncCopyFromDevice(CudaDeviceVariable<T> deviceVar, SizeT offsetSrc, SizeT offsetDest, SizeT aSizeInBytes, CUstream stream)
		{
			AsyncCopyFromDevice(deviceVar.DevicePointer, offsetSrc, offsetDest, aSizeInBytes, stream);
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
		/// Passes back the flags that were specified when allocating the pinned host buffer
		/// </summary>
		/// <returns></returns>
		public CUMemHostAllocFlags GetAllocFlags()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUMemHostAllocFlags flags = new CUMemHostAllocFlags();
			res = DriverAPINativeMethods.MemoryManagement.cuMemHostGetFlags(ref flags, _intPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemHostGetDevicePointer", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return flags;
		}
		#endregion

		#region IEnumerable
		IEnumerator<T> IEnumerable<T>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<T> enumerator = new CudaPageLockedHostMemoryEnumerator<T>(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaPageLockedHostMemoryEnumerator<T>(this);
			return enumerator;
		}

		#endregion
	}

	/// <summary>
	/// Enumerator class for CudaPageLockedHostMemory
	/// </summary>
	/// <typeparam name="T_"></typeparam>
	public class CudaPageLockedHostMemoryEnumerator<T_> : IEnumerator<T_> where T_ : struct
	{
		private CudaPageLockedHostMemory<T_> _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaPageLockedHostMemoryEnumerator(CudaPageLockedHostMemory<T_> memory)
		{
			_memory = memory;
		}

		void IDisposable.Dispose() { }

		/// <summary>
		/// 
		/// </summary>
		public void Reset()
		{
			_currentIndex = -1;
		}

		/// <summary>
		/// 
		/// </summary>
		public T_ Current
		{
			get { return _memory[_currentIndex]; }
		}

		/// <summary>
		/// 
		/// </summary>
		object IEnumerator.Current
		{
			get { return _memory[_currentIndex]; }
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public bool MoveNext()
		{
			_currentIndex += 1;
			if ((long)_currentIndex >= (long)_memory.Size)
				return false;
			else
				return true;
		}
	}
}
