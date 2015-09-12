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
	/// Type: byte
	/// </summary>
	public unsafe class CudaManagedMemory_byte: IDisposable, IEnumerable<byte>
	{
		CUdeviceptr _devPtr;
		byte* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_byte(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(byte));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (byte*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_byte(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(byte));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (byte*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_byte(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_byte()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator byte(CudaManagedMemory_byte d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<byte> IEnumerable<byte>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<byte> enumerator = new CudaManagedMemoryEnumerator_byte(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_byte(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_byte
	/// </summary>
	public class CudaManagedMemoryEnumerator_byte : IEnumerator<byte>
	{
		private CudaManagedMemory_byte _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_byte(CudaManagedMemory_byte memory)
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
		public byte Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uchar1
	/// </summary>
	public unsafe class CudaManagedMemory_uchar1: IDisposable, IEnumerable<uchar1>
	{
		CUdeviceptr _devPtr;
		uchar1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uchar1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uchar1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uchar1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uchar1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uchar1(CudaManagedMemory_uchar1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uchar1> IEnumerable<uchar1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uchar1> enumerator = new CudaManagedMemoryEnumerator_uchar1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uchar1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uchar1
	/// </summary>
	public class CudaManagedMemoryEnumerator_uchar1 : IEnumerator<uchar1>
	{
		private CudaManagedMemory_uchar1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uchar1(CudaManagedMemory_uchar1 memory)
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
		public uchar1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uchar2
	/// </summary>
	public unsafe class CudaManagedMemory_uchar2: IDisposable, IEnumerable<uchar2>
	{
		CUdeviceptr _devPtr;
		uchar2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uchar2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uchar2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uchar2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uchar2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uchar2(CudaManagedMemory_uchar2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uchar2> IEnumerable<uchar2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uchar2> enumerator = new CudaManagedMemoryEnumerator_uchar2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uchar2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uchar2
	/// </summary>
	public class CudaManagedMemoryEnumerator_uchar2 : IEnumerator<uchar2>
	{
		private CudaManagedMemory_uchar2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uchar2(CudaManagedMemory_uchar2 memory)
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
		public uchar2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uchar3
	/// </summary>
	public unsafe class CudaManagedMemory_uchar3: IDisposable, IEnumerable<uchar3>
	{
		CUdeviceptr _devPtr;
		uchar3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uchar3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uchar3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uchar3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uchar3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uchar3(CudaManagedMemory_uchar3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uchar3> IEnumerable<uchar3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uchar3> enumerator = new CudaManagedMemoryEnumerator_uchar3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uchar3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uchar3
	/// </summary>
	public class CudaManagedMemoryEnumerator_uchar3 : IEnumerator<uchar3>
	{
		private CudaManagedMemory_uchar3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uchar3(CudaManagedMemory_uchar3 memory)
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
		public uchar3 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uchar4
	/// </summary>
	public unsafe class CudaManagedMemory_uchar4: IDisposable, IEnumerable<uchar4>
	{
		CUdeviceptr _devPtr;
		uchar4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uchar4(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar4));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uchar4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar4(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uchar4));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uchar4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uchar4(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uchar4()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uchar4(CudaManagedMemory_uchar4 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uchar4> IEnumerable<uchar4>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uchar4> enumerator = new CudaManagedMemoryEnumerator_uchar4(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uchar4(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uchar4
	/// </summary>
	public class CudaManagedMemoryEnumerator_uchar4 : IEnumerator<uchar4>
	{
		private CudaManagedMemory_uchar4 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uchar4(CudaManagedMemory_uchar4 memory)
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
		public uchar4 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: sbyte
	/// </summary>
	public unsafe class CudaManagedMemory_sbyte: IDisposable, IEnumerable<sbyte>
	{
		CUdeviceptr _devPtr;
		sbyte* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_sbyte(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(sbyte));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (sbyte*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_sbyte(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(sbyte));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (sbyte*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_sbyte(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_sbyte()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator sbyte(CudaManagedMemory_sbyte d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<sbyte> IEnumerable<sbyte>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<sbyte> enumerator = new CudaManagedMemoryEnumerator_sbyte(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_sbyte(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_sbyte
	/// </summary>
	public class CudaManagedMemoryEnumerator_sbyte : IEnumerator<sbyte>
	{
		private CudaManagedMemory_sbyte _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_sbyte(CudaManagedMemory_sbyte memory)
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
		public sbyte Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: char1
	/// </summary>
	public unsafe class CudaManagedMemory_char1: IDisposable, IEnumerable<char1>
	{
		CUdeviceptr _devPtr;
		char1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_char1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (char1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(char1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (char1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_char1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator char1(CudaManagedMemory_char1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<char1> IEnumerable<char1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<char1> enumerator = new CudaManagedMemoryEnumerator_char1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_char1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_char1
	/// </summary>
	public class CudaManagedMemoryEnumerator_char1 : IEnumerator<char1>
	{
		private CudaManagedMemory_char1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_char1(CudaManagedMemory_char1 memory)
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
		public char1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: char2
	/// </summary>
	public unsafe class CudaManagedMemory_char2: IDisposable, IEnumerable<char2>
	{
		CUdeviceptr _devPtr;
		char2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_char2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (char2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(char2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (char2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_char2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator char2(CudaManagedMemory_char2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<char2> IEnumerable<char2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<char2> enumerator = new CudaManagedMemoryEnumerator_char2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_char2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_char2
	/// </summary>
	public class CudaManagedMemoryEnumerator_char2 : IEnumerator<char2>
	{
		private CudaManagedMemory_char2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_char2(CudaManagedMemory_char2 memory)
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
		public char2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: char3
	/// </summary>
	public unsafe class CudaManagedMemory_char3: IDisposable, IEnumerable<char3>
	{
		CUdeviceptr _devPtr;
		char3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_char3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (char3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(char3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (char3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_char3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator char3(CudaManagedMemory_char3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<char3> IEnumerable<char3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<char3> enumerator = new CudaManagedMemoryEnumerator_char3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_char3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_char3
	/// </summary>
	public class CudaManagedMemoryEnumerator_char3 : IEnumerator<char3>
	{
		private CudaManagedMemory_char3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_char3(CudaManagedMemory_char3 memory)
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
		public char3 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: char4
	/// </summary>
	public unsafe class CudaManagedMemory_char4: IDisposable, IEnumerable<char4>
	{
		CUdeviceptr _devPtr;
		char4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_char4(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(char4));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (char4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char4(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(char4));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (char4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_char4(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_char4()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator char4(CudaManagedMemory_char4 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<char4> IEnumerable<char4>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<char4> enumerator = new CudaManagedMemoryEnumerator_char4(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_char4(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_char4
	/// </summary>
	public class CudaManagedMemoryEnumerator_char4 : IEnumerator<char4>
	{
		private CudaManagedMemory_char4 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_char4(CudaManagedMemory_char4 memory)
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
		public char4 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: short
	/// </summary>
	public unsafe class CudaManagedMemory_short: IDisposable, IEnumerable<short>
	{
		CUdeviceptr _devPtr;
		short* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_short(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (short*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(short));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (short*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_short()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator short(CudaManagedMemory_short d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<short> IEnumerable<short>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<short> enumerator = new CudaManagedMemoryEnumerator_short(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_short(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_short
	/// </summary>
	public class CudaManagedMemoryEnumerator_short : IEnumerator<short>
	{
		private CudaManagedMemory_short _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_short(CudaManagedMemory_short memory)
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
		public short Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: short1
	/// </summary>
	public unsafe class CudaManagedMemory_short1: IDisposable, IEnumerable<short1>
	{
		CUdeviceptr _devPtr;
		short1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_short1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (short1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(short1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (short1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_short1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator short1(CudaManagedMemory_short1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<short1> IEnumerable<short1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<short1> enumerator = new CudaManagedMemoryEnumerator_short1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_short1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_short1
	/// </summary>
	public class CudaManagedMemoryEnumerator_short1 : IEnumerator<short1>
	{
		private CudaManagedMemory_short1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_short1(CudaManagedMemory_short1 memory)
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
		public short1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: short2
	/// </summary>
	public unsafe class CudaManagedMemory_short2: IDisposable, IEnumerable<short2>
	{
		CUdeviceptr _devPtr;
		short2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_short2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (short2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(short2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (short2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_short2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator short2(CudaManagedMemory_short2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<short2> IEnumerable<short2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<short2> enumerator = new CudaManagedMemoryEnumerator_short2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_short2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_short2
	/// </summary>
	public class CudaManagedMemoryEnumerator_short2 : IEnumerator<short2>
	{
		private CudaManagedMemory_short2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_short2(CudaManagedMemory_short2 memory)
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
		public short2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: short3
	/// </summary>
	public unsafe class CudaManagedMemory_short3: IDisposable, IEnumerable<short3>
	{
		CUdeviceptr _devPtr;
		short3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_short3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (short3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(short3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (short3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_short3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator short3(CudaManagedMemory_short3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<short3> IEnumerable<short3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<short3> enumerator = new CudaManagedMemoryEnumerator_short3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_short3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_short3
	/// </summary>
	public class CudaManagedMemoryEnumerator_short3 : IEnumerator<short3>
	{
		private CudaManagedMemory_short3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_short3(CudaManagedMemory_short3 memory)
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
		public short3 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: short4
	/// </summary>
	public unsafe class CudaManagedMemory_short4: IDisposable, IEnumerable<short4>
	{
		CUdeviceptr _devPtr;
		short4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_short4(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(short4));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (short4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short4(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(short4));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (short4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_short4(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_short4()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator short4(CudaManagedMemory_short4 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<short4> IEnumerable<short4>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<short4> enumerator = new CudaManagedMemoryEnumerator_short4(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_short4(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_short4
	/// </summary>
	public class CudaManagedMemoryEnumerator_short4 : IEnumerator<short4>
	{
		private CudaManagedMemory_short4 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_short4(CudaManagedMemory_short4 memory)
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
		public short4 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ushort
	/// </summary>
	public unsafe class CudaManagedMemory_ushort: IDisposable, IEnumerable<ushort>
	{
		CUdeviceptr _devPtr;
		ushort* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ushort(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ushort*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ushort*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ushort()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ushort(CudaManagedMemory_ushort d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ushort> IEnumerable<ushort>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ushort> enumerator = new CudaManagedMemoryEnumerator_ushort(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ushort(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ushort
	/// </summary>
	public class CudaManagedMemoryEnumerator_ushort : IEnumerator<ushort>
	{
		private CudaManagedMemory_ushort _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ushort(CudaManagedMemory_ushort memory)
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
		public ushort Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ushort1
	/// </summary>
	public unsafe class CudaManagedMemory_ushort1: IDisposable, IEnumerable<ushort1>
	{
		CUdeviceptr _devPtr;
		ushort1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ushort1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ushort1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ushort1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ushort1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ushort1(CudaManagedMemory_ushort1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ushort1> IEnumerable<ushort1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ushort1> enumerator = new CudaManagedMemoryEnumerator_ushort1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ushort1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ushort1
	/// </summary>
	public class CudaManagedMemoryEnumerator_ushort1 : IEnumerator<ushort1>
	{
		private CudaManagedMemory_ushort1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ushort1(CudaManagedMemory_ushort1 memory)
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
		public ushort1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ushort2
	/// </summary>
	public unsafe class CudaManagedMemory_ushort2: IDisposable, IEnumerable<ushort2>
	{
		CUdeviceptr _devPtr;
		ushort2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ushort2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ushort2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ushort2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ushort2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ushort2(CudaManagedMemory_ushort2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ushort2> IEnumerable<ushort2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ushort2> enumerator = new CudaManagedMemoryEnumerator_ushort2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ushort2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ushort2
	/// </summary>
	public class CudaManagedMemoryEnumerator_ushort2 : IEnumerator<ushort2>
	{
		private CudaManagedMemory_ushort2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ushort2(CudaManagedMemory_ushort2 memory)
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
		public ushort2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ushort3
	/// </summary>
	public unsafe class CudaManagedMemory_ushort3: IDisposable, IEnumerable<ushort3>
	{
		CUdeviceptr _devPtr;
		ushort3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ushort3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ushort3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ushort3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ushort3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ushort3(CudaManagedMemory_ushort3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ushort3> IEnumerable<ushort3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ushort3> enumerator = new CudaManagedMemoryEnumerator_ushort3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ushort3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ushort3
	/// </summary>
	public class CudaManagedMemoryEnumerator_ushort3 : IEnumerator<ushort3>
	{
		private CudaManagedMemory_ushort3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ushort3(CudaManagedMemory_ushort3 memory)
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
		public ushort3 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ushort4
	/// </summary>
	public unsafe class CudaManagedMemory_ushort4: IDisposable, IEnumerable<ushort4>
	{
		CUdeviceptr _devPtr;
		ushort4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ushort4(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort4));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ushort4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort4(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ushort4));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ushort4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ushort4(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ushort4()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ushort4(CudaManagedMemory_ushort4 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ushort4> IEnumerable<ushort4>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ushort4> enumerator = new CudaManagedMemoryEnumerator_ushort4(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ushort4(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ushort4
	/// </summary>
	public class CudaManagedMemoryEnumerator_ushort4 : IEnumerator<ushort4>
	{
		private CudaManagedMemory_ushort4 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ushort4(CudaManagedMemory_ushort4 memory)
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
		public ushort4 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: int
	/// </summary>
	public unsafe class CudaManagedMemory_int: IDisposable, IEnumerable<int>
	{
		CUdeviceptr _devPtr;
		int* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_int(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (int*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(int));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (int*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_int()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator int(CudaManagedMemory_int d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<int> IEnumerable<int>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<int> enumerator = new CudaManagedMemoryEnumerator_int(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_int(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_int
	/// </summary>
	public class CudaManagedMemoryEnumerator_int : IEnumerator<int>
	{
		private CudaManagedMemory_int _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_int(CudaManagedMemory_int memory)
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
		public int Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: int1
	/// </summary>
	public unsafe class CudaManagedMemory_int1: IDisposable, IEnumerable<int1>
	{
		CUdeviceptr _devPtr;
		int1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_int1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (int1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(int1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (int1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_int1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator int1(CudaManagedMemory_int1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<int1> IEnumerable<int1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<int1> enumerator = new CudaManagedMemoryEnumerator_int1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_int1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_int1
	/// </summary>
	public class CudaManagedMemoryEnumerator_int1 : IEnumerator<int1>
	{
		private CudaManagedMemory_int1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_int1(CudaManagedMemory_int1 memory)
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
		public int1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: int2
	/// </summary>
	public unsafe class CudaManagedMemory_int2: IDisposable, IEnumerable<int2>
	{
		CUdeviceptr _devPtr;
		int2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_int2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (int2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(int2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (int2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_int2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator int2(CudaManagedMemory_int2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<int2> IEnumerable<int2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<int2> enumerator = new CudaManagedMemoryEnumerator_int2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_int2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_int2
	/// </summary>
	public class CudaManagedMemoryEnumerator_int2 : IEnumerator<int2>
	{
		private CudaManagedMemory_int2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_int2(CudaManagedMemory_int2 memory)
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
		public int2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: int3
	/// </summary>
	public unsafe class CudaManagedMemory_int3: IDisposable, IEnumerable<int3>
	{
		CUdeviceptr _devPtr;
		int3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_int3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (int3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(int3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (int3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_int3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator int3(CudaManagedMemory_int3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<int3> IEnumerable<int3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<int3> enumerator = new CudaManagedMemoryEnumerator_int3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_int3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_int3
	/// </summary>
	public class CudaManagedMemoryEnumerator_int3 : IEnumerator<int3>
	{
		private CudaManagedMemory_int3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_int3(CudaManagedMemory_int3 memory)
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
		public int3 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: int4
	/// </summary>
	public unsafe class CudaManagedMemory_int4: IDisposable, IEnumerable<int4>
	{
		CUdeviceptr _devPtr;
		int4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_int4(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(int4));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (int4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int4(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(int4));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (int4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_int4(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_int4()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator int4(CudaManagedMemory_int4 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<int4> IEnumerable<int4>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<int4> enumerator = new CudaManagedMemoryEnumerator_int4(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_int4(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_int4
	/// </summary>
	public class CudaManagedMemoryEnumerator_int4 : IEnumerator<int4>
	{
		private CudaManagedMemory_int4 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_int4(CudaManagedMemory_int4 memory)
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
		public int4 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uint
	/// </summary>
	public unsafe class CudaManagedMemory_uint: IDisposable, IEnumerable<uint>
	{
		CUdeviceptr _devPtr;
		uint* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uint(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uint*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uint*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uint()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uint(CudaManagedMemory_uint d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uint> IEnumerable<uint>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uint> enumerator = new CudaManagedMemoryEnumerator_uint(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uint(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uint
	/// </summary>
	public class CudaManagedMemoryEnumerator_uint : IEnumerator<uint>
	{
		private CudaManagedMemory_uint _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uint(CudaManagedMemory_uint memory)
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
		public uint Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uint1
	/// </summary>
	public unsafe class CudaManagedMemory_uint1: IDisposable, IEnumerable<uint1>
	{
		CUdeviceptr _devPtr;
		uint1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uint1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uint1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uint1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uint1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uint1(CudaManagedMemory_uint1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uint1> IEnumerable<uint1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uint1> enumerator = new CudaManagedMemoryEnumerator_uint1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uint1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uint1
	/// </summary>
	public class CudaManagedMemoryEnumerator_uint1 : IEnumerator<uint1>
	{
		private CudaManagedMemory_uint1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uint1(CudaManagedMemory_uint1 memory)
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
		public uint1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uint2
	/// </summary>
	public unsafe class CudaManagedMemory_uint2: IDisposable, IEnumerable<uint2>
	{
		CUdeviceptr _devPtr;
		uint2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uint2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uint2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uint2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uint2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uint2(CudaManagedMemory_uint2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uint2> IEnumerable<uint2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uint2> enumerator = new CudaManagedMemoryEnumerator_uint2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uint2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uint2
	/// </summary>
	public class CudaManagedMemoryEnumerator_uint2 : IEnumerator<uint2>
	{
		private CudaManagedMemory_uint2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uint2(CudaManagedMemory_uint2 memory)
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
		public uint2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uint3
	/// </summary>
	public unsafe class CudaManagedMemory_uint3: IDisposable, IEnumerable<uint3>
	{
		CUdeviceptr _devPtr;
		uint3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uint3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uint3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uint3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uint3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uint3(CudaManagedMemory_uint3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uint3> IEnumerable<uint3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uint3> enumerator = new CudaManagedMemoryEnumerator_uint3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uint3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uint3
	/// </summary>
	public class CudaManagedMemoryEnumerator_uint3 : IEnumerator<uint3>
	{
		private CudaManagedMemory_uint3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uint3(CudaManagedMemory_uint3 memory)
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
		public uint3 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: uint4
	/// </summary>
	public unsafe class CudaManagedMemory_uint4: IDisposable, IEnumerable<uint4>
	{
		CUdeviceptr _devPtr;
		uint4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_uint4(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint4));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (uint4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint4(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(uint4));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (uint4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_uint4(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_uint4()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator uint4(CudaManagedMemory_uint4 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<uint4> IEnumerable<uint4>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<uint4> enumerator = new CudaManagedMemoryEnumerator_uint4(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_uint4(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_uint4
	/// </summary>
	public class CudaManagedMemoryEnumerator_uint4 : IEnumerator<uint4>
	{
		private CudaManagedMemory_uint4 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_uint4(CudaManagedMemory_uint4 memory)
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
		public uint4 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: long
	/// </summary>
	public unsafe class CudaManagedMemory_long: IDisposable, IEnumerable<long>
	{
		CUdeviceptr _devPtr;
		long* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_long(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(long));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (long*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_long(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(long));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (long*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_long(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_long()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator long(CudaManagedMemory_long d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<long> IEnumerable<long>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<long> enumerator = new CudaManagedMemoryEnumerator_long(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_long(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_long
	/// </summary>
	public class CudaManagedMemoryEnumerator_long : IEnumerator<long>
	{
		private CudaManagedMemory_long _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_long(CudaManagedMemory_long memory)
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
		public long Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: long1
	/// </summary>
	public unsafe class CudaManagedMemory_long1: IDisposable, IEnumerable<long1>
	{
		CUdeviceptr _devPtr;
		long1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_long1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(long1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (long1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_long1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(long1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (long1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_long1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_long1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator long1(CudaManagedMemory_long1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<long1> IEnumerable<long1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<long1> enumerator = new CudaManagedMemoryEnumerator_long1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_long1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_long1
	/// </summary>
	public class CudaManagedMemoryEnumerator_long1 : IEnumerator<long1>
	{
		private CudaManagedMemory_long1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_long1(CudaManagedMemory_long1 memory)
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
		public long1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: long2
	/// </summary>
	public unsafe class CudaManagedMemory_long2: IDisposable, IEnumerable<long2>
	{
		CUdeviceptr _devPtr;
		long2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_long2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(long2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (long2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_long2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(long2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (long2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_long2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_long2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator long2(CudaManagedMemory_long2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<long2> IEnumerable<long2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<long2> enumerator = new CudaManagedMemoryEnumerator_long2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_long2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_long2
	/// </summary>
	public class CudaManagedMemoryEnumerator_long2 : IEnumerator<long2>
	{
		private CudaManagedMemory_long2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_long2(CudaManagedMemory_long2 memory)
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
		public long2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ulong
	/// </summary>
	public unsafe class CudaManagedMemory_ulong: IDisposable, IEnumerable<ulong>
	{
		CUdeviceptr _devPtr;
		ulong* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ulong(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ulong*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ulong(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ulong*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ulong(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ulong()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ulong(CudaManagedMemory_ulong d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ulong> IEnumerable<ulong>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ulong> enumerator = new CudaManagedMemoryEnumerator_ulong(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ulong(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ulong
	/// </summary>
	public class CudaManagedMemoryEnumerator_ulong : IEnumerator<ulong>
	{
		private CudaManagedMemory_ulong _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ulong(CudaManagedMemory_ulong memory)
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
		public ulong Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ulong1
	/// </summary>
	public unsafe class CudaManagedMemory_ulong1: IDisposable, IEnumerable<ulong1>
	{
		CUdeviceptr _devPtr;
		ulong1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ulong1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ulong1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ulong1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ulong1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ulong1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ulong1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ulong1(CudaManagedMemory_ulong1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ulong1> IEnumerable<ulong1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ulong1> enumerator = new CudaManagedMemoryEnumerator_ulong1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ulong1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ulong1
	/// </summary>
	public class CudaManagedMemoryEnumerator_ulong1 : IEnumerator<ulong1>
	{
		private CudaManagedMemory_ulong1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ulong1(CudaManagedMemory_ulong1 memory)
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
		public ulong1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: ulong2
	/// </summary>
	public unsafe class CudaManagedMemory_ulong2: IDisposable, IEnumerable<ulong2>
	{
		CUdeviceptr _devPtr;
		ulong2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_ulong2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (ulong2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ulong2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(ulong2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (ulong2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_ulong2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_ulong2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator ulong2(CudaManagedMemory_ulong2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<ulong2> IEnumerable<ulong2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<ulong2> enumerator = new CudaManagedMemoryEnumerator_ulong2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_ulong2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_ulong2
	/// </summary>
	public class CudaManagedMemoryEnumerator_ulong2 : IEnumerator<ulong2>
	{
		private CudaManagedMemory_ulong2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_ulong2(CudaManagedMemory_ulong2 memory)
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
		public ulong2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: float
	/// </summary>
	public unsafe class CudaManagedMemory_float: IDisposable, IEnumerable<float>
	{
		CUdeviceptr _devPtr;
		float* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_float(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (float*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(float));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (float*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_float()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator float(CudaManagedMemory_float d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<float> IEnumerable<float>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<float> enumerator = new CudaManagedMemoryEnumerator_float(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_float(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_float
	/// </summary>
	public class CudaManagedMemoryEnumerator_float : IEnumerator<float>
	{
		private CudaManagedMemory_float _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_float(CudaManagedMemory_float memory)
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
		public float Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: float1
	/// </summary>
	public unsafe class CudaManagedMemory_float1: IDisposable, IEnumerable<float1>
	{
		CUdeviceptr _devPtr;
		float1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_float1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (float1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(float1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (float1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_float1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator float1(CudaManagedMemory_float1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<float1> IEnumerable<float1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<float1> enumerator = new CudaManagedMemoryEnumerator_float1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_float1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_float1
	/// </summary>
	public class CudaManagedMemoryEnumerator_float1 : IEnumerator<float1>
	{
		private CudaManagedMemory_float1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_float1(CudaManagedMemory_float1 memory)
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
		public float1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: float2
	/// </summary>
	public unsafe class CudaManagedMemory_float2: IDisposable, IEnumerable<float2>
	{
		CUdeviceptr _devPtr;
		float2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_float2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (float2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(float2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (float2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_float2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator float2(CudaManagedMemory_float2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<float2> IEnumerable<float2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<float2> enumerator = new CudaManagedMemoryEnumerator_float2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_float2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_float2
	/// </summary>
	public class CudaManagedMemoryEnumerator_float2 : IEnumerator<float2>
	{
		private CudaManagedMemory_float2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_float2(CudaManagedMemory_float2 memory)
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
		public float2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: float3
	/// </summary>
	public unsafe class CudaManagedMemory_float3: IDisposable, IEnumerable<float3>
	{
		CUdeviceptr _devPtr;
		float3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_float3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (float3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(float3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (float3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_float3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator float3(CudaManagedMemory_float3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<float3> IEnumerable<float3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<float3> enumerator = new CudaManagedMemoryEnumerator_float3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_float3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_float3
	/// </summary>
	public class CudaManagedMemoryEnumerator_float3 : IEnumerator<float3>
	{
		private CudaManagedMemory_float3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_float3(CudaManagedMemory_float3 memory)
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
		public float3 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: float4
	/// </summary>
	public unsafe class CudaManagedMemory_float4: IDisposable, IEnumerable<float4>
	{
		CUdeviceptr _devPtr;
		float4* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_float4(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(float4));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (float4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float4(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(float4));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (float4*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_float4(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_float4()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator float4(CudaManagedMemory_float4 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<float4> IEnumerable<float4>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<float4> enumerator = new CudaManagedMemoryEnumerator_float4(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_float4(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_float4
	/// </summary>
	public class CudaManagedMemoryEnumerator_float4 : IEnumerator<float4>
	{
		private CudaManagedMemory_float4 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_float4(CudaManagedMemory_float4 memory)
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
		public float4 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: double
	/// </summary>
	public unsafe class CudaManagedMemory_double: IDisposable, IEnumerable<double>
	{
		CUdeviceptr _devPtr;
		double* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_double(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(double));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (double*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_double(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(double));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (double*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_double(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_double()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator double(CudaManagedMemory_double d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<double> IEnumerable<double>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<double> enumerator = new CudaManagedMemoryEnumerator_double(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_double(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_double
	/// </summary>
	public class CudaManagedMemoryEnumerator_double : IEnumerator<double>
	{
		private CudaManagedMemory_double _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_double(CudaManagedMemory_double memory)
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
		public double Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: double1
	/// </summary>
	public unsafe class CudaManagedMemory_double1: IDisposable, IEnumerable<double1>
	{
		CUdeviceptr _devPtr;
		double1* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_double1(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(double1));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (double1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_double1(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(double1));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (double1*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_double1(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_double1()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator double1(CudaManagedMemory_double1 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<double1> IEnumerable<double1>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<double1> enumerator = new CudaManagedMemoryEnumerator_double1(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_double1(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_double1
	/// </summary>
	public class CudaManagedMemoryEnumerator_double1 : IEnumerator<double1>
	{
		private CudaManagedMemory_double1 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_double1(CudaManagedMemory_double1 memory)
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
		public double1 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: double2
	/// </summary>
	public unsafe class CudaManagedMemory_double2: IDisposable, IEnumerable<double2>
	{
		CUdeviceptr _devPtr;
		double2* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_double2(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(double2));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (double2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_double2(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(double2));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (double2*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_double2(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_double2()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator double2(CudaManagedMemory_double2 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<double2> IEnumerable<double2>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<double2> enumerator = new CudaManagedMemoryEnumerator_double2(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_double2(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_double2
	/// </summary>
	public class CudaManagedMemoryEnumerator_double2 : IEnumerator<double2>
	{
		private CudaManagedMemory_double2 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_double2(CudaManagedMemory_double2 memory)
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
		public double2 Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: cuDoubleComplex
	/// </summary>
	public unsafe class CudaManagedMemory_cuDoubleComplex: IDisposable, IEnumerable<cuDoubleComplex>
	{
		CUdeviceptr _devPtr;
		cuDoubleComplex* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_cuDoubleComplex(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuDoubleComplex));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (cuDoubleComplex*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuDoubleComplex(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuDoubleComplex));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (cuDoubleComplex*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuDoubleComplex(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_cuDoubleComplex()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator cuDoubleComplex(CudaManagedMemory_cuDoubleComplex d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<cuDoubleComplex> IEnumerable<cuDoubleComplex>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<cuDoubleComplex> enumerator = new CudaManagedMemoryEnumerator_cuDoubleComplex(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_cuDoubleComplex(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_cuDoubleComplex
	/// </summary>
	public class CudaManagedMemoryEnumerator_cuDoubleComplex : IEnumerator<cuDoubleComplex>
	{
		private CudaManagedMemory_cuDoubleComplex _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_cuDoubleComplex(CudaManagedMemory_cuDoubleComplex memory)
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
		public cuDoubleComplex Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: cuDoubleReal
	/// </summary>
	public unsafe class CudaManagedMemory_cuDoubleReal: IDisposable, IEnumerable<cuDoubleReal>
	{
		CUdeviceptr _devPtr;
		cuDoubleReal* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_cuDoubleReal(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuDoubleReal));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (cuDoubleReal*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuDoubleReal(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuDoubleReal));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (cuDoubleReal*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuDoubleReal(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_cuDoubleReal()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator cuDoubleReal(CudaManagedMemory_cuDoubleReal d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<cuDoubleReal> IEnumerable<cuDoubleReal>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<cuDoubleReal> enumerator = new CudaManagedMemoryEnumerator_cuDoubleReal(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_cuDoubleReal(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_cuDoubleReal
	/// </summary>
	public class CudaManagedMemoryEnumerator_cuDoubleReal : IEnumerator<cuDoubleReal>
	{
		private CudaManagedMemory_cuDoubleReal _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_cuDoubleReal(CudaManagedMemory_cuDoubleReal memory)
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
		public cuDoubleReal Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: cuFloatComplex
	/// </summary>
	public unsafe class CudaManagedMemory_cuFloatComplex: IDisposable, IEnumerable<cuFloatComplex>
	{
		CUdeviceptr _devPtr;
		cuFloatComplex* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_cuFloatComplex(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuFloatComplex));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (cuFloatComplex*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuFloatComplex(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuFloatComplex));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (cuFloatComplex*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuFloatComplex(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_cuFloatComplex()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator cuFloatComplex(CudaManagedMemory_cuFloatComplex d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<cuFloatComplex> IEnumerable<cuFloatComplex>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<cuFloatComplex> enumerator = new CudaManagedMemoryEnumerator_cuFloatComplex(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_cuFloatComplex(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_cuFloatComplex
	/// </summary>
	public class CudaManagedMemoryEnumerator_cuFloatComplex : IEnumerator<cuFloatComplex>
	{
		private CudaManagedMemory_cuFloatComplex _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_cuFloatComplex(CudaManagedMemory_cuFloatComplex memory)
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
		public cuFloatComplex Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: cuFloatReal
	/// </summary>
	public unsafe class CudaManagedMemory_cuFloatReal: IDisposable, IEnumerable<cuFloatReal>
	{
		CUdeviceptr _devPtr;
		cuFloatReal* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_cuFloatReal(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuFloatReal));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (cuFloatReal*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuFloatReal(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(cuFloatReal));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (cuFloatReal*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_cuFloatReal(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_cuFloatReal()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator cuFloatReal(CudaManagedMemory_cuFloatReal d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<cuFloatReal> IEnumerable<cuFloatReal>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<cuFloatReal> enumerator = new CudaManagedMemoryEnumerator_cuFloatReal(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_cuFloatReal(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_cuFloatReal
	/// </summary>
	public class CudaManagedMemoryEnumerator_cuFloatReal : IEnumerator<cuFloatReal>
	{
		private CudaManagedMemory_cuFloatReal _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_cuFloatReal(CudaManagedMemory_cuFloatReal memory)
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
		public cuFloatReal Current
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

	
	/// <summary>
	/// A variable located in page locked (pinned) host memory. Use this type of variabe for asynchronous memcpy.<para/>
	/// Type: dim3
	/// </summary>
	public unsafe class CudaManagedMemory_dim3: IDisposable, IEnumerable<dim3>
	{
		CUdeviceptr _devPtr;
		dim3* _ptr;
		SizeT _size = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructor
		/// <summary>
		/// Creates a new CudaManagedMemory and allocates the memory on host/device.
		/// </summary>
		/// <param name="size">In elements</param>
		/// <param name="attachFlags"></param>
		public CudaManagedMemory_dim3(SizeT size, CUmemAttach_flags attachFlags)
		{
			_devPtr = new CUdeviceptr();
			_size = size;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(dim3));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocManaged(ref _devPtr, _typeSize * size, attachFlags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocManaged", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptr = (dim3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="module">The module where the variable is defined in.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_dim3(CUmodule module, string name)
		{
			_devPtr = new CUdeviceptr();
			SizeT _sizeInBytes = new SizeT();
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref _devPtr, ref _sizeInBytes, module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}. Name: {3}, Size (in bytes): {4}", DateTime.Now, "cuModuleGetGlobal_v2", res, name, _sizeInBytes.ToString()));
			if (res != CUResult.Success) throw new CudaException(res);

			_typeSize = (SizeT)Marshal.SizeOf(typeof(dim3));
			_size = _sizeInBytes / _typeSize;

			if (_sizeInBytes != _size * _typeSize)
				throw new CudaException("Variable size is not a multiple of its type size.");

			_ptr = (dim3*) (UIntPtr)_devPtr.Pointer;
			_isOwner = false;
		}

		/// <summary>
		/// Creates a new CudaManagedMemory from definition in cu-file.
		/// </summary>
		/// <param name="kernel">The kernel which module defines the variable.</param>
		/// <param name="name">The variable name as defined in the cu-file.</param>
		public CudaManagedMemory_dim3(CudaKernel kernel, string name)
			: this(kernel.CUModule, name)
		{
			
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaManagedMemory_dim3()
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
					res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree_v2", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// UIntPtr to managed memory.
		/// </summary>
		public UIntPtr HostPointer
		{
			get { return _devPtr.Pointer; }
		}

		/// <summary>
		/// CUdeviceptr to managed memory.
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
		/// Size in elements
		/// </summary>
		public SizeT Size
		{
			get { return _size; }
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
		/// Converts a managed variable to a host value. In case of multiple managed values (array), only the first value is converted.
		/// </summary>
		/// <param name="d">managed variable</param>
		/// <returns>newly allocated host variable with value from managed memory</returns>
		public static implicit operator dim3(CudaManagedMemory_dim3 d)
		{
			return d[0];
		}
		#endregion

		#region GetAttributeMethods
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		public CUcontext AttributeContext
		{
			get 
			{
				CUcontext ret = new CUcontext();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		public CUMemoryType AttributeMemoryType
		{
			get
			{
				CUMemoryType ret = new CUMemoryType();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device <para/>
		/// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
		/// </summary>
		public CUdeviceptr AttributeDevicePointer
		{
			get
			{
				CUdeviceptr ret = new CUdeviceptr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		public IntPtr AttributeHostPointer
		{
			get
			{
				IntPtr ret = new IntPtr();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		public CudaPointerAttributeP2PTokens AttributeP2PTokens
		{
			get
			{
				CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		public bool AttributeSyncMemops
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
			set 
			{
				int val = value ? 1 : 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		public ulong AttributeBufferID
		{
			get
			{
				ulong ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret;
			}
		}

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		public bool AttributeIsManaged
		{
			get
			{
				int ret = 0;
				CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, _devPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
				if (res != CUResult.Success) throw new CudaException(res);
				return ret != 0;
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Attach memory to a stream asynchronously
		/// <para/>
		/// Enqueues an operation in <c>hStream</c> to specify stream association of
		/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
		/// stream-ordered operation, meaning that it is dependent on, and will
		/// only take effect when, previous work in stream has completed. Any
		/// previous association is automatically replaced.
		/// <para/>
		/// <c>dptr</c> must point to an address within managed memory space declared
		/// using the __managed__ keyword or allocated with cuMemAllocManaged.
		/// <para/>
		/// <c>length</c> must be zero, to indicate that the entire allocation's
		/// stream association is being changed. Currently, it's not possible
		/// to change stream association for a portion of an allocation.
		/// <para/>
		/// The stream association is specified using <c>flags</c> which must be
		/// one of <see cref="CUmemAttach_flags"/>.
		/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
		/// by any stream on any device.
		/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
		/// that it won't access the memory on the device from any stream.
		/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
		/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
		/// to attach singly to the NULL stream, because the NULL stream is a virtual global
		/// stream and not a specific stream. An error will be returned in this case.
		/// <para/>
		/// When memory is associated with a single stream, the Unified Memory system will
		/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
		/// have completed, regardless of whether other streams are active. In effect,
		/// this constrains exclusive ownership of the managed memory region by
		/// an active GPU to per-stream activity instead of whole-GPU activity.
		/// <para/>
		/// Accessing memory on the device from streams that are not associated with
		/// it will produce undefined results. No error checking is performed by the
		/// Unified Memory system to ensure that kernels launched into other streams
		/// do not access this region. 
		/// <para/>
		/// It is a program's responsibility to order calls to <see cref="DriverAPINativeMethods.Streams.cuStreamAttachMemAsync"/>
		/// via events, synchronization or other means to ensure legal access to memory
		/// at all times. Data visibility and coherency will be changed appropriately
		/// for all kernels which follow a stream-association change.
		/// <para/>
		/// If <c>hStream</c> is destroyed while data is associated with it, the association is
		/// removed and the association reverts to the default visibility of the allocation
		/// as specified at cuMemAllocManaged. For __managed__ variables, the default
		/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
		/// asynchronous operation, and as a result, the change to default association won't
		/// happen until all work in the stream has completed.
		/// <para/>
		/// </summary>
		/// <param name="hStream">Stream in which to enqueue the attach operation</param>
		/// <param name="length">Length of memory (must be zero)</param>
		/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
		/// <returns></returns>
		public void StreamAttachMemAsync(CUstream hStream, SizeT length, CUmemAttach_flags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			res = DriverAPINativeMethods.Streams.cuStreamAttachMemAsync(hStream, _devPtr, length, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamAttachMemAsync", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}


		#endregion

		#region IEnumerable
		IEnumerator<dim3> IEnumerable<dim3>.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator<dim3> enumerator = new CudaManagedMemoryEnumerator_dim3(this);
			return enumerator;
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			IEnumerator enumerator = new CudaManagedMemoryEnumerator_dim3(this);
			return enumerator;
		}

		#endregion
	}
	
	/// <summary>
	/// Enumerator class for CudaManagedMemory_dim3
	/// </summary>
	public class CudaManagedMemoryEnumerator_dim3 : IEnumerator<dim3>
	{
		private CudaManagedMemory_dim3 _memory = null;
		private SizeT _currentIndex = -1;

		/// <summary>
		/// 
		/// </summary>
		/// <param name="memory"></param>
		public CudaManagedMemoryEnumerator_dim3(CudaManagedMemory_dim3 memory)
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
		public dim3 Current
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
