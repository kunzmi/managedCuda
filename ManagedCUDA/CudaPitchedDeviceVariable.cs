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
	/// A variable located in CUDA device memory. The data is aligned following <see cref="CudaPitchedDeviceVariable{T}.Pitch"/>
	/// </summary>
	/// <typeparam name="T">variable base type</typeparam>
	public class CudaPitchedDeviceVariable<T> : IDisposable where T:struct
	{
		CUdeviceptr _devPtr;
		SizeT _height = 0, _width = 0, _pitch = 0;
		SizeT _typeSize = 0;
		CUResult res;
		bool disposed;
		bool _isOwner;

		#region Constructors
		/// <summary>
		/// Creates a new CudaPitchedDeviceVariable and allocates the memory on the device
		/// </summary>
		/// <param name="width">In elements</param>
		/// <param name="height">In elements</param>
		public CudaPitchedDeviceVariable(SizeT width, SizeT height)
		{
			_devPtr = new CUdeviceptr();
			_height = height;
			_width = width;
			_typeSize = (uint)Marshal.SizeOf(typeof(T));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2(ref _devPtr, ref _pitch, _typeSize * width, height, (uint)_typeSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}", DateTime.Now, "cuMemAllocPitch", res, _pitch));
			if (res != CUResult.Success) throw new CudaException(res);
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaPitchedDeviceVariable and allocates the memory on the device
		/// </summary>
		/// <param name="width">In elements</param>
		/// <param name="height">In elements</param>
		/// <param name="pack">Group <c>pack</c> elements as one type. E.g. 4 floats in host code to one float4 in device code</param>
		public CudaPitchedDeviceVariable(SizeT width, SizeT height, SizeT pack)
		{
			_devPtr = new CUdeviceptr();
			_height = height;
			_width = width;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(T));

			res = DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2(ref _devPtr, ref _pitch, _typeSize * width, height, (uint)(_typeSize * pack));
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}", DateTime.Now, "cuMemAllocPitch", res, _pitch));
			if (res != CUResult.Success) throw new CudaException(res);
			_isOwner = true;
		}

		/// <summary>
		/// Creates a new CudaPitchedDeviceVariable from an existing CUdeviceptr
		/// The CUdeviceptr won't be freed when disposing.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="width">In elements</param>
		/// <param name="height">In elements</param>
		/// <param name="pitch">In bytes</param>
		public CudaPitchedDeviceVariable(CUdeviceptr devPtr, SizeT width, SizeT height, SizeT pitch)
			: this(devPtr, width, height, pitch, false)
		{

		}

		/// <summary>
		/// Creates a new CudaPitchedDeviceVariable from an existing CUdeviceptr
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="width">In elements</param>
		/// <param name="height">In elements</param>
		/// <param name="pitch">In bytes</param>
		/// <param name="isOwner">The CUdeviceptr will be freed while disposing if the CudaPitchedDeviceVariable is the owner</param>
		public CudaPitchedDeviceVariable(CUdeviceptr devPtr, SizeT width, SizeT height, SizeT pitch, bool isOwner)
		{
			_devPtr = devPtr;
			_height = height;
			_width = width;
			_pitch = pitch;
			_typeSize = (SizeT)Marshal.SizeOf(typeof(T));
			_isOwner = isOwner;
		}
		
		/// <summary>
		/// For dispose
		/// </summary>
		~CudaPitchedDeviceVariable()
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

		#region Copy Sync
		#region Device to device

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
			copyParams.dstPitch = _pitch;
			copyParams.Height = _height;
			copyParams.WidthInBytes = _width * _typeSize;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToDevice(CudaPitchedDeviceVariable<T> deviceSrc, SizeT SrcXInBytes, SizeT SrcY, SizeT DestXInBytes, SizeT DestY, SizeT widthInBytes, SizeT height, SizeT SrcPitch, SizeT DestPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = SrcPitch;
			copyParams.srcXInBytes = SrcXInBytes;
			copyParams.srcY = SrcY;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = DestPitch;
			copyParams.dstXInBytes = DestXInBytes;
			copyParams.dstY = DestY;
			copyParams.Height = height;
			copyParams.WidthInBytes = widthInBytes;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDevice(CudaDeviceVariable<T> deviceSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _height;
			copyParams.WidthInBytes = _width * _typeSize;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToDevice(CudaDeviceVariable<T> deviceSrc, SizeT SrcXInBytes, SizeT SrcY, SizeT DestXInBytes, SizeT DestY, SizeT widthInBytes, SizeT height, SizeT SrcPitch, SizeT DestPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = SrcPitch;
			copyParams.srcXInBytes = SrcXInBytes;
			copyParams.srcY = SrcY;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = DestPitch;
			copyParams.dstXInBytes = DestXInBytes;
			copyParams.dstY = DestY;
			copyParams.Height = height;
			copyParams.WidthInBytes = widthInBytes;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDevice(CUdeviceptr deviceSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _height;
			copyParams.WidthInBytes = _width * _typeSize;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="pitchSrc">Source pitch</param>
		public void CopyToDevice(CUdeviceptr deviceSrc, SizeT pitchSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc;
			copyParams.srcPitch = pitchSrc;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _height;
			copyParams.WidthInBytes = _width * _typeSize;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToDevice(CUdeviceptr deviceSrc, SizeT SrcXInBytes, SizeT SrcY, SizeT DestXInBytes, SizeT DestY, SizeT widthInBytes, SizeT height, SizeT SrcPitch, SizeT DestPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = SrcPitch;
			copyParams.srcXInBytes = SrcXInBytes;
			copyParams.srcY = SrcY;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = DestPitch;
			copyParams.dstXInBytes = DestXInBytes;
			copyParams.dstY = DestY;
			copyParams.Height = height;
			copyParams.WidthInBytes = widthInBytes;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region Host to device
		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		public void CopyToDevice(IntPtr hostSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcHost = hostSrc;
			copyParams.srcMemoryType = CUMemoryType.Host;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _height;
			copyParams.WidthInBytes = _width * _typeSize;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="width">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		public void CopyToDevice(IntPtr hostSrc, SizeT width, SizeT height)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcHost = hostSrc;
			copyParams.srcMemoryType = CUMemoryType.Host;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = height;
			copyParams.WidthInBytes = width;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToDevice(IntPtr hostSrc, SizeT SrcXInBytes, SizeT SrcY, SizeT DestXInBytes, SizeT DestY, SizeT widthInBytes, SizeT height, SizeT SrcPitch, SizeT DestPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcHost = hostSrc;
			copyParams.srcMemoryType = CUMemoryType.Host;
			copyParams.srcPitch = SrcPitch;
			copyParams.srcXInBytes = SrcXInBytes;
			copyParams.srcY = SrcY;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = DestPitch;
			copyParams.dstXInBytes = DestXInBytes;
			copyParams.dstY = DestY;
			copyParams.Height = height;
			copyParams.WidthInBytes = widthInBytes;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		public void CopyToDevice(T[] hostSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = _pitch;
				copyParams.Height = _height;
				copyParams.WidthInBytes = _width * _typeSize;

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
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="width">Width in elements</param>
		/// <param name="height">Height in elements</param>
		public void CopyToDevice(T[] hostSrc, SizeT width, SizeT height)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = _pitch;
				copyParams.Height = height;
				copyParams.WidthInBytes = width * (SizeT)Marshal.SizeOf(typeof(T));

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
		/// Copy from host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToDevice(T[] hostSrc, SizeT SrcXInBytes, SizeT SrcY, SizeT DestXInBytes, SizeT DestY, SizeT widthInBytes, SizeT height, SizeT SrcPitch, SizeT DestPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.srcPitch = SrcPitch;
				copyParams.srcXInBytes = SrcXInBytes;
				copyParams.srcY = SrcY;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = DestPitch;
				copyParams.dstXInBytes = DestXInBytes;
				copyParams.dstY = DestY;
				copyParams.Height = height;
				copyParams.WidthInBytes = widthInBytes;

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
		/// Copy from Host to device memory. Assumes that aHostDest has no additional line padding.
		/// </summary>
		/// <param name="hostSrc">Source</param>
		public void CopyToDevice(T[,] hostSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = _pitch;
				copyParams.Height = _height;
				copyParams.WidthInBytes = _width * _typeSize;

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
		/// Copy from host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToDevice(T[,] hostSrc, SizeT SrcXInBytes, SizeT SrcY, SizeT DestXInBytes, SizeT DestY, SizeT widthInBytes, SizeT height, SizeT SrcPitch, SizeT DestPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.srcPitch = SrcPitch;
				copyParams.srcXInBytes = SrcXInBytes;
				copyParams.srcY = SrcY;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = DestPitch;
				copyParams.dstXInBytes = DestXInBytes;
				copyParams.dstY = DestY;
				copyParams.Height = height;
				copyParams.WidthInBytes = widthInBytes;

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
		#endregion

		#region Device to host
		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">IntPtr to destination in host memory</param>
		public void CopyToHost(IntPtr hostDest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.dstHost = hostDest;
			copyParams.dstMemoryType = CUMemoryType.Host;
			copyParams.srcDevice = _devPtr;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = _pitch;
			copyParams.Height = _height;
			copyParams.WidthInBytes = _width * _typeSize;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">IntPtr to destination in host memory</param>
		/// <param name="width">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		public void CopyToHost(IntPtr hostDest, SizeT width, SizeT height)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.dstHost = hostDest;
			copyParams.dstMemoryType = CUMemoryType.Host;
			copyParams.srcDevice = _devPtr;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = _pitch;
			copyParams.Height = height;
			copyParams.WidthInBytes = width;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">Destination</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToHost(IntPtr hostDest, SizeT DestXInBytes, SizeT DestY, SizeT SrcXInBytes, SizeT SrcY, SizeT widthInBytes, SizeT height, SizeT DestPitch, SizeT SrcPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.dstHost = hostDest;
			copyParams.dstMemoryType = CUMemoryType.Host;
			copyParams.dstPitch = DestPitch;
			copyParams.dstXInBytes = DestXInBytes;
			copyParams.dstY = DestY;
			copyParams.srcDevice = _devPtr;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = SrcPitch;
			copyParams.srcXInBytes = SrcXInBytes;
			copyParams.srcY = SrcY;
			copyParams.Height = height;
			copyParams.WidthInBytes = widthInBytes;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="aHostDest">Destination</param>
		public void CopyToHost(T[] aHostDest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(aHostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstMemoryType = CUMemoryType.Host;
				//copyParams.dstPitch = _width * _typeSize;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _pitch;
				copyParams.Height = _height;
				copyParams.WidthInBytes = _width * _typeSize;

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
		/// Copy data from this device to host memory
		/// </summary>
		/// <param name="hostDest">Destination</param>
		/// <param name="width">Width in elements</param>
		/// <param name="height">Height in elements</param>
		public void CopyToHost(T[] hostDest, SizeT width, SizeT height)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstMemoryType = CUMemoryType.Host;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _pitch;
				copyParams.Height = height;
				copyParams.WidthInBytes = width * (SizeT)Marshal.SizeOf(typeof(T));

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
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">Destination</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToHost(T[] hostDest, SizeT DestXInBytes, SizeT DestY, SizeT SrcXInBytes, SizeT SrcY, SizeT widthInBytes, SizeT height, SizeT DestPitch, SizeT SrcPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstMemoryType = CUMemoryType.Host;
				copyParams.dstPitch = DestPitch;
				copyParams.dstXInBytes = DestXInBytes;
				copyParams.dstY = DestY;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = SrcPitch;
				copyParams.srcXInBytes = SrcXInBytes;
				copyParams.srcY = SrcY;
				copyParams.Height = height;
				copyParams.WidthInBytes = widthInBytes;

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
		/// Copy data from device to host memory. Assumes that aHostDest has no additional line padding.
		/// </summary>
		/// <param name="aHostDest">Destination</param>
		public void CopyToHost(T[,] aHostDest)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(aHostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstMemoryType = CUMemoryType.Host;
				//copyParams.dstPitch = _width * _typeSize;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _pitch;
				copyParams.Height = _height;
				copyParams.WidthInBytes = _width * _typeSize;

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
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">Destination</param>
		/// <param name="SrcXInBytes">Source X in bytes</param>
		/// <param name="SrcY">Source Y</param>
		/// <param name="DestXInBytes">Destination X in bytes</param>
		/// <param name="DestY">Destination Y</param>
		/// <param name="widthInBytes">Width in bytes</param>
		/// <param name="height">Height in elements</param>
		/// <param name="SrcPitch">Source pitch</param>
		/// <param name="DestPitch">Destination pitch</param>
		public void CopyToHost(T[,] hostDest, SizeT DestXInBytes, SizeT DestY, SizeT SrcXInBytes, SizeT SrcY, SizeT widthInBytes, SizeT height, SizeT DestPitch, SizeT SrcPitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstMemoryType = CUMemoryType.Host;
				copyParams.dstPitch = DestPitch;
				copyParams.dstXInBytes = DestXInBytes;
				copyParams.dstY = DestY;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = SrcPitch;
				copyParams.srcXInBytes = SrcXInBytes;
				copyParams.srcY = SrcY;
				copyParams.Height = height;
				copyParams.WidthInBytes = widthInBytes;

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
		#endregion

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
			SizeT aSizeInBytes = _pitch * _height;
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
			SizeT aSizeInBytes = _pitch * _height;
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
			copyParams.dstPitch = _pitch;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = deviceSrc.Height;
			copyParams.WidthInBytes = deviceSrc.Width * _typeSize;

			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}


		/// <summary>
		/// Async Copy data from device to device memory (1D Copy, copies destination pitch * height bytes data)
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CUdeviceptr source, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _pitch * _height;
			CUResult res;
			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(_devPtr, source, aSizeInBytes, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoDAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy data from device to device memory (1D Copy, copies destination pitch * height bytes data)
		/// </summary>
		/// <param name="source">Source</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(CudaDeviceVariable<T> source, CudaStream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _pitch * _height;
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
			copyParams.Height = _height;
			copyParams.WidthInBytes = _width * _typeSize;

			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream.Stream);
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
			res = DriverAPINativeMethods.Memset.cuMemsetD2D8_v2(_devPtr, _pitch, aValue, _width * _typeSize, _height);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D8", res));
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
			res = DriverAPINativeMethods.Memset.cuMemsetD2D16_v2(_devPtr, _pitch, aValue, _width * _typeSize / sizeof(ushort), _height);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D16", res));
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
			res = DriverAPINativeMethods.Memset.cuMemsetD2D32_v2(_devPtr, _pitch, aValue, _width * _typeSize / sizeof(uint), _height);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D32", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Memset
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="stream"></param>
		public void MemsetAsync(byte aValue, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUResult res;
			res = DriverAPINativeMethods.MemsetAsync.cuMemsetD2D8Async(_devPtr, _pitch, aValue, _width * _typeSize, _height, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D8Async", res));
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
			res = DriverAPINativeMethods.MemsetAsync.cuMemsetD2D16Async(_devPtr, _pitch, aValue, _width * _typeSize / sizeof(ushort), _height, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D16Async", res));
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
			res = DriverAPINativeMethods.MemsetAsync.cuMemsetD2D32Async(_devPtr, _pitch, aValue, _width * _typeSize / sizeof(uint), _height, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemsetD2D32Async", res));
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
			SizeT aSizeInBytes = _pitch * _height;
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
			SizeT aSizeInBytes = _pitch * _height;
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
			SizeT aSizeInBytes = _pitch * _height;
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
			SizeT aSizeInBytes = _pitch * _height;
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
		/// <param name="x">X-index in elements</param>
		/// <param name="y">Y-index in elements</param>
		/// <returns></returns>
		public T this[SizeT x, SizeT y]
		{
			get
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());

				CUdeviceptr position = _devPtr + _pitch * y + x * _typeSize;
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
				CUdeviceptr position = _devPtr + _pitch * y + x * _typeSize;

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
		/// Width in elements
		/// </summary>
		public SizeT Width
		{
			get { return _width; }
		}

		/// <summary>
		/// Width in bytes
		/// </summary>
		public SizeT WidthInBytes
		{
			get { return _width * _typeSize; }
		}

		/// <summary>
		/// Height in elements
		/// </summary>
		public SizeT Height
		{
			get { return _height; }
		}

		/// <summary>
		/// Pitch in bytes
		/// </summary>
		public SizeT Pitch
		{
			get { return _pitch; }
		}

		/// <summary>
		/// Total size in bytes (Pitch * Height)
		/// </summary>
		public SizeT TotalSizeInBytes
		{
			get { return _pitch * _height; }
		}

		/// <summary>
		/// Type size in bytes
		/// </summary>
		public SizeT TypeSize
		{
			get { return _typeSize; }
		}
		#endregion

		/// <summary>
		/// Converts a device variable to a host array
		/// </summary>
		/// <param name="d">device variable</param>
		/// <returns>newly allocated host array with values from device memory</returns>
		public static implicit operator T[](CudaPitchedDeviceVariable<T> d)
		{
			T[] ret = new T[(long)d.Width * (long)d.Height];
			d.CopyToHost(ret);
			return ret;
		}
	}
}
