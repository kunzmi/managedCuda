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
using System.Drawing;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
using ManagedCuda.BasicTypes;


namespace ManagedCuda.NPP
{
	/// <summary>
	/// Abstract base class for derived NPP typed images.
	/// </summary>
	public abstract class NPPImageBase : IDisposable
	{
		/// <summary>
		/// Base pointer to image data.
		/// </summary>
		protected CUdeviceptr _devPtr;
		/// <summary>
		/// Base pointer moved to actual ROI.
		/// </summary>
		protected CUdeviceptr _devPtrRoi;
		/// <summary>
		/// Size of the entire image.
		/// </summary>
		protected NppiSize _sizeOriginal = new NppiSize();
		/// <summary>
		/// Size of the actual ROI.
		/// </summary>
		protected NppiSize _sizeRoi = new NppiSize();
		/// <summary>
		/// First pixel in the ROI.
		/// </summary>
		protected NppiPoint _pointRoi = new NppiPoint();
		/// <summary>
		/// Width of one image line + alignment bytes.
		/// </summary>
		protected int _pitch = 0;
		/// <summary>
		/// Number of color channels in image.
		/// </summary>
		protected int _channels = 0;
		/// <summary>
		/// Type size in bytes of one pixel in one channel.
		/// </summary>
		protected int _typeSize = 0;
		/// <summary>
		/// Last CUResult.
		/// </summary>
		protected CUResult res;
		/// <summary>
		/// Last NPPStatus result.
		/// </summary>
		protected NppStatus status;
		/// <summary>
		/// 
		/// </summary>
		protected bool disposed;
		/// <summary>
		/// 
		/// </summary>
		protected bool _isOwner;
		
		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public virtual void Dispose()
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
					NPPNativeMethods.NPPi.MemAlloc.nppiFree(_devPtr);
					Debug.WriteLine(String.Format("{0:G}, {1}", DateTime.Now, "nppiFree"));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("NPP not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Copy Sync
		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="stride">Size of one image line in bytes with padding</param>
		///// <param name="height">Height in elements</param>
		public void CopyToDevice(IntPtr hostSrc, SizeT stride)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcHost = hostSrc;
			copyParams.srcPitch = stride;
			copyParams.srcMemoryType = CUMemoryType.Host;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDevice<T>(CudaPitchedDeviceVariable<T> deviceSrc) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDevice(NPPImageBase deviceSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDevice<T>(CudaDeviceVariable<T> deviceSrc) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

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
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="pitch">Pitch of deviceSrc</param>
		public void CopyToDevice(CUdeviceptr deviceSrc, SizeT pitch)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc;
			copyParams.srcPitch = pitch;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">IntPtr to destination in host memory</param>
		/// <param name="stride">Size of one image line in bytes with padding</param>
		///// <param name="width">Width in bytes</param>
		///// <param name="height">Height in elements</param>
		public void CopyToHost(IntPtr hostDest, SizeT stride)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.dstHost = hostDest;
			copyParams.dstPitch = stride;
			copyParams.dstMemoryType = CUMemoryType.Host;
			copyParams.srcDevice = _devPtr;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = _pitch;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="stride">Size of one image line in bytes with padding</param>
		///// <param name="height">Height in elements</param>
		public void CopyToDevice(Array hostSrc, SizeT stride) 
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcPitch = stride;
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = _pitch;
				copyParams.Height = _sizeOriginal.height;
				copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

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
		/// <param name="stride">Size of one image line in bytes with padding</param>
		public void CopyToHost(Array hostDest, SizeT stride)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstPitch = stride;
				copyParams.dstMemoryType = CUMemoryType.Host;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _pitch;
				copyParams.Height = _sizeOriginal.height;
				copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

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
		public void CopyToDevice(IntPtr hostSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcHost = hostSrc;
			copyParams.srcMemoryType = CUMemoryType.Host;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

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
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		public void CopyToDevice(Array hostSrc) 
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
				copyParams.Height = _sizeOriginal.height;
				copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

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
		/// <param name="aHostDest">Destination</param>
		public void CopyToHost<T>(T[] aHostDest) where T : struct
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
				copyParams.Height = _sizeOriginal.height;
				copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

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
		/// Copy data from a System.Drawing.Bitmap. There is no check if the bitmap pixel type corresponds to the current NPPImage!
		/// </summary>
		/// <param name="bitmap"></param>
		public void CopyToDevice(Bitmap bitmap)
		{
			Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
			System.Drawing.Imaging.PixelFormat format = bitmap.PixelFormat;
			System.Drawing.Imaging.BitmapData data;

			data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, format);

			try
			{
				CopyToDevice(data.Scan0, Math.Abs(data.Stride));
			}
			finally
			{
				bitmap.UnlockBits(data);
			}
		}

		/// <summary>
		/// Copy data to a System.Drawing.Bitmap. There is no check if the bitmap pixel type corresponds to the current NPPImage!
		/// </summary>
		/// <param name="bitmap"></param>
		public void CopyToHost(Bitmap bitmap)
		{
			Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
			System.Drawing.Imaging.PixelFormat format = bitmap.PixelFormat;
			System.Drawing.Imaging.BitmapData data;

			data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly, format);
			try
			{
				CopyToHost(data.Scan0, Math.Abs(data.Stride));
			}
			finally
			{
				bitmap.UnlockBits(data);
			}
		}
		#endregion

		#region Copy Sync ROI
		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="stride">Size of one image line in bytes with padding</param>
		/// <param name="roi">ROI of source image</param>
		///// <param name="height">Height in elements</param>
		public void CopyToDeviceRoi(IntPtr hostSrc, SizeT stride, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcHost = hostSrc;
			copyParams.srcPitch = stride;
			copyParams.srcXInBytes = roi.x * _typeSize * _channels;
			copyParams.srcY = roi.y;
			copyParams.srcMemoryType = CUMemoryType.Host;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.dstY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="roi">ROI of source image</param>
		public void CopyToDeviceRoi<T>(CudaPitchedDeviceVariable<T> deviceSrc, NppiRect roi) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.srcXInBytes = roi.x * _typeSize * _channels;
			copyParams.srcY = roi.y;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.dstY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDeviceRoi(NPPImageBase deviceSrc)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.srcXInBytes = deviceSrc.PointRoi.x * _typeSize * _channels;
			copyParams.srcY = deviceSrc.PointRoi.y;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.dstY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="roi">ROI of source image</param>
		public void CopyToDeviceRoi<T>(CudaDeviceVariable<T> deviceSrc, NppiRect roi) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcXInBytes = roi.x * _typeSize * _channels;
			copyParams.srcY = roi.y;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.dstY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="roi">ROI of source image</param>
		public void CopyToDeviceRoi(CUdeviceptr deviceSrc, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc;
			copyParams.srcXInBytes = roi.x * _typeSize * _channels;
			copyParams.srcY = roi.y;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.dstY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="pitch">Pitch of deviceSrc</param>
		/// <param name="roi">ROI of source image</param>
		public void CopyToDeviceRoi(CUdeviceptr deviceSrc, SizeT pitch, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc;
			copyParams.srcPitch = pitch;
			copyParams.srcXInBytes = roi.x * _typeSize * _channels;
			copyParams.srcY = roi.y;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.dstY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">IntPtr to destination in host memory</param>
		/// <param name="stride">Size of one image line in bytes with padding</param>
		/// <param name="roi">ROI of destination image</param>
		///// <param name="width">Width in bytes</param>
		///// <param name="height">Height in elements</param>
		public void CopyToHostRoi(IntPtr hostDest, SizeT stride, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.dstHost = hostDest;
			copyParams.dstPitch = stride;
			copyParams.dstXInBytes = roi.x * _typeSize * _channels;
			copyParams.dstY = roi.y;
			copyParams.dstMemoryType = CUMemoryType.Host;
			copyParams.srcDevice = _devPtr;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = _pitch;
			copyParams.srcXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.srcY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="stride">Size of one image line in bytes with padding</param>
		/// <param name="roi">ROI of source image</param>
		///// <param name="height">Height in elements</param>
		public void CopyToDeviceRoi(Array hostSrc, SizeT stride, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcPitch = stride;
				copyParams.srcXInBytes = roi.x * _typeSize * _channels;
				copyParams.srcY = roi.y;
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = _pitch;
				copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
				copyParams.dstY = _pointRoi.y;
				copyParams.Height = _sizeRoi.height;
				copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

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
		/// <param name="stride">Size of one image line in bytes with padding</param>
		/// <param name="roi">ROI of destination image</param>
		public void CopyToHostRoi(Array hostDest, SizeT stride, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstPitch = stride;
				copyParams.dstXInBytes = roi.x * _typeSize * _channels;
				copyParams.dstY = roi.y;
				copyParams.dstMemoryType = CUMemoryType.Host;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _pitch;
				copyParams.srcXInBytes = _pointRoi.x * _typeSize * _channels;
				copyParams.srcY = _pointRoi.y;
				copyParams.Height = _sizeRoi.height;
				copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

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
		/// <param name="roi">ROI of source image</param>
		public void CopyToDeviceRoi(IntPtr hostSrc, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcHost = hostSrc;
			copyParams.srcXInBytes = roi.x * _typeSize * _channels;
			copyParams.srcY = roi.y;
			copyParams.srcMemoryType = CUMemoryType.Host;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.dstPitch = _pitch;
			copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.dstY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from device to host memory
		/// </summary>
		/// <param name="hostDest">IntPtr to destination in host memory</param>
		/// <param name="roi">ROI of destination image</param>
		public void CopyToHostRoi(IntPtr hostDest, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.dstHost = hostDest;
			copyParams.dstXInBytes = roi.x * _typeSize * _channels;
			copyParams.dstY = roi.y;
			copyParams.dstMemoryType = CUMemoryType.Host;
			copyParams.srcDevice = _devPtr;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = _pitch;
			copyParams.srcXInBytes = _pointRoi.x * _typeSize * _channels;
			copyParams.srcY = _pointRoi.y;
			copyParams.Height = _sizeRoi.height;
			copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy from Host to device memory
		/// </summary>
		/// <param name="hostSrc">Source</param>
		/// <param name="roi">ROI of source image</param>
		public void CopyToDeviceRoi(Array hostSrc, NppiRect roi)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcHost = handle.AddrOfPinnedObject();
				copyParams.srcXInBytes = roi.x * _typeSize * _channels;
				copyParams.srcY = roi.y;
				copyParams.srcMemoryType = CUMemoryType.Host;
				copyParams.dstDevice = _devPtr;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.dstPitch = _pitch;
				copyParams.dstXInBytes = _pointRoi.x * _typeSize * _channels;
				copyParams.dstY = _pointRoi.y;
				copyParams.Height = _sizeRoi.height;
				copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

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
		/// <param name="aHostDest">Destination</param>
		/// <param name="roi">ROI of destination image</param>
		public void CopyToHostRoi<T>(T[] aHostDest, NppiRect roi) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			GCHandle handle = GCHandle.Alloc(aHostDest, GCHandleType.Pinned);
			try
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.dstHost = handle.AddrOfPinnedObject();
				copyParams.dstXInBytes = roi.x * _typeSize * _channels;
				copyParams.dstY = roi.y;
				copyParams.dstMemoryType = CUMemoryType.Host;
				//copyParams.dstPitch = _width * _typeSize;
				copyParams.srcDevice = _devPtr;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _pitch;
				copyParams.srcXInBytes = _pointRoi.x * _typeSize * _channels;
				copyParams.srcY = _pointRoi.y;
				copyParams.Height = _sizeRoi.height;
				copyParams.WidthInBytes = _sizeRoi.width * _typeSize * _channels;

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
		/// Copy data from a System.Drawing.Bitmap. There is no check if the bitmap pixel type corresponds to the current NPPImage!
		/// </summary>
		/// <param name="bitmap">Source Bitmap</param>
		/// <param name="roi">ROI of source image</param>
		public void CopyToDeviceRoi(Bitmap bitmap, NppiRect roi)
		{
			Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
			System.Drawing.Imaging.PixelFormat format = bitmap.PixelFormat;
			System.Drawing.Imaging.BitmapData data;

			data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, format);

			try
			{
				CopyToDeviceRoi(data.Scan0, Math.Abs(data.Stride), roi);
			}
			finally
			{
				bitmap.UnlockBits(data);
			}
		}

		/// <summary>
		/// Copy data to a System.Drawing.Bitmap. There is no check if the bitmap pixel type corresponds to the current NPPImage!
		/// </summary>
		/// <param name="bitmap">Destination Bitmap</param>
		/// <param name="roi">ROI of destination image</param>
		public void CopyToHostRoi(Bitmap bitmap, NppiRect roi)
		{
			Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
			System.Drawing.Imaging.PixelFormat format = bitmap.PixelFormat;
			System.Drawing.Imaging.BitmapData data;

			data = bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly, format);
			try
			{
				CopyToHostRoi(data.Scan0, Math.Abs(data.Stride), roi);
			}
			finally
			{
				bitmap.UnlockBits(data);
			}
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
			SizeT aSizeInBytes = _pitch * _sizeOriginal.height;
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
		public void AsyncCopyToDevice<T>(CudaDeviceVariable<T> source, CUstream stream) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = _pitch * _sizeOriginal.height;
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
		public void AsyncCopyToDevice<T>(CudaPitchedDeviceVariable<T> deviceSrc, CUstream stream) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Async Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		/// <param name="stream"></param>
		public void AsyncCopyToDevice(NPPImageBase deviceSrc, CUstream stream)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _devPtr;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = _sizeOriginal.height;
			copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

			res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpy2DAsync_v2(ref copyParams, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2DAsync", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region Properties
		/// <summary>
		/// Size of the entire image.
		/// </summary>
		public NppiSize Size
		{
			get { return _sizeOriginal; }
		}
		/// <summary>
		/// Size of the actual ROI.
		/// </summary>
		public NppiSize SizeRoi
		{
			get { return _sizeRoi; }
		}
		/// <summary>
		/// First pixel in the ROI.
		/// </summary>
		public NppiPoint PointRoi
		{
			get { return _pointRoi; }
		}

		/// <summary>
		/// Device pointer to image data.
		/// </summary>
		public CUdeviceptr DevicePointer
		{
			get { return _devPtr; }
		}

		/// <summary>
		/// Device pointer to first pixel in ROI.
		/// </summary>
		public CUdeviceptr DevicePointerRoi
		{
			get { return _devPtrRoi; }
		}

		/// <summary>
		/// Width in pixels
		/// </summary>
		public int Width
		{
			get { return _sizeOriginal.width; }
		}

		/// <summary>
		/// Width in bytes
		/// </summary>
		public int WidthInBytes
		{
			get { return _sizeOriginal.width * _typeSize * _channels; }
		}

		/// <summary>
		/// Height in pixels
		/// </summary>
		public int Height
		{
			get { return _sizeOriginal.height; }
		}

		/// <summary>
		/// Width in pixels
		/// </summary>
		public int WidthRoi
		{
			get { return _sizeRoi.width; }
		}

		/// <summary>
		/// Width in bytes
		/// </summary>
		public int WidthRoiInBytes
		{
			get { return _sizeRoi.width * _typeSize * _channels; }
		}

		/// <summary>
		/// Height in pixels
		/// </summary>
		public int HeightRoi
		{
			get { return _sizeRoi.height; }
		}

		/// <summary>
		/// Pitch in bytes
		/// </summary>
		public int Pitch
		{
			get { return _pitch; }
		}

		/// <summary>
		/// Total size in bytes (Pitch * Height)
		/// </summary>
		public int TotalSizeInBytes
		{
			get { return _pitch * _sizeOriginal.height; }
		}

		/// <summary>
		/// Color channels
		/// </summary>
		public int Channels
		{
			get { return _channels; }
		}
		#endregion

		#region ROI
		/// <summary>
		/// Defines the ROI on which all following operations take place
		/// </summary>
		/// <param name="roi"></param>
		public void SetRoi(NppiRect roi)
		{
			_devPtrRoi = _devPtr + _typeSize * _channels * roi.x + _pitch * roi.y;
			_pointRoi = roi.Location;
			_sizeRoi = roi.Size;
		}

		/// <summary>
		/// Defines the ROI on which all following operations take place
		/// </summary>
		/// <param name="x"></param>
		/// <param name="y"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		public void SetRoi(int x, int y, int width, int height)
		{
			SetRoi(new NppiRect(x, y, width, height));
		}

		/// <summary>
		/// Resets the ROI to the full image
		/// </summary>
		public void ResetRoi()
		{
			_devPtrRoi = _devPtr;
			_pointRoi = new NppiPoint(0, 0);
			_sizeRoi = _sizeOriginal;
		}
		#endregion

		/// <summary>
		/// Returns NppiRect which represents the offset and size of the destination rectangle that would be generated by
		/// resizing the source NppiRect by the requested scale factors and shifts.
		/// </summary>
		/// <param name="nXFactor">Factor by which x dimension is changed. </param>
		/// <param name="nYFactor">Factor by which y dimension is changed. </param>
		/// <param name="nXShift">Source pixel shift in x-direction.</param>
		/// <param name="nYShift">Source pixel shift in y-direction.</param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public NppiRect GetResizeRect(double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect dstRect = new NppiRect();
			status = NPPNativeMethods.NPPi.GetResizeRect.nppiGetResizeRect(srcRect, ref dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetResizeRect", status));
			NPPException.CheckNppStatus(status, this);
			return dstRect;
		}

		#region helper methods
		///// <summary>
		///// Checks if two images have the same dimensions
		///// </summary>
		///// <param name="image">image to compare</param>
		///// <returns>true if dimensions fit</returns>
		//protected bool CheckSize(NPPImageBase image)
		//{
		//    return this.Width == image.Width && this.Height == image.Height;
		//}
		#endregion
	}
}
