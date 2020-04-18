//	Copyright (c) 2020, Michael Kunz. All rights reserved.
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
using System.Text;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NvJpeg
{

	/// <summary>
	/// Wrapper class for nvjpegHandle
	/// </summary>
	public class NvJpeg : IDisposable
	{
		private nvjpegHandle _handle;
		private nvjpegStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public NvJpeg()
		{
			_handle = new nvjpegHandle();
			res = NvJpegNativeMethods.nvjpegCreateSimple(ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegCreateSimple", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// </summary>
		public NvJpeg(nvjpegBackend backend, nvjpegDevAllocator devAllocator, nvjpegPinnedAllocator pinnedAllocator, uint flags)
		{
			_handle = new nvjpegHandle();
			res = NvJpegNativeMethods.nvjpegCreateEx(backend, ref devAllocator, ref pinnedAllocator, flags, ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegCreateEx", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// </summary>
		public NvJpeg(nvjpegBackend backend, nvjpegPinnedAllocator pinnedAllocator, uint flags)
		{
			_handle = new nvjpegHandle();
			res = NvJpegNativeMethods.nvjpegCreateEx(backend, IntPtr.Zero, ref pinnedAllocator, flags, ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegCreateEx", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// </summary>
		public NvJpeg(nvjpegBackend backend, nvjpegDevAllocator devAllocator, uint flags)
		{
			_handle = new nvjpegHandle();
			res = NvJpegNativeMethods.nvjpegCreateEx(backend, ref devAllocator, IntPtr.Zero, flags, ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegCreateEx", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// </summary>
		public NvJpeg(nvjpegBackend backend, uint flags)
		{
			_handle = new nvjpegHandle();
			res = NvJpegNativeMethods.nvjpegCreateEx(backend, IntPtr.Zero, IntPtr.Zero, flags, ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegCreateEx", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~NvJpeg()
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
				//Ignore if failing
				res = NvJpegNativeMethods.nvjpegDestroy(_handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public nvjpegHandle Handle
		{
			get { return _handle; }
		}

		#region Create Methods
		public DecoderState CreateJpegState()
		{
			return new DecoderState(this);
		}

		public EncoderState CreateEncoderState(CudaStream stream)
		{
			return new EncoderState(this, stream);
		}

		public EncoderParams CreateEncoderParams(CudaStream stream)
		{
			return new EncoderParams(this, stream);
		}

		public DecodeParams CreateDecodeParams()
		{
			return new DecodeParams(this);
		}

		public BufferPinned CreateBufferPinned(nvjpegPinnedAllocator pinnedAllocator)
		{
			return new BufferPinned(this, pinnedAllocator);
		}

		public BufferDevice CreateBufferDevice(nvjpegDevAllocator deviceAllocator)
		{
			return new BufferDevice(this, deviceAllocator);
		}

		public JpegStream CreateJpegStream()
		{
			return new JpegStream(this);
		}

		public JpegDecoder CreateJpegDecoder(nvjpegBackend backend)
		{
			return new JpegDecoder(this, backend);
		}
		#endregion

		#region API Getter/Setter
		public SizeT DeviceMemoryPadding
		{
			get
			{
				SizeT value = new SizeT();
				res = NvJpegNativeMethods.nvjpegGetDeviceMemoryPadding(ref value, _handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegGetDeviceMemoryPadding", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
			set
			{
				res = NvJpegNativeMethods.nvjpegSetDeviceMemoryPadding(value, _handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegSetDeviceMemoryPadding", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
			}
		}
		public SizeT PinnedMemoryPadding
		{
			get
			{
				SizeT value = new SizeT();
				res = NvJpegNativeMethods.nvjpegGetPinnedMemoryPadding(ref value, _handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegGetPinnedMemoryPadding", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
			set
			{
				res = NvJpegNativeMethods.nvjpegSetPinnedMemoryPadding(value, _handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegSetPinnedMemoryPadding", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
			}
		}
		#endregion

		#region API
		public void GetImageInfo(IntPtr data, SizeT length, ref int nComponents, ref nvjpegChromaSubsampling subsampling, int[] widths, int[] heights)
		{
			res = NvJpegNativeMethods.nvjpegGetImageInfo(_handle, data, length, ref nComponents, ref subsampling, widths, heights);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegGetImageInfo", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}
		public void GetImageInfo(byte[] data, ref int nComponents, ref nvjpegChromaSubsampling subsampling, int[] widths, int[] heights)
		{
			res = NvJpegNativeMethods.nvjpegGetImageInfo(_handle, data, data.Length, ref nComponents, ref subsampling, widths, heights);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegGetImageInfo", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}
		#endregion
	}
}
