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
	/// Wrapper class for nvjpegBufferDevice
	/// </summary>
	public class BufferDevice : IDisposable
	{
		private nvjpegBufferDevice _buffer;
		private NvJpeg _nvJpeg;
		private nvjpegStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		internal BufferDevice(NvJpeg nvJpeg, nvjpegDevAllocator deviceAllocator)
		{
			_nvJpeg = nvJpeg;
			_buffer = new nvjpegBufferDevice();
			res = NvJpegNativeMethods.nvjpegBufferDeviceCreate(nvJpeg.Handle, ref deviceAllocator, ref _buffer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferDeviceCreate", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~BufferDevice()
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
				res = NvJpegNativeMethods.nvjpegBufferDeviceDestroy(_buffer);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferDeviceDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public nvjpegBufferDevice Buffer
		{
			get { return _buffer; }
		}


		public SizeT Size
		{
			get
			{
				SizeT value = 0;
				CUdeviceptr dummy = new CUdeviceptr();
				res = NvJpegNativeMethods.nvjpegBufferDeviceRetrieve(_buffer, ref value, ref dummy);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferDeviceRetrieve", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
		}

		public CUdeviceptr Ptr
		{
			get
			{
				SizeT dummy = 0;
				CUdeviceptr value = new CUdeviceptr();
				res = NvJpegNativeMethods.nvjpegBufferDeviceRetrieve(_buffer, ref dummy, ref value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferDeviceRetrieve", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
		}

	}
}
