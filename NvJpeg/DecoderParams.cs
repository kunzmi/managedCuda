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
	/// Wrapper class for nvjpegDecodeParams
	/// </summary>
	public class DecodeParams : IDisposable
	{
		private nvjpegDecodeParams _params;
		private NvJpeg _nvJpeg;
		private nvjpegStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		internal DecodeParams(NvJpeg nvJpeg)
		{
			_nvJpeg = nvJpeg;
			_params = new nvjpegDecodeParams();
			res = NvJpegNativeMethods.nvjpegDecodeParamsCreate(nvJpeg.Handle, ref _params);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeParamsCreate", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~DecodeParams()
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
				res = NvJpegNativeMethods.nvjpegDecodeParamsDestroy(_params);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeParamsDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public nvjpegDecodeParams Params
		{
			get { return _params; }
		}

		///////////////////////////////////////////////////////////////////////////////////
		// Decode parameters //
		///////////////////////////////////////////////////////////////////////////////////

		// set output pixel format - same value as in nvjpegDecode()
		public void SetOutputFormat(nvjpegOutputFormat output_format)
		{
			res = NvJpegNativeMethods.nvjpegDecodeParamsSetOutputFormat(_params, output_format);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeParamsSetOutputFormat", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		// set to desired ROI. set to (0, 0, -1, -1) to disable ROI decode (decode whole image)
		public void SetROI(int offset_x, int offset_y, int roi_width, int roi_height)
		{
			res = NvJpegNativeMethods.nvjpegDecodeParamsSetROI(_params, offset_x, offset_y, roi_width, roi_height);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeParamsSetROI", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		// set to true to allow conversion from CMYK to RGB or YUV that follows simple subtractive scheme
		public void SetAllowCMYK(int allow_cmyk)
		{
			res = NvJpegNativeMethods.nvjpegDecodeParamsSetAllowCMYK(_params, allow_cmyk);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeParamsSetAllowCMYK", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}


	}
}
