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
	/// Wrapper class for nvjpegJpegStream
	/// </summary>
	public class JpegStream : IDisposable
	{
		private nvjpegJpegStream _stream;
		private NvJpeg _nvJpeg;
		private nvjpegStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		internal JpegStream(NvJpeg nvJpeg)
		{
			_nvJpeg = nvJpeg;
			_stream = new nvjpegJpegStream();
			res = NvJpegNativeMethods.nvjpegJpegStreamCreate(nvJpeg.Handle, ref _stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamCreate", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~JpegStream()
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
				res = NvJpegNativeMethods.nvjpegJpegStreamDestroy(_stream);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public nvjpegJpegStream Stream
		{
			get { return _stream; }
		}




		public void Parse(IntPtr data, SizeT length, int save_metadata, int save_stream)
		{
			res = NvJpegNativeMethods.nvjpegJpegStreamParse(_nvJpeg.Handle, data, length, save_metadata, save_stream, _stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamParse", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}
		public void Parse(byte[] data, int save_metadata, int save_stream)
		{
			res = NvJpegNativeMethods.nvjpegJpegStreamParse(_nvJpeg.Handle, data, data.Length, save_metadata, save_stream, _stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamParse", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		public nvjpegJpegEncoding JpegEncoding
		{
			get
			{
				nvjpegJpegEncoding value = new nvjpegJpegEncoding();
				res = NvJpegNativeMethods.nvjpegJpegStreamGetJpegEncoding(_stream, ref value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamGetJpegEncoding", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
		}

		public uint Width
		{
			get
			{
				uint value = 0;
				uint dummy = 0;
				res = NvJpegNativeMethods.nvjpegJpegStreamGetFrameDimensions(_stream, ref value, ref dummy);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamGetJpegEncoding", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
		}
		public uint Height
		{
			get
			{
				uint dummy = 0;
				uint value = 0;
				res = NvJpegNativeMethods.nvjpegJpegStreamGetFrameDimensions(_stream, ref dummy, ref value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamGetJpegEncoding", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
		}
		public uint ComponentsNum
		{
			get
			{
				uint value = 0;
				res = NvJpegNativeMethods.nvjpegJpegStreamGetComponentsNum(_stream, ref value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamGetComponentsNum", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
		}

		// if encoded is 1 color component then it assumes 4:0:0 (NVJPEG_CSS_GRAY, grayscale)
		// if encoded is 3 color components it tries to assign one of the known subsamplings
		//   based on the components subsampling infromation
		// in case sampling factors are not stadard or number of components is different 
		//   it will return NVJPEG_CSS_UNKNOWN
		public nvjpegChromaSubsampling ChromaSubsampling
		{
			get
			{
				nvjpegChromaSubsampling value = 0;
				res = NvJpegNativeMethods.nvjpegJpegStreamGetChromaSubsampling(_stream, ref value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamGetChromaSubsampling", res));
				if (res != nvjpegStatus.Success)
					throw new NvJpegException(res);
				return value;
			}
		}


		public uint[] ComponentWidths
		{
			get
			{
				uint count = ComponentsNum;

				uint[] widths = new uint[count];
				for (uint i = 0; i < count; i++)
				{
					uint value = 0;
					uint dummy = 0;
					res = NvJpegNativeMethods.nvjpegJpegStreamGetComponentDimensions(_stream, i, ref value, ref dummy);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamGetComponentDimensions", res));
					if (res != nvjpegStatus.Success)
						throw new NvJpegException(res);
					widths[i] = value;
				}
				return widths;
			}
		}
		public uint[] ComponentHeights
		{
			get
			{
				uint count = ComponentsNum;

				uint[] heights = new uint[count];
				for (uint i = 0; i < count; i++)
				{
					uint dummy = 0;
					uint value = 0;
					res = NvJpegNativeMethods.nvjpegJpegStreamGetComponentDimensions(_stream, i, ref dummy, ref value);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStreamGetComponentDimensions", res));
					if (res != nvjpegStatus.Success)
						throw new NvJpegException(res);
					heights[i] = value;
				}
				return heights;
			}
		}


	}
}
