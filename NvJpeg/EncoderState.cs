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
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NvJpeg
{

	/// <summary>
	/// Wrapper class for nvjpegEncoderState
	/// </summary>
	public class EncoderState : IDisposable
	{
		private nvjpegEncoderState _state;
		private NvJpeg _nvJpeg;
		private nvjpegStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		internal EncoderState(NvJpeg nvJpeg, CudaStream stream)
		{
			_nvJpeg = nvJpeg;
			_state = new nvjpegEncoderState();
			res = NvJpegNativeMethods.nvjpegEncoderStateCreate(nvJpeg.Handle, ref _state, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderStateCreate", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~EncoderState()
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
				res = NvJpegNativeMethods.nvjpegEncoderStateDestroy(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderStateDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public nvjpegEncoderState State
		{
			get { return _state; }
		}


		public void EncodeYUV(EncoderParams encoderParams, nvjpegImage source, nvjpegChromaSubsampling chromaSubsampling, int image_width, int image_height, CudaStream stream)
		{
			res = NvJpegNativeMethods.nvjpegEncodeYUV(_nvJpeg.Handle, _state, encoderParams.Params, ref source, chromaSubsampling, image_width, image_height, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncodeYUV", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}
		public void EncodeImage(EncoderParams encoderParams, nvjpegImage source, nvjpegInputFormat input_format, int image_width, int image_height, CudaStream stream)
		{
			res = NvJpegNativeMethods.nvjpegEncodeImage(_nvJpeg.Handle, _state, encoderParams.Params, ref source, input_format, image_width, image_height, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncodeImage", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// Retrieves the compressed stream from the encoder state that was previously used in one of the encoder functions.
		/// </summary>
		/// <param name="ptr">Pointer to the buffer in the host memory where the compressed stream will be stored. Can be NULL</param>
		/// <param name="length">input buffer size.</param>
		/// <param name="stream">CUDA stream where all the required device operations will be placed.</param>
		/// <returns>On return the NVJPEG library will give the actual compressed stream size in this value.</returns>
		public SizeT RetrieveBitstream(IntPtr ptr, SizeT length, CudaStream stream)
		{
			res = NvJpegNativeMethods.nvjpegEncodeRetrieveBitstream(_nvJpeg.Handle, _state, ptr, ref length, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncodeRetrieveBitstream", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);

			return length;
		}

		/// <summary>
		/// Retrieves the compressed stream from the encoder state that was previously used in one of the encoder functions.<para/>
		/// Note: Synchronizes on stream. For async use IntPtr!
		/// </summary>
		/// <param name="data">Buffer in the host memory where the compressed stream will be stored.</param>
		/// <param name="stream">CUDA stream where all the required device operations will be placed.</param>
		/// <returns>On return the NVJPEG library will give the actual compressed stream size in this value.</returns>
		public SizeT RetrieveBitstream(byte[] data, CudaStream stream)
		{
			SizeT size = data.Length;
			GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
			
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = NvJpegNativeMethods.nvjpegEncodeRetrieveBitstream(_nvJpeg.Handle, _state, ptr, ref size, stream.Stream);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncodeRetrieveBitstream", res));
				stream.Synchronize();
			}
			finally
			{
				handle.Free();
			}
			
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);

			return size;
		}

		/// <summary>
		/// Retrieves the compressed stream from the encoder state that was previously used in one of the encoder functions.<para/>
		/// Note: Synchronizes on stream. For async use IntPtr!
		/// </summary>
		/// <param name="stream">CUDA stream where all the required device operations will be placed.</param>
		/// <returns>Byte array containing the data.</returns>
		public byte[] RetrieveBitstream(CudaStream stream)
		{
			SizeT size = new SizeT();
			res = NvJpegNativeMethods.nvjpegEncodeRetrieveBitstream(_nvJpeg.Handle, _state, IntPtr.Zero, ref size, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncodeRetrieveBitstream", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);

			byte[] data = new byte[size];
			GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = NvJpegNativeMethods.nvjpegEncodeRetrieveBitstream(_nvJpeg.Handle, _state, ptr, ref size, stream.Stream);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncodeRetrieveBitstream", res));
				stream.Synchronize();
			}
			finally
			{
				handle.Free();
			}
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);

			return data;
		}


		// copies metadata (JFIF, APP, EXT, COM markers) from parsed stream
		public void CopyMetadata(EncoderParams encoderParams, JpegStream jpeg, CudaStream stream)
		{
			res = NvJpegNativeMethods.nvjpegEncoderParamsCopyMetadata(_state, encoderParams.Params, jpeg.Stream, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsCopyMetadata", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		// copies huffman tables from parsed stream. should require same scans structure
		public void CopyHuffmanTables(EncoderParams encoderParams, JpegStream jpeg, CudaStream stream)
		{
			res = NvJpegNativeMethods.nvjpegEncoderParamsCopyHuffmanTables(_state, encoderParams.Params, jpeg.Stream, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsCopyHuffmanTables", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

	}
}
