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
	/// Wrapper class for nvjpegJpegState
	/// </summary>
	public class DecoderState : IDisposable
	{
		private nvjpegJpegState _state;
		private NvJpeg _nvJpeg;
		private JpegDecoder _decoder;
		private nvjpegStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		internal DecoderState(NvJpeg nvJpeg)
		{
			_nvJpeg = nvJpeg;
			_decoder = null;
			_state = new nvjpegJpegState();
			res = NvJpegNativeMethods.nvjpegJpegStateCreate(nvJpeg.Handle, ref _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStateCreate", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}
		/// <summary>
		/// </summary>
		internal DecoderState(NvJpeg nvJpeg, JpegDecoder decoder)
		{
			_nvJpeg = nvJpeg;
			_decoder = decoder;
			_state = new nvjpegJpegState();
			res = NvJpegNativeMethods.nvjpegDecoderStateCreate(nvJpeg.Handle, decoder.Decoder, ref _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecoderStateCreate", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~DecoderState()
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
				res = NvJpegNativeMethods.nvjpegJpegStateDestroy(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegJpegStateDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public nvjpegJpegState State
		{
			get { return _state; }
		}

		/// <summary>
		/// Decodes a single image, and writes the decoded image in the desired format to the output buffers. This function is asynchronous with respect to the host. All GPU tasks for this function will be submitted to the provided stream.
		/// </summary>
		/// <param name="data">the encoded data.</param>
		/// <param name="length">Size of the encoded data in bytes.</param>
		/// <param name="output_format">Format in which the decoded output will be saved.</param>
		/// <param name="destination">Pointer to the structure that describes the output destination. This structure should be on the host (CPU), but the pointers in this structure should be pointing to the device (i.e., GPU) memory.</param>
		/// <param name="stream">The CUDA stream where all of the GPU work will be submitted.</param>
		public void Decode(IntPtr data, SizeT length, nvjpegOutputFormat output_format, ref nvjpegImage destination, CudaStream stream)
		{
			res = NvJpegNativeMethods.nvjpegDecode(_nvJpeg.Handle, _state, data, length, output_format, ref destination, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecode", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		/// <summary>
		/// Decodes a single image, and writes the decoded image in the desired format to the output buffers. This function is asynchronous with respect to the host. All GPU tasks for this function will be submitted to the provided stream.<para/>
		/// Note: Synchronizes on stream. For async use IntPtr!
		/// </summary>
		/// <param name="data">the encoded data.</param>
		/// <param name="output_format">Format in which the decoded output will be saved.</param>
		/// <param name="destination">Pointer to the structure that describes the output destination. This structure should be on the host (CPU), but the pointers in this structure should be pointing to the device (i.e., GPU) memory.</param>
		/// <param name="stream">The CUDA stream where all of the GPU work will be submitted.</param>
		public void Decode(byte[] data, nvjpegOutputFormat output_format, ref nvjpegImage destination, CudaStream stream)
		{
			GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);

			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				res = NvJpegNativeMethods.nvjpegDecode(_nvJpeg.Handle, _state, ptr, data.Length, output_format, ref destination, stream.Stream);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecode", res));
				stream.Synchronize();
			}
			finally
			{
				handle.Free();
			}
			
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		public void DecodeBatchedInitialize(int batch_size, int max_cpuhreads, nvjpegOutputFormat output_format)
		{
			res = NvJpegNativeMethods.nvjpegDecodeBatchedInitialize(_nvJpeg.Handle, _state, batch_size, max_cpuhreads, output_format);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeBatchedInitialize", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		public void DecodeBatchedInitialize(IntPtr[] data, SizeT[] lengths, nvjpegImage[] destinations, CUstream stream)
		{
			res = NvJpegNativeMethods.nvjpegDecodeBatched(_nvJpeg.Handle, _state, data, lengths, destinations, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeBatched", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		public void AttachBuffer(BufferPinned buffer)
		{
			res = NvJpegNativeMethods.nvjpegStateAttachPinnedBuffer(_state, buffer.Buffer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegStateAttachPinnedBuffer", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		public void AttachBuffer(BufferDevice buffer)
		{
			res = NvJpegNativeMethods.nvjpegStateAttachDeviceBuffer(_state, buffer.Buffer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegStateAttachDeviceBuffer", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		// starts decoding on host and save decode parameters to the state
		public void DecodeJpegHost(DecodeParams param, JpegStream stream)
		{
			res = NvJpegNativeMethods.nvjpegDecodeJpegHost(_nvJpeg.Handle, _decoder.Decoder, _state, param.Params, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeJpegHost", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		// hybrid stage of decoding image,  involves device async calls
		// note that jpeg stream is a parameter here - because we still might need copy 
		// parts of bytestream to device
		public void DecodeJpegTransferToDevice(JpegStream jpegStream, CudaStream cudaStream)
		{
			res = NvJpegNativeMethods.nvjpegDecodeJpegTransferToDevice(_nvJpeg.Handle, _decoder.Decoder, _state, jpegStream.Stream, cudaStream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeJpegTransferToDevice", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}

		// finishing async operations on the device
		public void DecodeJpegDevice(ref nvjpegImage destination, CudaStream cudaStream)
		{
			res = NvJpegNativeMethods.nvjpegDecodeJpegDevice(_nvJpeg.Handle, _decoder.Decoder, _state, ref destination, cudaStream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegDecodeJpegDevice", res));
			if (res != nvjpegStatus.Success)
				throw new NvJpegException(res);
		}



	}
}
