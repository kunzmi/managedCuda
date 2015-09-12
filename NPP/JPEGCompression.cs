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
using System.Runtime.InteropServices;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NPP
{
	/// <summary>
	/// The JPEG standard defines a flow of level shift, DCT and quantization for
	/// forward JPEG transform and inverse level shift, IDCT and de-quantization
	/// for inverse JPEG transform. This group has the functions for both forward
	/// and inverse functions. 
	/// </summary>
	public class JPEGCompression : IDisposable
	{
		NppStatus status;
		NppiDCTState _state;
		bool disposed;

		#region Constructors
		/// <summary>
		/// Initializes DCT state structure and allocates additional resources
		/// </summary>
		public JPEGCompression()
		{
			_state = new NppiDCTState();
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDCTInitAlloc(ref _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDCTInitAlloc", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

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

				//Ignore if failing
				status = NPPNativeMethods.NPPi.CompressionDCT.nppiDCTFree(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDCTFree", status));

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("NPP not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region static old API

		/// <summary>
		/// Apply quality factor to raw 8-bit quantization table.<para/>
		/// This is effectively and in-place method that modifies a given raw
		/// quantization table based on a quality factor.<para/>
		/// Note that this method is a host method and that the pointer to the
		/// raw quantization table is a host pointer.
		/// </summary>
		/// <param name="QuantRawTable">Raw quantization table.</param>
		/// <param name="nQualityFactor">Quality factor for the table. Range is [1:100].</param>
		public static void QuantFwdRawTableInit(byte[] QuantRawTable, int nQualityFactor)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.ImageCompression.nppiQuantFwdRawTableInit_JPEG_8u(QuantRawTable, nQualityFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQuantFwdRawTableInit_JPEG_8u", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Initializes a quantization table for DCTQuantFwd8x8LS().<para/>
		/// The method creates a 16-bit version of the raw table and converts the 
		/// data order from zigzag layout to original row-order layout since raw
		/// quantization tables are typically stored in zigzag format.<para/>
		/// This method is a host method. It consumes and produces host data. I.e. the pointers
		/// passed to this function must be host pointers. The resulting table needs to be
		/// transferred to device memory in order to be used with nppiDCTQuantFwd8x8LS()
		/// function.
		/// </summary>
		/// <param name="QuantRawTable">Host pointer to raw quantization table as returned by 
		/// QuantFwdRawTableInit(). The raw quantization table is assumed to be in
		/// zigzag order.</param>
		/// <param name="QuantFwdRawTable">Forward quantization table for use with DCTQuantFwd8x8LS().</param>
		public static void QuantFwdTableInit(byte[] QuantRawTable, ushort[] QuantFwdRawTable)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.ImageCompression.nppiQuantFwdTableInit_JPEG_8u16u(QuantRawTable, QuantFwdRawTable);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQuantFwdTableInit_JPEG_8u16u", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Initializes a quantization table for DCTQuantInv8x8LS().<para/>
		/// The DCTQuantFwd8x8LS() method uses a quantization table
		/// in a 16-bit format allowing for faster processing. In addition it converts the 
		/// data order from zigzag layout to original row-order layout. Typically raw
		/// quantization tables are stored in zigzag format.<para/>
		/// This method is a host method. It consumes and produces host data. I.e. the pointers
		/// passed to this function must be host pointers. The resulting table needs to be
		/// transferred to device memory in order to be used with DCTQuantFwd8x8LS()
		/// function.
		/// </summary>
		/// <param name="QuantRawTable">Raw quantization table.</param>
		/// <param name="QuantInvRawTable">Inverse quantization table.</param>
		public static void QuantInvTableInit(byte[] QuantRawTable, ushort[] QuantInvRawTable)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.ImageCompression.nppiQuantInvTableInit_JPEG_8u16u(QuantRawTable, QuantInvRawTable);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQuantInvTableInit_JPEG_8u16u", status));
			NPPException.CheckNppStatus(status, null);
		}


		/// <summary>
		/// Forward DCT, quantization and level shift part of the JPEG encoding.
		/// Input is expected in 8x8 macro blocks and output is expected to be in 64x1
		/// macro blocks.
		/// </summary>
		/// <param name="src">Source image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="QuantFwdTable">Forward quantization tables for JPEG encoding created using QuantInvTableInit()</param>
		/// <param name="oSizeRoi">Roi size (in macro blocks?).</param>
		public static void DCTQuantFwd8x8LS(NPPImage_8uC1 src, NPPImage_16sC1 dst, CudaDeviceVariable<ushort> QuantFwdTable, NppiSize oSizeRoi)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.ImageCompression.nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R(src.DevicePointer, src.Pitch, dst.DevicePointer, dst.Pitch, QuantFwdTable.DevicePointer, oSizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R", status));
			NPPException.CheckNppStatus(status, null);
		}


		/// <summary>
		/// Inverse DCT, de-quantization and level shift part of the JPEG decoding.
		/// Input is expected in 64x1 macro blocks and output is expected to be in 8x8
		/// macro blocks.
		/// </summary>
		/// <param name="src">Source image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="QuantInvTable">Inverse quantization tables for JPEG decoding created using QuantInvTableInit()</param>
		/// <param name="oSizeRoi">Roi size (in macro blocks?).</param>
		public static void DCTQuantInv8x8LS(NPPImage_16sC1 src, NPPImage_8uC1 dst, CudaDeviceVariable<ushort> QuantInvTable, NppiSize oSizeRoi)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.ImageCompression.nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(src.DevicePointer, src.Pitch, dst.DevicePointer, dst.Pitch, QuantInvTable.DevicePointer, oSizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region new API
		/// <summary>
		/// Forward DCT, quantization and level shift part of the JPEG encoding.
		/// Input is expected in 8x8 macro blocks and output is expected to be in 64x1
		/// macro blocks. The new version of the primitive takes the ROI in image pixel size and
		/// works with DCT coefficients that are in zig-zag order.
		/// </summary>
		/// <param name="src">Source image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="QuantFwdTable">Quantization Table in zig-zag order</param>
		/// <param name="oSizeRoi">Roi size (in pixels).</param>
		public void DCTQuantFwd8x8LS(NPPImage_8uC1 src, NPPImage_16sC1 dst, NppiSize oSizeRoi, CudaDeviceVariable<byte> QuantFwdTable)
		{
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(src.DevicePointer, src.Pitch, dst.DevicePointer, dst.Pitch, QuantFwdTable.DevicePointer, oSizeRoi, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Inverse DCT, de-quantization and level shift part of the JPEG decoding.
		/// Input is expected in 64x1 macro blocks and output is expected to be in 8x8
		/// macro blocks. The new version of the primitive takes the ROI in image pixel size and
		/// works with DCT coefficients that are in zig-zag order.
		/// </summary>
		/// <param name="src">Source image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="QuantInvTable">Quantization Table in zig-zag order.</param>
		/// <param name="oSizeRoi">Roi size (in pixels).</param>
		public void DCTQuantInv8x8LS(NPPImage_16sC1 src, NPPImage_8uC1 dst, NppiSize oSizeRoi, CudaDeviceVariable<byte> QuantInvTable)
		{
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW(src.DevicePointer, src.Pitch, dst.DevicePointer, dst.Pitch, QuantInvTable.DevicePointer, oSizeRoi, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Huff coding static

		/// <summary>
		/// Huffman Decoding of the JPEG decoding on the host.<para/>
		/// Input is expected in byte stuffed huffman encoded JPEG scan and output is expected to be 64x1 macro blocks.
		/// </summary>
		/// <param name="pSrc">Byte-stuffed huffman encoded JPEG scan.</param>
		/// <param name="restartInterval">Restart Interval, see JPEG standard.</param>
		/// <param name="Ss">Start Coefficient, see JPEG standard.</param>
		/// <param name="Se">End Coefficient, see JPEG standard.</param>
		/// <param name="Ah">Bit Approximation High, see JPEG standard.</param>
		/// <param name="Al">Bit Approximation Low, see JPEG standard.</param>
		/// <param name="pDst">Destination image pointer</param>
		/// <param name="nDstStep">destination image line step.</param>
		/// <param name="pHuffmanTableDC">DC Huffman table.</param>
		/// <param name="pHuffmanTableAC">AC Huffman table.</param>
		/// <param name="oSizeROI">ROI</param>
		public static void DecodeHuffmanScanHost(byte[] pSrc, int restartInterval, int Ss, int Se, int Ah, int Al,
					short[] pDst, int nDstStep, NppiDecodeHuffmanSpec pHuffmanTableDC, NppiDecodeHuffmanSpec pHuffmanTableAC, NppiSize oSizeROI)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R(pSrc, pSrc.Length, restartInterval, Ss,Se, Ah, Al, pDst, nDstStep, pHuffmanTableDC, pHuffmanTableAC, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDecodeHuffmanScanHost_JPEG_8u16s_P1R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Huffman Decoding of the JPEG decoding on the host.<para/>
		/// Input is expected in byte stuffed huffman encoded JPEG scan and output is expected to be 64x1 macro blocks.
		/// </summary>
		/// <param name="pSrc">Byte-stuffed huffman encoded JPEG scan.</param>
		/// <param name="restartInterval">Restart Interval, see JPEG standard.</param>
		/// <param name="Ss">Start Coefficient, see JPEG standard.</param>
		/// <param name="Se">End Coefficient, see JPEG standard.</param>
		/// <param name="Ah">Bit Approximation High, see JPEG standard.</param>
		/// <param name="Al">Bit Approximation Low, see JPEG standard.</param>
		/// <param name="pDst">Destination image pointer</param>
		/// <param name="nDstStep">destination image line step.</param>
		/// <param name="pHuffmanTableDC">DC Huffman table.</param>
		/// <param name="pHuffmanTableAC">AC Huffman table.</param>
		/// <param name="oSizeROI">ROI</param>
		public static void DecodeHuffmanScanHost(byte[] pSrc, int restartInterval, int Ss, int Se, int Ah, int Al,
					IntPtr[] pDst, int[] nDstStep, NppiDecodeHuffmanSpec[] pHuffmanTableDC, NppiDecodeHuffmanSpec[] pHuffmanTableAC, NppiSize[] oSizeROI)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(pSrc, pSrc.Length, restartInterval, Ss, Se, Ah, Al, pDst, nDstStep, pHuffmanTableDC, pHuffmanTableAC, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R", status));
			NPPException.CheckNppStatus(status, null);			
		}

		/// <summary>
		/// Calculates the size of the temporary buffer for huffman encoding.
		/// </summary>
		/// <param name="oSize">Image Dimension</param>
		/// <param name="nChannels">Number of channels</param>
		/// <returns>the size of the temporary buffer</returns>
		public static int EncodeHuffmanGetSize(NppiSize oSize, int nChannels)
		{
			int size = 0;
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiEncodeHuffmanGetSize(oSize, nChannels, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEncodeHuffmanGetSize", status));
			NPPException.CheckNppStatus(status, null);
			return size;
		}


		/// <summary>
		/// Huffman Encoding of the JPEG Encoding.<para/>
		/// Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
		/// </summary>
		/// <param name="pSrc">Source image.</param>
		/// <param name="restartInterval">Restart Interval, see JPEG standard.</param>
		/// <param name="Ss">Start Coefficient, see JPEG standard.</param>
		/// <param name="Se">End Coefficient, see JPEG standard.</param>
		/// <param name="Ah">Bit Approximation High, see JPEG standard.</param>
		/// <param name="Al">Bit Approximation Low, see JPEG standard.</param>
		/// <param name="pDst">Byte-stuffed huffman encoded JPEG scan.</param>
		/// <param name="nLength">Byte length of the huffman encoded JPEG scan.</param>
		/// <param name="pHuffmanTableDC">DC Huffman table.</param>
		/// <param name="pHuffmanTableAC">AC Huffman table.</param>
		/// <param name="oSizeROI">ROI</param>
		/// <param name="buffer">Scratch buffer</param>
		public static void EnodeHuffmanScan(NPPImage_16sC1 pSrc, int restartInterval, int Ss, int Se, int Ah, int Al,
					CudaDeviceVariable<byte> pDst, ref int nLength, NppiEncodeHuffmanSpec pHuffmanTableDC, NppiEncodeHuffmanSpec pHuffmanTableAC, NppiSize oSizeROI, CudaDeviceVariable<byte> buffer)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiEncodeHuffmanScan_JPEG_8u16s_P1R(pSrc.DevicePointer, pSrc.Pitch, restartInterval, Ss, Se, Ah, Al, pDst.DevicePointer, ref nLength, pHuffmanTableDC, pHuffmanTableAC, oSizeROI, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEncodeHuffmanScan_JPEG_8u16s_P1R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Huffman Encoding of the JPEG Encoding.<para/>
		/// Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
		/// </summary>
		/// <param name="pSrc">Source image.</param>
		/// <param name="restartInterval">Restart Interval, see JPEG standard.</param>
		/// <param name="Ss">Start Coefficient, see JPEG standard.</param>
		/// <param name="Se">End Coefficient, see JPEG standard.</param>
		/// <param name="Ah">Bit Approximation High, see JPEG standard.</param>
		/// <param name="Al">Bit Approximation Low, see JPEG standard.</param>
		/// <param name="pDst">Byte-stuffed huffman encoded JPEG scan.</param>
		/// <param name="nLength">Byte length of the huffman encoded JPEG scan.</param>
		/// <param name="pHuffmanTableDC">DC Huffman table.</param>
		/// <param name="pHuffmanTableAC">AC Huffman table.</param>
		/// <param name="oSizeROI">ROI</param>
		/// <param name="buffer">Scratch buffer</param>
		public static void EncodeHuffmanScan(NPPImage_16sC1[] pSrc, int restartInterval, int Ss, int Se, int Ah, int Al,
					CudaDeviceVariable<byte> pDst, ref int nLength, NppiEncodeHuffmanSpec[] pHuffmanTableDC, NppiEncodeHuffmanSpec[] pHuffmanTableAC, NppiSize[] oSizeROI, CudaDeviceVariable<byte> buffer)
		{
			NppStatus status;

			CUdeviceptr[] srcs = new CUdeviceptr[] { pSrc[0].DevicePointer, pSrc[1].DevicePointer, pSrc[2].DevicePointer };
			int[] steps = new int[] { pSrc[0].Pitch, pSrc[1].Pitch, pSrc[2].Pitch };

			status = NPPNativeMethods.NPPi.CompressionDCT.nppiEncodeHuffmanScan_JPEG_8u16s_P3R(srcs, steps, restartInterval, Ss, Se, Ah, Al, pDst.DevicePointer, ref nLength, pHuffmanTableDC, pHuffmanTableAC, oSizeROI, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEncodeHuffmanScan_JPEG_8u16s_P3R", status));
			NPPException.CheckNppStatus(status, null);
			
		}

		/// <summary>
		/// Returns the length of the NppiDecodeHuffmanSpec structure.
		/// </summary>
		/// <returns>the length of the NppiDecodeHuffmanSpec structure.</returns>
		public static int DecodeHuffmanSpecGetBufSize_JPEG()
		{
			NppStatus status;
			int res = 0;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDecodeHuffmanSpecGetBufSize_JPEG(ref res);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDecodeHuffmanSpecGetBufSize_JPEG", status));
			NPPException.CheckNppStatus(status, null);
			return res;
		}


		/// <summary>
		/// Creates a Huffman table in a format that is suitable for the decoder on the host.
		/// </summary>
		/// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
		/// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable)</param>
		/// <param name="pHuffmanSpec">Pointer to the Huffman table for the decoder</param>
		/// <returns>NPP_NULL_POINTER_ERROR If one of the pointers is 0.</returns>
		public static void DecodeHuffmanSpecInitHost_JPEG(byte[] pRawHuffmanTable, NppiHuffmanTableType eTableType, NppiDecodeHuffmanSpec pHuffmanSpec)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDecodeHuffmanSpecInitHost_JPEG(pRawHuffmanTable, eTableType, pHuffmanSpec);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDecodeHuffmanSpecInitHost_JPEG", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Allocates memory and creates a Huffman table in a format that is suitable for the decoder on the host.
		/// </summary>
		/// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
		/// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable).</param>
		/// <returns>Huffman table for the decoder</returns>
		public static NppiDecodeHuffmanSpec DecodeHuffmanSpecInitAllocHost_JPEG(byte[] pRawHuffmanTable, NppiHuffmanTableType eTableType)
		{
			NppiDecodeHuffmanSpec spec = new NppiDecodeHuffmanSpec();
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDecodeHuffmanSpecInitAllocHost_JPEG(pRawHuffmanTable, eTableType, ref spec);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDecodeHuffmanSpecInitAllocHost_JPEG", status));
			NPPException.CheckNppStatus(status, null);
			return spec;
		}

		/// <summary>
		/// Frees the host memory allocated by nppiDecodeHuffmanSpecInitAllocHost_JPEG.
		/// </summary>
		/// <param name="pHuffmanSpec">Pointer to the Huffman table for the decoder</param>
		public static void DecodeHuffmanSpecFreeHost_JPEG(NppiDecodeHuffmanSpec pHuffmanSpec)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiDecodeHuffmanSpecFreeHost_JPEG(pHuffmanSpec);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDecodeHuffmanSpecFreeHost_JPEG", status));
			NPPException.CheckNppStatus(status, null);
		}




		/// <summary>
		/// Returns the length of the NppiEncodeHuffmanSpec structure.
		/// </summary>
		/// <returns>length of the NppiEncodeHuffmanSpec structure.</returns>
		public static int EncodeHuffmanSpecGetBufSize_JPEG()
		{
			NppStatus status;
			int res = 0;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiEncodeHuffmanSpecGetBufSize_JPEG(ref res);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEncodeHuffmanSpecGetBufSize_JPEG", status));
			NPPException.CheckNppStatus(status, null);
			return res;
		}

		/// <summary>
		/// Creates a Huffman table in a format that is suitable for the encoder.
		/// </summary>
		/// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
		/// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable).</param>
		/// <param name="pHuffmanSpec">Pointer to the Huffman table for the decoder</param>
		/// <returns>Huffman table for the encoder</returns>
		public static void EncodeHuffmanSpecInit_JPEG(byte[] pRawHuffmanTable, NppiHuffmanTableType eTableType, NppiEncodeHuffmanSpec pHuffmanSpec)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiEncodeHuffmanSpecInit_JPEG(pRawHuffmanTable, eTableType, pHuffmanSpec);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEncodeHuffmanSpecInit_JPEG", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Allocates memory and creates a Huffman table in a format that is suitable for the encoder.
		/// </summary>
		/// <param name="pRawHuffmanTable">Huffman table formated as specified in the JPEG standard.</param>
		/// <param name="eTableType">Enum specifying type of table (nppiDCTable or nppiACTable).</param>
		/// <returns>Huffman table for the encoder.</returns>
		public static NppiEncodeHuffmanSpec EncodeHuffmanSpecInitAllocHost_JPEG(byte[] pRawHuffmanTable, NppiHuffmanTableType eTableType)
		{
			NppiEncodeHuffmanSpec spec = new NppiEncodeHuffmanSpec();
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiEncodeHuffmanSpecInitAlloc_JPEG(pRawHuffmanTable, eTableType, ref spec);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEncodeHuffmanSpecInitAlloc_JPEG", status));
			NPPException.CheckNppStatus(status, null);
			return spec;
		}

		/// <summary>
		/// Frees the memory allocated by nppiEncodeHuffmanSpecInitAlloc_JPEG.
		/// </summary>
		/// <param name="pHuffmanSpec">Pointer to the Huffman table for the encoder</param>
		public static void EncodeHuffmanSpecFree_JPEG(NppiEncodeHuffmanSpec pHuffmanSpec)
		{
			NppStatus status;
			status = NPPNativeMethods.NPPi.CompressionDCT.nppiEncodeHuffmanSpecFree_JPEG(pHuffmanSpec);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEncodeHuffmanSpecFree_JPEG", status));
			NPPException.CheckNppStatus(status, null);
		}

		#endregion
	}
}
