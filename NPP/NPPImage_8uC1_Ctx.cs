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
	/// 
	/// </summary>
	public partial class NPPImage_8uC1 : NPPImageBase
	{
		#region Copy
		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="mask">Mask image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_8uC1 dst, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_8u_C1MR_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_8uC3 dst, int channel, NppStreamContext nppStreamCtx)
		{
			if (channel < 0 | channel >= dst.Channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_8u_C1C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi + channel * _typeSize, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C1C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		
		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_8uC4 dst, int channel, NppStreamContext nppStreamCtx) 
		{
			if (channel < 0 | channel >= dst.Channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..3].");
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_8u_C1C4R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi + channel * _typeSize, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C1C4R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		
		/// <summary>
		/// Copy image and pad borders with a constant, user-specifiable color.
		/// </summary>
		/// <param name="dst">Destination image. The image ROI defines the destination region, i.e. the region that gets filled with data from
		/// the source image (inner part) and constant border color (outer part).</param>
		/// <param name="nTopBorderHeight">Height (in pixels) of the top border. The height of the border at the bottom of
		/// the destination ROI is implicitly defined by the size of the source ROI: nBottomBorderHeight =
		/// oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
		/// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		/// <param name="nValue">The pixel value to be set for border pixels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_8uC1 dst, int nTopBorderHeight, int nLeftBorderWidth, byte nValue, NppStreamContext nppStreamCtx) 
		{
			status = NPPNativeMethods_Ctx.NPPi.CopyConstBorder.nppiCopyConstBorder_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Set(byte nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_8u_C1R_Ctx(nValue, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Set pixel values to nValue. <para/>
		/// The 8-bit mask image affects setting of the respective pixels in the destination image. <para/>
		/// If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
		/// destination pixel is set to specified value.
		/// </summary>
		/// <param name="nValue">Value to be set</param>
		/// <param name="mask">Mask image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Set(byte nValue, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_8u_C1MR_Ctx(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Convert
		/// <summary>
		/// 8-bit unsigned to 16-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_16sC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned to 16-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_16uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_8u16u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u16u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned to 32-bit floating point conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_8u32f_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned to 8-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Round mode</param>
		/// <param name="scaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_8sC1 dst, NppRoundMode roundMode, int scaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_8u8s_C1RSfs_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u8s_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned to 32-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_32sC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_8u32s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u32s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Transpose
		/// <summary>
		/// image transpose
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Transpose(NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Transpose.nppiTranspose_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sum
		/// <summary>
		/// Scratch-buffer size for nppiSum_8u_C1R.
		/// </summary>
		/// <returns></returns>
		public int SumDoubleGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSumGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for nppiSum_8u64s_C1R.
		/// </summary>
		/// <returns></returns>
		public int SumLongGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSumGetBufferHostSize_8u64s_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumGetBufferHostSize_8u64s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image sum with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sum(CudaDeviceVariable<double> result, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SumDoubleGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSum_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SumDoubleGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sum(CudaDeviceVariable<double> result, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SumDoubleGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSum_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit long long result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 1 * sizeof(long)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sum(CudaDeviceVariable<long> result, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SumLongGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSum_8u64s_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_8u64s_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit long long result. No additional buffer is allocated.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 1 * sizeof(long)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SumLongGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sum(CudaDeviceVariable<long> result, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SumLongGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSum_8u64s_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_8u64s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Add
		/// <summary>
		/// Image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(NPPImage_8uC1 src2, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(NPPImage_8uC1 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_8u_C1IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value to add</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(byte nConstant, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to add</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(byte nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_8u_C1IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MeanStdDev
		/// <summary>
		/// Scratch-buffer size for MeanStdDev.
		/// </summary>
		/// <returns></returns>
		public int MeanStdDevGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for MeanStdDev (masked).
		/// </summary>
		/// <returns></returns>
		public int MeanStdDevMaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanStdDevGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanStdDevMaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Histogram
		/// <summary>
		/// Scratch-buffer size for HistogramEven.
		/// </summary>
		/// <param name="nLevels"></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// <returns></returns>
		public int HistogramEvenGetBufferSize(int nLevels, NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(_sizeRoi, nLevels, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEvenGetBufferSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Compute levels with even distribution.
		/// </summary>
		/// <param name="nLevels">The number of levels being computed. nLevels must be at least 2, otherwise an NPP_-
		/// HISTO_NUMBER_OF_LEVELS_ERROR error is returned.</param>
		/// <param name="nLowerBound">Lower boundary value of the lowest level.</param>
		/// <param name="nUpperBound">Upper boundary value of the greatest level.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// <returns>An array of size nLevels which receives the levels being computed.</returns>
		public int[] EvenLevels(int nLevels, int nLowerBound, int nUpperBound, NppStreamContext nppStreamCtx)
		{
			int[] Levels = new int[nLevels];
			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiEvenLevelsHost_32s_Ctx(Levels, nLevels, nLowerBound, nUpperBound, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEvenLevelsHost_32s_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return Levels;
		}

		/// <summary>
		/// Histogram with evenly distributed bins. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="histogram">Allocated device memory of size nLevels</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void HistogramEven(CudaDeviceVariable<int> histogram, int nLowerLevel, int nUpperLevel, NppStreamContext nppStreamCtx)
		{
			int bufferSize = HistogramEvenGetBufferSize(histogram.Size + 1, nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramEven_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, histogram.DevicePointer, histogram.Size + 1, nLowerLevel, nUpperLevel, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with evenly distributed bins. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">Allocated device memory of size nLevels</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramEvenGetBufferSize(int)"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void HistogramEven(CudaDeviceVariable<int> histogram, int nLowerLevel, int nUpperLevel, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = HistogramEvenGetBufferSize(histogram.Size + 1, nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramEven_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, histogram.DevicePointer, histogram.Size + 1, nLowerLevel, nUpperLevel, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Scratch-buffer size for HistogramRange.
		/// </summary>
		/// <param name="nLevels"></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// <returns></returns>
		public int HistogramRangeGetBufferSize(int nLevels, NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(_sizeRoi, nLevels, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRangeGetBufferSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The array must be of size nLevels-1.</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The array must be of size nLevels</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void HistogramRange(CudaDeviceVariable<int> histogram, CudaDeviceVariable<int> pLevels, NppStreamContext nppStreamCtx)
		{
			int bufferSize = HistogramRangeGetBufferSize(histogram.Size, nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramRange_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, histogram.DevicePointer, pLevels.DevicePointer, pLevels.Size, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The array must be of size nLevels-1.</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The array must be of size nLevels</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramRangeGetBufferSize(int)"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void HistogramRange(CudaDeviceVariable<int> histogram, CudaDeviceVariable<int> pLevels, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = HistogramRangeGetBufferSize(histogram.Size, nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramRange_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, histogram.DevicePointer, pLevels.DevicePointer, pLevels.Size, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Min
		/// <summary>
		/// Scratch-buffer size for Min.
		/// </summary>
		/// <returns></returns>
		public int MinGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Min.nppiMinGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Min(CudaDeviceVariable<byte> min, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Min.nppiMin_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Min(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Min.nppiMin_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinIndex
		/// <summary>
		/// Scratch-buffer size for MinIndex.
		/// </summary>
		/// <returns></returns>
		public int MinIndexGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MinIdx.nppiMinIndxGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndxGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinIdx.nppiMinIndx_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinIdx.nppiMinIndx_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Max
		/// <summary>
		/// Scratch-buffer size for Max.
		/// </summary>
		/// <returns></returns>
		public int MaxGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Max.nppiMaxGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Max(CudaDeviceVariable<byte> max, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Max.nppiMax_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Max(CudaDeviceVariable<byte> max, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Max.nppiMax_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MaxIndex
		/// <summary>
		/// Scratch-buffer size for MaxIndex.
		/// </summary>
		/// <returns></returns>
		public int MaxIndexGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaxIdx.nppiMaxIndxGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndxGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxIndex(CudaDeviceVariable<byte> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MaxIdx.nppiMaxIndx_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxIndex(CudaDeviceVariable<byte> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaxIdx.nppiMaxIndx_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMax
		/// <summary>
		/// Scratch-buffer size for MinMax.
		/// </summary>
		/// <returns></returns>
		public int MinMaxGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MinMaxNew.nppiMinMaxGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMax(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinMaxNew.nppiMinMax_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMax(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinMaxNew.nppiMinMax_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMaxIndex
		/// <summary>
		/// Scratch-buffer size for MinMaxIndex.
		/// </summary>
		/// <returns></returns>
		public int MinMaxIndexGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for MinMaxIndex with mask.
		/// </summary>
		/// <returns></returns>
		public int MinMaxIndexMaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexMaskedGetBufferHostSize()"/></param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Mean
		/// <summary>
		/// Scratch-buffer size for Mean.
		/// </summary>
		/// <returns></returns>
		public int MeanGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMeanGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for Mean with mask.
		/// </summary>
		/// <returns></returns>
		public int MeanMaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMeanGetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(CudaDeviceVariable<double> mean, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(CudaDeviceVariable<double> mean, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(CudaDeviceVariable<double> mean, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanMaskedGetBufferHostSize()"/></param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(CudaDeviceVariable<double> mean, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region Geometric Transforms

        ///// <summary>
        ///// Resizes images.
        ///// </summary>
        ///// <param name="dest">Destination image</param>
        ///// <param name="xFactor">X scaling factor</param>
        ///// <param name="yFactor">Y scaling factor</param>
        ///// <param name="eInterpolation">Interpolation mode</param>
        //public void Resize(NPPImage_8uC1 dest, double xFactor, double yFactor, InterpolationMode eInterpolation)
        //{
        //	status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResize_8u_C1R_Ctx(_devPtr, _sizeOriginal, _pitch, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, xFactor, yFactor, eInterpolation, nppStreamCtx);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_8u_C1R_Ctx", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        /// <summary>
        /// Resizes images.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="eInterpolation">Interpolation mode</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void Resize(NPPImage_8uC1 dest, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResize_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointer, dest.Pitch, dest.Size, new NppiRect(dest.PointRoi, dest.SizeRoi), eInterpolation, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_8u_C1R_Ctx_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Rotate images.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="nAngle">The angle of rotation in degrees.</param>
        /// <param name="nShiftX">Shift along horizontal axis</param>
        /// <param name="nShiftY">Shift along vertical axis</param>
        /// <param name="eInterpolation">Interpolation mode</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void Rotate(NPPImage_8uC1 dest, double nAngle, double nShiftX, double nShiftY, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiRotate_8u_C1R_Ctx(_devPtr, _sizeRoi, _pitch, new NppiRect(_pointRoi, _sizeRoi), 
				dest.DevicePointer, dest.Pitch, new NppiRect(dest.PointRoi, dest.SizeRoi), nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRotate_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Mirror image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mirror(NPPImage_8uC1 dest, NppiAxis flip, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiMirror_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, flip, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Filter
		/// <summary>
		/// Pixels under the mask are multiplied by the respective weights in the mask and the results are summed.<para/>
		/// Before writing the result pixel the sum is scaled back via division by nDivisor.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="aKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Filter(NPPImage_8uC1 dest, CudaDeviceVariable<int> Kernel, NppiSize aKernelSize, NppiPoint oAnchor, int nDivisor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Convolution.nppiFilter_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, aKernelSize, oAnchor, nDivisor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Apply convolution filter with user specified 1D column of weights. Result pixel is equal to the sum of
		/// the products between the kernel coefficients (pKernel array) and corresponding neighboring column pixel
		/// values in the source image defined by nKernelDim and nAnchorY, divided by nDivisor.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nKernelSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterColumn(NPPImage_8uC1 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterColumn_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Apply general linear Row convolution filter, with rescaling, in a 1D mask region around each source pixel. 
		/// Result pixel is equal to the sum of the products between the kernel
		/// coefficients (pKernel array) and corresponding neighboring row pixel values in the source image defined
		/// by iKernelDim and iAnchorX, divided by iDivisor.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nKernelSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRow(NPPImage_8uC1 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterRow_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Apply general linear Row convolution filter, with rescaling, in a 1D mask region around each source pixel with border control. 
		/// Result pixel is equal to the sum of the products between the kernel
		/// coefficients (pKernel array) and corresponding neighboring row pixel values in the source image defined
		/// by iKernelDim and iAnchorX, divided by iDivisor.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nKernelSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRowBorder(NPPImage_8uC1 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterRowBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBox(NPPImage_8uC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFixedFilters2D.nppiFilterBox_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBox_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMin(NPPImage_8uC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMin_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMin_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMax(NPPImage_8uC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMax_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMax_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Morphology
		/// <summary>
		/// Dilation computes the output pixel as the maximum pixel value of the pixels under the mask. Pixels who’s
		/// corresponding mask values are zero to not participate in the maximum search.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Mask">Pointer to the start address of the mask array.</param>
		/// <param name="aMaskSize">Width and Height mask array.</param>
		/// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dilate(NPPImage_8uC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiDilate_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Erosion computes the output pixel as the minimum pixel value of the pixels under the mask. Pixels who’s
		/// corresponding mask values are zero to not participate in the maximum search.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Mask">Pointer to the start address of the mask array.</param>
		/// <param name="aMaskSize">Width and Height mask array.</param>
		/// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Erode(NPPImage_8uC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiErode_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Dilation computes the output pixel as the maximum pixel value of the pixels under the mask. Pixels who’s
		/// corresponding mask values are zero to not participate in the maximum search. With border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Mask">Pointer to the start address of the mask array.</param>
		/// <param name="aMaskSize">Width and Height mask array.</param>
		/// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void DilateBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DilationWithBorderControl.nppiDilateBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilateBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Erosion computes the output pixel as the minimum pixel value of the pixels under the mask. Pixels who’s
		/// corresponding mask values are zero to not participate in the maximum search. With border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Mask">Pointer to the start address of the mask array.</param>
		/// <param name="aMaskSize">Width and Height mask array.</param>
		/// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ErodeBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ErosionWithBorderControl.nppiErodeBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErodeBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 dilation with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dilate3x3Border(NPPImage_8uC1 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Dilate3x3Border.nppiDilate3x3Border_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3Border_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Erode3x3Border(NPPImage_8uC1 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Erode3x3Border.nppiErode3x3Border_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3Border_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Logical
		/// <summary>
		/// image bit shift by constant (left).
		/// </summary>
		/// <param name="nConstant">Constant</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LShiftC(uint nConstant, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LeftShiftConst.nppiLShiftC_8u_C1R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (left), inplace.
		/// </summary>
		/// <param name="nConstant">Constant</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LShiftC(uint nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LeftShiftConst.nppiLShiftC_8u_C1IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (right).
		/// </summary>
		/// <param name="nConstant">Constant</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void RShiftC(uint nConstant, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RightShiftConst.nppiRShiftC_8u_C1R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (right), inplace.
		/// </summary>
		/// <param name="nConstant">Constant</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void RShiftC(uint nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RightShiftConst.nppiRShiftC_8u_C1IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical and.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void And(NPPImage_8uC1 src2, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.And.nppiAnd_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical and.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void And(NPPImage_8uC1 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.And.nppiAnd_8u_C1IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical and with constant.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void And(byte nConstant, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AndConst.nppiAndC_8u_C1R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical and with constant.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void And(byte nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AndConst.nppiAndC_8u_C1IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Or.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Or(NPPImage_8uC1 src2, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Or.nppiOr_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Or.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Or(NPPImage_8uC1 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Or.nppiOr_8u_C1IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Or with constant.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Or(byte nConstant, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.OrConst.nppiOrC_8u_C1R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Or with constant.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Or(byte nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.OrConst.nppiOrC_8u_C1IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Xor.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Xor(NPPImage_8uC1 src2, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Xor.nppiXor_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Xor.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Xor(NPPImage_8uC1 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Xor.nppiXor_8u_C1IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Xor with constant.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Xor(byte nConstant, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.XorConst.nppiXorC_8u_C1R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Xor with constant.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Xor(byte nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.XorConst.nppiXorC_8u_C1IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Image logical Not.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Not(NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Not.nppiNot_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNot_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Not.
		/// </summary>
		public void Not(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Not.nppiNot_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNot_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sub
		/// <summary>
		/// Image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(NPPImage_8uC1 src2, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(NPPImage_8uC1 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_8u_C1IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(byte nConstant, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(byte nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_8u_C1IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Mul
		/// <summary>
		/// Image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_8uC1 src2, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_8uC1 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_8u_C1IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(byte nConstant, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(byte nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_8u_C1IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image multiplication and scale by max bit width value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_8uC1 src2, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulScale.nppiMulScale_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulScale_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication and scale by max bit width value
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_8uC1 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulScale.nppiMulScale_8u_C1IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulScale_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image and scale by max bit width value
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(byte nConstant, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConstScale.nppiMulCScale_8u_C1R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulCScale_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image and scale by max bit width value
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(byte nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConstScale.nppiMulCScale_8u_C1IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulCScale_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Div
		/// <summary>
		/// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_8uC1 src2, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_8uC1 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_8u_C1IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(byte nConstant, NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(byte nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_8u_C1IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_8uC1 src2, NPPImage_8uC1 dest, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivRound.nppiDiv_Round_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, rndMode, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_8uC1 src2, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivRound.nppiDiv_Round_8u_C1IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, rndMode, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region AbsDiff
		/// <summary>
		/// Absolute difference of this minus src2.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AbsDiff(NPPImage_8uC1 src2, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AbsDiff.nppiAbsDiff_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbsDiff_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Absolute difference with constant.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AbsDiff(byte nConstant, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AbsDiffConst.nppiAbsDiffC_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nConstant, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbsDiffC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Exp
		/// <summary>
		/// Exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Exp(NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Exp.nppiExp_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Exp(int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Exp.nppiExp_8u_C1IRSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Ln
		/// <summary>
		/// Natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Ln(NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Ln.nppiLn_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Ln(int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Ln.nppiLn_8u_C1IRSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqr
		/// <summary>
		/// Image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqr(NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqr.nppiSqr_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqr(int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqr.nppiSqr_8u_C1IRSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqrt
		/// <summary>
		/// Image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqrt(NPPImage_8uC1 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqrt.nppiSqrt_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqrt(int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqrt.nppiSqrt_8u_C1IRSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_8u_C1IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Alpha Composition

		/// <summary>
		/// Image composition using constant alpha.
		/// </summary>
		/// <param name="alpha1">constant alpha for this image</param>
		/// <param name="src2">2nd source image</param>
		/// <param name="alpha2">constant alpha for src2</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppAlphaOp">alpha compositing operation</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AlphaComp(byte alpha1, NPPImage_8uC1 src2, byte alpha2, NPPImage_8uC1 dest, NppiAlphaOp nppAlphaOp, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AlphaCompConst.nppiAlphaCompC_8u_C1R_Ctx(_devPtrRoi, _pitch, alpha1, src2.DevicePointerRoi, src2.Pitch, alpha2, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaCompC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image premultiplication using constant alpha.
		/// </summary>
		/// <param name="alpha">alpha</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AlphaPremul(byte alpha, NPPImage_8uC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AlphaPremulConst.nppiAlphaPremulC_8u_C1R_Ctx(_devPtrRoi, _pitch, alpha, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaPremulC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// In place alpha premultiplication using constant alpha.
		/// </summary>
		/// <param name="alpha">alpha</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AlphaPremul(byte alpha, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AlphaPremulConst.nppiAlphaPremulC_8u_C1IR_Ctx(alpha, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaPremulC_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region LUT
		/// <summary>
		/// look-up-table color conversion.<para/>
		/// The LUT is derived from a set of user defined mapping points through linear interpolation.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="values">array of user defined OUTPUT values</param>
		/// <param name="levels">array of user defined INPUT values</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTLinear(NPPImage_8uC1 dest, CudaDeviceVariable<int> values, CudaDeviceVariable<int> levels, NppStreamContext nppStreamCtx)
		{
			if (values.Size != levels.Size) throw new ArgumentException("values and levels must have same size.");

			status = NPPNativeMethods_Ctx.NPPi.ColorProcessing.nppiLUT_Linear_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, values.DevicePointer, levels.DevicePointer, (int)levels.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//#region NormDiffOld
		///// <summary>
		///// Infinity Norm of pixel differences.
		///// </summary>
		///// <param name="src2">2nd source image</param>
		///// <param name="retVal">Contains computed Inf-norm of differences.</param>
		//public void NormDiffInf(NPPImage_8uC1 src2, CudaDeviceVariable<double> retVal)
		//{
		//    //status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, retVal.DevicePointer, nppStreamCtx);
		//    //Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C1R_Ctx", status));
		//    //NPPException.CheckNppStatus(status, this);
		//}

		///// <summary>
		///// L1 Norm of pixel differences.
		///// </summary>
		///// <param name="src2">2nd source image</param>
		///// <param name="retVal">Contains computed L1-norm of differences.</param>
		//public void NormDiffL1(NPPImage_8uC1 src2, CudaDeviceVariable<double> retVal)
		//{
		//    //status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, retVal.DevicePointer, nppStreamCtx);
		//    //Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C1R_Ctx", status));
		//    //NPPException.CheckNppStatus(status, this);
		//}

		///// <summary>
		///// L2 Norm of pixel differences.
		///// </summary>
		///// <param name="src2">2nd source image</param>
		///// <param name="retVal">Contains computed L2-norm of differences.</param>
		//public void NormDiffL2(NPPImage_8uC1 src2, CudaDeviceVariable<double> retVal)
		//{
		//    //status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, retVal.DevicePointer, nppStreamCtx);
		//    //Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C1R_Ctx", status));
		//    //NPPException.CheckNppStatus(status, this);
		//}
		//#endregion

		#region Affine Transformations
		/// <summary>
		/// Calculates affine transform coefficients given source rectangular ROI and its destination quadrangle projection
		/// </summary>
		/// <param name="quad">Destination quadrangle [4,2]</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// <returns>Affine transform coefficients [2,3]</returns>
		public double[,] GetAffineTransform(double[,] quad, NppStreamContext nppStreamCtx)
		{
			double[,] coeffs = new double[2, 3];
			NppiRect rect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiGetAffineTransform(rect, quad, coeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetAffineTransform", status));
			NPPException.CheckNppStatus(status, this);
			return coeffs;
		}

		/// <summary>
		/// Affine transform of an image. <para/>This
		/// function operates using given transform coefficients that can be obtained
		/// by using nppiGetAffineTransform function or set explicitly. The function
		/// operates on source and destination regions of interest. The affine warp
		/// function transforms the source image pixel coordinates (x,y) according
		/// to the following formulas:<para/>
		/// X_new = C_00 * x + C_01 * y + C_02<para/>
		/// Y_new = C_10 * x + C_11 * y + C_12<para/>
		/// The transformed part of the source image is resampled using the specified
		/// interpolation method and written to the destination ROI.
		/// The functions nppiGetAffineQuad and nppiGetAffineBound can help with 
		/// destination ROI specification.<para/>
		/// <para/>
		/// NPPI specific recommendation: <para/>
		/// The function operates using 2 types of kernels: fast and accurate. The fast
		/// method is about 4 times faster than its accurate variant,
		/// but does not perform memory access checks and requires the destination ROI
		/// to be 64 bytes aligned. Hence any destination ROI is 
		/// chunked into 3 vertical stripes: the first and the third are processed by
		/// accurate kernels and the central one is processed by the
		/// fast one.<para/>
		/// In order to get the maximum available speed of execution, the projection of
		/// destination ROI onto image addresses must be 64 bytes
		/// aligned. This is always true if the values <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x))</code> and <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</code> <para/>
		/// are multiples of 64. Another rule of thumb is to specify destination ROI in
		/// such way that left and right sides of the projected 
		/// image are separated from the ROI by at least 63 bytes from each side.
		/// However, this requires the whole ROI to be part of allocated memory. In case
		/// when the conditions above are not satisfied, the function may decrease in
		/// speed slightly and will return NPP_MISALIGNED_DST_ROI_WARNING warning.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="coeffs">Affine transform coefficients [2,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void WarpAffine(NPPImage_8uC1 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffine_8u_C1R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffine_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inverse affine transform of an image.<para/>
		/// This function operates using given transform coefficients that can be
		/// obtained by using nppiGetAffineTransform function or set explicitly. Thus
		/// there is no need to invert coefficients in your application before calling
		/// WarpAffineBack. The function operates on source and destination regions of
		/// interest.<para/>
		/// The affine warp function transforms the source image pixel coordinates
		/// (x,y) according to the following formulas:<para/>
		/// X_new = C_00 * x + C_01 * y + C_02<para/>
		/// Y_new = C_10 * x + C_11 * y + C_12<para/>
		/// The transformed part of the source image is resampled using the specified
		/// interpolation method and written to the destination ROI.
		/// The functions nppiGetAffineQuad and nppiGetAffineBound can help with
		/// destination ROI specification.<para/><para/>
		/// NPPI specific recommendation: <para/>
		/// The function operates using 2 types of kernels: fast and accurate. The fast
		/// method is about 4 times faster than its accurate variant,
		/// but doesn't perform memory access checks and requires the destination ROI
		/// to be 64 bytes aligned. Hence any destination ROI is 
		/// chunked into 3 vertical stripes: the first and the third are processed by
		/// accurate kernels and the central one is processed by the fast one.
		/// In order to get the maximum available speed of execution, the projection of
		/// destination ROI onto image addresses must be 64 bytes aligned. This is
		/// always true if the values <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x))</code> and <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</code> <para/>
		/// are multiples of 64. Another rule of thumb is to specify destination ROI in
		/// such way that left and right sides of the projected image are separated from
		/// the ROI by at least 63 bytes from each side. However, this requires the
		/// whole ROI to be part of allocated memory. In case when the conditions above
		/// are not satisfied, the function may decrease in speed slightly and will
		/// return NPP_MISALIGNED_DST_ROI_WARNING warning.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="coeffs">Affine transform coefficients [2,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void WarpAffineBack(NPPImage_8uC1 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffineBack_8u_C1R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBack_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Affine transform of an image. <para/>This
		/// function performs affine warping of a the specified quadrangle in the
		/// source image to the specified quadrangle in the destination image. The
		/// function nppiWarpAffineQuad uses the same formulas for pixel mapping as in
		/// nppiWarpAffine function. The transform coefficients are computed internally.
		/// The transformed part of the source image is resampled using the specified
		/// eInterpolation method and written to the destination ROI.<para/>
		/// NPPI specific recommendation: <para/>
		/// The function operates using 2 types of kernels: fast and accurate. The fast
		/// method is about 4 times faster than its accurate variant,
		/// but doesn't perform memory access checks and requires the destination ROI
		/// to be 64 bytes aligned. Hence any destination ROI is 
		/// chunked into 3 vertical stripes: the first and the third are processed by
		/// accurate kernels and the central one is processed by the fast one.
		/// In order to get the maximum available speed of execution, the projection of
		/// destination ROI onto image addresses must be 64 bytes aligned. This is
		/// always true if the values <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x))</code> and <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</code> <para/>
		/// are multiples of 64. Another rule of thumb is to specify destination ROI in
		/// such way that left and right sides of the projected image are separated from
		/// the ROI by at least 63 bytes from each side. However, this requires the
		/// whole ROI to be part of allocated memory. In case when the conditions above
		/// are not satisfied, the function may decrease in speed slightly and will
		/// return NPP_MISALIGNED_DST_ROI_WARNING warning.
		/// </summary>
		/// <param name="srcQuad">Source quadrangle [4,2]</param>
		/// <param name="dest">Destination image</param>
		/// <param name="dstQuad">Destination quadrangle [4,2]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void WarpAffineQuad(double[,] srcQuad, NPPImage_8uC1 dest, double[,] dstQuad, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffineQuad_8u_C1R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, dstQuad, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineQuad_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region Perspective Transformations

		/// <summary>
		/// Perspective transform of an image.<para/>
		/// This function operates using given transform coefficients that 
		/// can be obtained by using nppiGetPerspectiveTransform function or set
		/// explicitly. The function operates on source and destination regions 
		/// of interest. The perspective warp function transforms the source image
		/// pixel coordinates (x,y) according to the following formulas:<para/>
		/// X_new = (C_00 * x + C_01 * y + C_02) / (C_20 * x + C_21 * y + C_22)<para/>
		/// Y_new = (C_10 * x + C_11 * y + C_12) / (C_20 * x + C_21 * y + C_22)<para/>
		/// The transformed part of the source image is resampled using the specified
		/// interpolation method and written to the destination ROI.
		/// The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
		/// with destination ROI specification.<para/>
		/// NPPI specific recommendation: <para/>
		/// The function operates using 2 types of kernels: fast and accurate. The fast
		/// method is about 4 times faster than its accurate variant,
		/// but doesn't perform memory access checks and requires the destination ROI
		/// to be 64 bytes aligned. Hence any destination ROI is 
		/// chunked into 3 vertical stripes: the first and the third are processed by
		/// accurate kernels and the central one is processed by the fast one.
		/// In order to get the maximum available speed of execution, the projection of
		/// destination ROI onto image addresses must be 64 bytes aligned. This is
		/// always true if the values <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x))</code> and <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</code> <para/>
		/// are multiples of 64. Another rule of thumb is to specify destination ROI in
		/// such way that left and right sides of the projected image are separated from
		/// the ROI by at least 63 bytes from each side. However, this requires the
		/// whole ROI to be part of allocated memory. In case when the conditions above
		/// are not satisfied, the function may decrease in speed slightly and will
		/// return NPP_MISALIGNED_DST_ROI_WARNING warning.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="coeffs">Perspective transform coefficients [3,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void WarpPerspective(NPPImage_8uC1 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspective_8u_C1R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspective_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inverse perspective transform of an image. <para/>
		/// This function operates using given transform coefficients that 
		/// can be obtained by using nppiGetPerspectiveTransform function or set
		/// explicitly. Thus there is no need to invert coefficients in your application 
		/// before calling WarpPerspectiveBack. The function operates on source and
		/// destination regions of interest. The perspective warp function transforms the source image
		/// pixel coordinates (x,y) according to the following formulas:<para/>
		/// X_new = (C_00 * x + C_01 * y + C_02) / (C_20 * x + C_21 * y + C_22)<para/>
		/// Y_new = (C_10 * x + C_11 * y + C_12) / (C_20 * x + C_21 * y + C_22)<para/>
		/// The transformed part of the source image is resampled using the specified
		/// interpolation method and written to the destination ROI.
		/// The functions nppiGetPerspectiveQuad and nppiGetPerspectiveBound can help
		/// with destination ROI specification.<para/>
		/// NPPI specific recommendation: <para/>
		/// The function operates using 2 types of kernels: fast and accurate. The fast
		/// method is about 4 times faster than its accurate variant,
		/// but doesn't perform memory access checks and requires the destination ROI
		/// to be 64 bytes aligned. Hence any destination ROI is 
		/// chunked into 3 vertical stripes: the first and the third are processed by
		/// accurate kernels and the central one is processed by the fast one.
		/// In order to get the maximum available speed of execution, the projection of
		/// destination ROI onto image addresses must be 64 bytes aligned. This is
		/// always true if the values <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x))</code> and <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</code> <para/>
		/// are multiples of 64. Another rule of thumb is to specify destination ROI in
		/// such way that left and right sides of the projected image are separated from
		/// the ROI by at least 63 bytes from each side. However, this requires the
		/// whole ROI to be part of allocated memory. In case when the conditions above
		/// are not satisfied, the function may decrease in speed slightly and will
		/// return NPP_MISALIGNED_DST_ROI_WARNING warning.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="coeffs">Perspective transform coefficients [3,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void WarpPerspectiveBack(NPPImage_8uC1 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspectiveBack_8u_C1R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBack_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Perspective transform of an image.<para/>
		/// This function performs perspective warping of a the specified
		/// quadrangle in the source image to the specified quadrangle in the
		/// destination image. The function nppiWarpPerspectiveQuad uses the same
		/// formulas for pixel mapping as in nppiWarpPerspective function. The
		/// transform coefficients are computed internally.
		/// The transformed part of the source image is resampled using the specified
		/// interpolation method and written to the destination ROI.<para/>
		/// NPPI specific recommendation: <para/>
		/// The function operates using 2 types of kernels: fast and accurate. The fast
		/// method is about 4 times faster than its accurate variant,
		/// but doesn't perform memory access checks and requires the destination ROI
		/// to be 64 bytes aligned. Hence any destination ROI is 
		/// chunked into 3 vertical stripes: the first and the third are processed by
		/// accurate kernels and the central one is processed by the fast one.
		/// In order to get the maximum available speed of execution, the projection of
		/// destination ROI onto image addresses must be 64 bytes aligned. This is
		/// always true if the values <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x))</code> and <para/>
		/// <code>(int)((void *)(pDst + dstRoi.x + dstRoi.width))</code> <para/>
		/// are multiples of 64. Another rule of thumb is to specify destination ROI in
		/// such way that left and right sides of the projected image are separated from
		/// the ROI by at least 63 bytes from each side. However, this requires the
		/// whole ROI to be part of allocated memory. In case when the conditions above
		/// are not satisfied, the function may decrease in speed slightly and will
		/// return NPP_MISALIGNED_DST_ROI_WARNING warning.
		/// </summary>
		/// <param name="srcQuad">Source quadrangle [4,2]</param>
		/// <param name="dest">Destination image</param>
		/// <param name="destQuad">Destination quadrangle [4,2]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void WarpPerspectiveQuad(double[,] srcQuad, NPPImage_8uC1 dest, double[,] destQuad, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspectiveQuad_8u_C1R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, destQuad, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveQuad_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region SqrIntegral
		//
		//nppiSqrIntegral_8u32s32f_C1R is not implemented in nppi but is listed in the nppi header file
		//
		///// <summary>
		///// Transforms an image to integral and integral of pixel squares representation.
		///// </summary>
		///// <param name="dst">Destination integral image</param>
		///// <param name="sqr">Destination integral image of pixel squares</param>
		///// <param name="val">The value to add to dst image pixels</param>
		///// <param name="valSqr">The value to add to pSqr image pixels</param>
		///// <param name="integralImageNewHeight">Extended height of output surfaces (needed by transpose in primitive)</param>
		//public void SqrIntegral(NPPImage_32sC1 dst, NPPImage_32fC1 sqr, int val, float valSqr, int integralImageNewHeight)
		//{
		//    status = NPPNativeMethods_Ctx.NPPi.unsorted.nppiSqrIntegral_8u32s32f_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, sqr.DevicePointerRoi, sqr.Pitch, _sizeRoi, val, valSqr, integralImageNewHeight, nppStreamCtx);
		//    Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrIntegral_8u32s32f_C1R_Ctx", status);
		//    NPPException.CheckNppStatus(status, this);
		//}

		#endregion

		#region AddProduct
		/// <summary>
		/// One 8-bit unsigned char channel image product added to in place floating point destination image.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AddProduct(NPPImage_8uC1 src2, NPPImage_32fC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddProduct.nppiAddProduct_8u32f_C1IR_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddProduct_8u32f_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One 8-bit unsigned char channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="mask">Mask image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AddProduct(NPPImage_8uC1 src2, NPPImage_32fC1 dest, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddProduct.nppiAddProduct_8u32f_C1IMR_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, mask.DevicePointerRoi, mask.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddProduct_8u32f_C1IMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region AddSquare
		/// <summary>
		/// One 8-bit unsigned char channel image squared then added to in place floating point destination image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AddProduct(NPPImage_32fC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddSquare.nppiAddSquare_8u32f_C1IR_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddSquare_8u32f_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One 8-bit unsigned char channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="mask">Mask image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AddProduct(NPPImage_32fC1 dest, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddSquare.nppiAddSquare_8u32f_C1IMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddSquare_8u32f_C1IMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region AddWeighted
		/// <summary>
		/// One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AddWeighted(NPPImage_32fC1 dest, float nAlpha, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddWeighted.nppiAddWeighted_8u32f_C1IR_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAlpha, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddWeighted_8u32f_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="mask">Mask image</param>
		/// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AddWeighted(NPPImage_32fC1 dest, NPPImage_8uC1 mask, float nAlpha, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddWeighted.nppiAddWeighted_8u32f_C1IMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAlpha, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddWeighted_8u32f_C1IMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region SumWindow
		/// <summary>
		/// 8-bit unsigned 1D (column) sum to 32f.
		/// Apply Column Window Summation filter over a 1D mask region around each
		/// source pixel for 1-channel 8 bit/pixel input images with 32-bit floating point
		/// output.  <para/>
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring column pixel values in a mask region of the source image defined by
		/// nMaskSize and nAnchor. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SumWindowColumn(NPPImage_32fC1 dest, int nMaskSize, int nAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.WindowSum1D.nppiSumWindowColumn_8u32f_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowColumn_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned 1D (row) sum to 32f.<para/>
		/// Apply Row Window Summation filter over a 1D mask region around each source
		/// pixel for 1-channel 8-bit pixel input images with 32-bit floating point output.  
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring row pixel values in a mask region of the source image defined
		/// by nKernelDim and nAnchorX. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SumWindowRow(NPPImage_32fC1 dest, int nMaskSize, int nAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.WindowSum1D.nppiSumWindowRow_8u32f_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowRow_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region CompColorKey
		/// <summary>
		/// packed color complement color key replacement of source image 1 by source image 2
		/// </summary>
		/// <param name="src2">source2 packed pixel format image.</param>
		/// <param name="dest">Destination image</param>
		/// <param name="colorKeyConst">color key constant</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CompColorKey(NPPImage_8uC1 src2, NPPImage_8uC1 dest, byte colorKeyConst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CompColorKey.nppiCompColorKey_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, colorKeyConst, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompColorKey_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region NormInf
		/// <summary>
		/// Scratch-buffer size for Norm inf.
		/// </summary>
		/// <returns></returns>
		public int NormInfGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNormInfGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for Norm inf (masked).
		/// </summary>
		/// <returns></returns>
		public int NormInfMaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNormInfGetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(CudaDeviceVariable<double> norm, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfMaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region NormL1
		/// <summary>
		/// Scratch-buffer size for Norm L1.
		/// </summary>
		/// <returns></returns>
		public int NormL1GetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNormL1GetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for Norm L1 (masked).
		/// </summary>
		/// <returns></returns>
		public int NormL1MaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNormL1GetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(CudaDeviceVariable<double> norm, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region NormL2
		/// <summary>
		/// Scratch-buffer size for Norm L2.
		/// </summary>
		/// <returns></returns>
		public int NormL2GetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNormL2GetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for Norm L2 (masked).
		/// </summary>
		/// <returns></returns>
		public int NormL2MaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNormL2GetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(CudaDeviceVariable<double> norm, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_8u_C1MR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Threshold
		/// <summary>
		/// Image threshold.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Threshold(NPPImage_8uC1 dest, byte nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Threshold(byte nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdGT
		/// <summary>
		/// Image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdGT(NPPImage_8uC1 dest, byte nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GT_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdGT(byte nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GT_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdLT
		/// <summary>
		/// Image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdLT(NPPImage_8uC1 dest, byte nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LT_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdLT(byte nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LT_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdVal
		/// <summary>
		/// Image threshold.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Threshold(NPPImage_8uC1 dest, byte nThreshold, byte nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_Val_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Threshold(byte nThreshold, byte nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_Val_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdGTVal
		/// <summary>
		/// Image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdGT(NPPImage_8uC1 dest, byte nThreshold, byte nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GTVal_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdGT(byte nThreshold, byte nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GTVal_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdLTVal
		/// <summary>
		/// Image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdLT(NPPImage_8uC1 dest, byte nThreshold, byte nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTVal_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdLT(byte nThreshold, byte nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTVal_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdLTValGTVal
		/// <summary>
		/// Image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
		/// to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThresholdLT">The thresholdLT value.</param>
		/// <param name="nValueLT">The thresholdLT replacement value.</param>
		/// <param name="nThresholdGT">The thresholdGT value.</param>
		/// <param name="nValueGT">The thresholdGT replacement value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdLTGT(NPPImage_8uC1 dest, byte nThresholdLT, byte nValueLT, byte nThresholdGT, byte nValueGT, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTValGTVal_8u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
		/// to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThresholdLT">The thresholdLT value.</param>
		/// <param name="nValueLT">The thresholdLT replacement value.</param>
		/// <param name="nThresholdGT">The thresholdGT value.</param>
		/// <param name="nValueGT">The thresholdGT replacement value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdLTGT(byte nThresholdLT, byte nValueLT, byte nThresholdGT, byte nValueGT, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTValGTVal_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Compare
		/// <summary>
		/// Compare pSrc1's pixels with corresponding pixels in pSrc2.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Compare(NPPImage_8uC1 src2, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Compare.nppiCompare_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompare_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc's pixels with constant value.
		/// </summary>
		/// <param name="nConstant">constant value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Compare(byte nConstant, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Compare.nppiCompareC_8u_C1R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareC_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//new in Cuda 5.5:
		#region DotProduct
		/// <summary>
		/// Device scratch buffer size (in bytes) for nppiDotProd_8u64f_C1R.
		/// </summary>
		/// <returns></returns>
		public int DotProdGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.DotProd.nppiDotProdGetBufferHostSize_8u64f_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProdGetBufferHostSize_8u64f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// One-channel 8-bit unsigned image DotProd.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="DotProdGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void DotProduct(NPPImage_8uC1 src2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = DotProdGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.DotProd.nppiDotProd_8u64f_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_8u64f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One-channel 8-bit unsigned image DotProd. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (1 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void DotProduct(NPPImage_8uC1 src2, CudaDeviceVariable<double> pDp, NppStreamContext nppStreamCtx)
		{
			int bufferSize = DotProdGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.DotProd.nppiDotProd_8u64f_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_8u64f_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region CopyNew

		/// <summary>
		/// image copy with nearest source image pixel color.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nTopBorderHeight">Height (in pixels) of the top border. The height of the border at the bottom of
		/// the destination ROI is implicitly defined by the size of the source ROI: nBottomBorderHeight =
		/// oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
		/// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		public void CopyReplicateBorder(NPPImage_8uC1 dst, int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image copy with the borders wrapped by replication of source image pixel colors.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nTopBorderHeight">Height (in pixels) of the top border. The height of the border at the bottom of
		/// the destination ROI is implicitly defined by the size of the source ROI: nBottomBorderHeight =
		/// oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
		/// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		public void CopyWrapBorder(NPPImage_8uC1 dst, int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CopyWrapBorder.nppiCopyWrapBorder_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// linearly interpolated source image subpixel coordinate color copy.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nDx">Fractional part of source image X coordinate.</param>
		/// <param name="nDy">Fractional part of source image Y coordinate.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CopySubpix(NPPImage_8uC1 dst, float nDx, float nDy, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CopySubpix.nppiCopySubpix_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region LUTNew
		/// <summary>
		/// look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUT(NPPImage_8uC1 dst, CudaDeviceVariable<int> pValues, CudaDeviceVariable<int> pLevels, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUT.nppiLUT_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTCubic(NPPImage_8uC1 dst, CudaDeviceVariable<int> pValues, CudaDeviceVariable<int> pLevels, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTCubic.nppiLUT_Cubic_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// range restricted palette look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points in a palette and 
		/// source pixels are then processed using a restricted bit range when looking up palette values.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pTable">Pointer to an array of user defined OUTPUT palette values</param>
		/// <param name="nBitSize">Number of least significant bits (must be &gt; 0 and &lt;= 8) of each source pixel value to use as index into palette table during conversion.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTPalette(NPPImage_8uC1 dst, CudaDeviceVariable<byte> pTable, int nBitSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTPalette.nppiLUTPalette_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pTable.DevicePointer, nBitSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUTPalette_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// One channel 8-bit unsigned bit range restricted 24-bit palette look-up-table color conversion with 24-bit destination output per pixel.
		/// The LUT is derived from a set of user defined mapping points in a palette and 
		/// source pixels are then processed using a restricted bit range when looking up palette values.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (3 bytes per pixel)</param>
		/// <param name="nBitSize">Number of least significant bits (must be &gt; 0 and &lt;= 8) of each source pixel value to use as index into palette table during conversion.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTPalette(NPPImage_8uC3 dst, CudaDeviceVariable<byte> pTable, int nBitSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTPalette.nppiLUTPalette_8u24u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pTable.DevicePointer, nBitSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUTPalette_8u24u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// One channel 8-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit destination output per pixel.
		/// The LUT is derived from a set of user defined mapping points in a palette and 
		/// source pixels are then processed using a restricted bit range when looking up palette values.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (4 bytes per pixel)</param>
		/// <param name="nBitSize">Number of least significant bits (must be &gt; 0 and &lt;= 8) of each source pixel value to use as index into palette table during conversion.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTPalette(NPPImage_8uC4 dst, CudaDeviceVariable<byte> pTable, int nBitSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTPalette.nppiLUTPalette_8u32u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pTable.DevicePointer, nBitSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUTPalette_8u32u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// One channel 8-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit destination output per pixel.
		/// The LUT is derived from a set of user defined mapping points in a palette and 
		/// source pixels are then processed using a restricted bit range when looking up palette values.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (4 bytes per pixel)</param>
		/// <param name="nBitSize">Number of least significant bits (must be &gt; 0 and &lt;= 8) of each source pixel value to use as index into palette table during conversion.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTPalette(NPPImage_32uC1 dst, CudaDeviceVariable<uint> pTable, int nBitSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTPalette.nppiLUTPalette_8u32u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pTable.DevicePointer, nBitSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUTPalette_8u32u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		
		/// <summary>
		/// Inplace look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUT(CudaDeviceVariable<int> pValues, CudaDeviceVariable<int> pLevels, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUT.nppiLUT_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTCubic(CudaDeviceVariable<int> pValues, CudaDeviceVariable<int> pLevels, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTCubic.nppiLUT_Cubic_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace linear interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTLinear(CudaDeviceVariable<int> pValues, CudaDeviceVariable<int> pLevels, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTLinear.nppiLUT_Linear_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
				

		#endregion

		#region MorphologicalNew
		/// <summary>
		/// 3x3 dilation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dilate3x3(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiDilate3x3_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Erode3x3(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiErode3x3_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterNew
		/// <summary>
		/// 1D column convolution.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterColumn(NPPImage_8uC1 dst, CudaDeviceVariable<float> pKernel, int nAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterColumn32f_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 1D row convolution.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRow(NPPImage_8uC1 dst, CudaDeviceVariable<float> pKernel, int nAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterRow32f_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// convolution filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array.<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Filter(NPPImage_8uC1 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Convolution.nppiFilter32f_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// convolution filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array.<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Filter(NPPImage_16sC1 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Convolution.nppiFilter32f_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterPrewittHoriz(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittHoriz_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHoriz_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterPrewittVert(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittVert_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVert_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SobelHoriz(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelHoriz_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHoriz_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVert(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelVert_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVert_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelHoriz(NPPImage_16sC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelHoriz_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHoriz_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVert(NPPImage_16sC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelVert_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVert_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// second derivative, horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelHorizSecond(NPPImage_16sC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelHorizSecond_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizSecond_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// second derivative, vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVertSecond(NPPImage_16sC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelVertSecond_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertSecond_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// second cross derivative Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelCross(NPPImage_16sC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelCross_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelCross_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsDown(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsDown_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDown_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter..
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsUp(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsUp_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUp_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLaplace(NPPImage_8uC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLaplace_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplace_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Gauss filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGauss(NPPImage_8uC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterGauss_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGauss_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterHighPass(NPPImage_8uC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterHighPass_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPass_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLowPass(NPPImage_8uC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLowPass_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPass_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSharpen(NPPImage_8uC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSharpen_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpen_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Scharr filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterScharrHoriz(NPPImage_16sC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterScharrHoriz_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrHoriz_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Scharr filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterScharrVert(NPPImage_16sC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterScharrVert_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrVert_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLaplace(NPPImage_16sC1 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLaplace_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplace_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region NormNew
		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormDiffInfGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L1.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL1GetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L2.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL2GetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormRelInfGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L1.
		/// </summary>
		/// <returns></returns>
		public int NormRelL1GetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L2.
		/// </summary>
		/// <returns></returns>
		public int NormRelL2GetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_8u_C1R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}





		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrFull_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int FullNormLevelGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_8u32f_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrFull_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="FullNormLevelGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_NormLevel(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FullNormLevelGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_NormLevel(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FullNormLevelGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u32f_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrSame_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int SameNormLevelGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_8u32f_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrSame_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SameNormLevelGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_NormLevel(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SameNormLevelGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_NormLevel(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SameNormLevelGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u32f_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrValid_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int ValidNormLevelGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_8u32f_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrValid_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="ValidNormLevelGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_NormLevel(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_NormLevel(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u32f_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}






		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrFull_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int FullNormLevelScaledGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_8u_C1RSfs_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrFull_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="FullNormLevelScaledGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_NormLevel(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FullNormLevelScaledGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_NormLevel(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FullNormLevelScaledGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u_C1RSfs_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrSame_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int SameNormLevelScaledGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_8u_C1RSfs_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrSame_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SameNormLevelScaledGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_NormLevel(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SameNormLevelScaledGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_NormLevel(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SameNormLevelScaledGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u_C1RSfs_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrValid_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int ValidNormLevelScaledGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_8u_C1RSfs_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrValid_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="ValidNormLevelScaledGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_NormLevel(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ValidNormLevelScaledGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_NormLevel(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ValidNormLevelScaledGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u_C1RSfs_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}






		/// <summary>
		/// image SqrDistanceFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceFull_Norm(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceFull_Norm_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceFull_Norm_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceSame_Norm(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceSame_Norm_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceSame_Norm_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceValid_Norm(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceValid_Norm_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceValid_Norm_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// image SqrDistanceFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceFull_Norm(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceFull_Norm_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceFull_Norm_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceSame_Norm(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceSame_Norm_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceSame_Norm_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceValid_Norm(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// image CrossCorrFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_Norm(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_Norm_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_Norm(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_Norm_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_Norm(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_Norm_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid(NPPImage_8uC1 tpl, NPPImage_32fC1 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_8u32f_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// image CrossCorrFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_Norm(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_Norm_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_Norm(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_Norm_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_Norm(NPPImage_8uC1 tpl, NPPImage_8uC1 dst, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region NormMaskedNew
		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormDiffInfMaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfMaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L1.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL1MaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L2.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL2MaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormRelInfMaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfMaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L1.
		/// </summary>
		/// <returns></returns>
		public int NormRelL1MaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L2.
		/// </summary>
		/// <returns></returns>
		public int NormRelL2MaskedGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_8u_C1MR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C1MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_8uC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_8u_C1MR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C1MR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}






		#endregion

		#region IntegralNew
		/// <summary>
		/// Integral with 32-bit floating point output.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nVal">The value to add to pDst image pixels</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Integral(NPPImage_32fC1 dst, float nVal, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Integral.nppiIntegral_8u32f_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nVal, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiIntegral_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Integral with 32-bit signed output.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nVal">The value to add to pDst image pixels</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Integral(NPPImage_32sC1 dst, int nVal, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Integral.nppiIntegral_8u32s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nVal, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiIntegral_8u32s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One-channel 8-bit unsigned image SqrIntegral. 
		/// Destination integral image is 32-bit floating point.
		/// Destination square integral image is 64-bit double floating point.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="sqr">Destination-Image</param>
		/// <param name="nVal">The value to add to pDst image pixels</param>
		/// <param name="nValSqr">The value to add to pSqr image pixels</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Integral(NPPImage_32fC1 dst, CudaPitchedDeviceVariable<double> sqr, float nVal, double nValSqr, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Integral.nppiSqrIntegral_8u32f64f_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, sqr.DevicePointer, sqr.Pitch, _sizeRoi, nVal, nValSqr, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrIntegral_8u32f64f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One-channel 8-bit unsigned image SqrIntegral. 
		/// Destination integral image and square integral image are 32-bit signed int.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="sqr">Destination-Image</param>
		/// <param name="nVal">The value to add to pDst image pixels</param>
		/// <param name="nValSqr">The value to add to pSqr image pixels</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Integral(NPPImage_32sC1 dst, NPPImage_32sC1 sqr, int nVal, int nValSqr, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Integral.nppiSqrIntegral_8u32s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, sqr.DevicePointerRoi, sqr.Pitch, _sizeRoi, nVal, nValSqr, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrIntegral_8u32s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One-channel 8-bit unsigned image SqrIntegral. 
		/// Destination integral image is 32-bit signed int.
		/// Destination square integral image is 64-bit double floating point.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="sqr">Destination-Image</param>
		/// <param name="nVal">The value to add to pDst image pixels</param>
		/// <param name="nValSqr">The value to add to pSqr image pixels</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Integral(NPPImage_32sC1 dst, CudaPitchedDeviceVariable<double> sqr, int nVal, double nValSqr, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Integral.nppiSqrIntegral_8u32s64f_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, sqr.DevicePointer, sqr.Pitch, _sizeRoi, nVal, nValSqr, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrIntegral_8u32s64f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region CountInRange
		/// <summary>
		/// Device scratch buffer size (in bytes) for CountInRange.
		/// </summary>
		/// <returns></returns>
		public int CountInRangeGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.CountInRange.nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image CountInRange.
		/// </summary>
		/// <param name="pCounts">Pointer to the number of pixels that fall into the specified range. (1 * sizeof(int))</param>
		/// <param name="nLowerBound">Lower bound of the specified range.</param>
		/// <param name="nUpperBound">Upper bound of the specified range.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="CountInRangeGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CountInRange(CudaDeviceVariable<int> pCounts, byte nLowerBound, byte nUpperBound, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = CountInRangeGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.CountInRange.nppiCountInRange_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image CountInRange.
		/// </summary>
		/// <param name="pCounts">Pointer to the number of pixels that fall into the specified range. (1 * sizeof(int))</param>
		/// <param name="nLowerBound">Lower bound of the specified range.</param>
		/// <param name="nUpperBound">Upper bound of the specified range.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CountInRange(CudaDeviceVariable<int> pCounts, byte nLowerBound, byte nUpperBound, NppStreamContext nppStreamCtx)
		{
			int bufferSize = CountInRangeGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.CountInRange.nppiCountInRange_8u_C1R_Ctx(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region QualityIndex
		/// <summary>
		/// Device scratch buffer size (in bytes) for QualityIndex.
		/// </summary>
		/// <returns></returns>
		public int QualityIndexGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.QualityIndex.nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndexGetBufferHostSize_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (1 * sizeof(float))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="QualityIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void QualityIndex(NPPImage_8uC1 src2, CudaDeviceVariable<float> dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = QualityIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.QualityIndex.nppiQualityIndex_8u32f_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (1 * sizeof(float))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void QualityIndex(NPPImage_8uC1 src2, CudaDeviceVariable<float> dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = QualityIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.QualityIndex.nppiQualityIndex_8u32f_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_8u32f_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GeometryNew

		/// <summary>
		/// image resize.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nXFactor">Factor by which x dimension is changed. </param>
		/// <param name="nYFactor">Factor by which y dimension is changed. </param>
		/// <param name="nXShift">Source pixel shift in x-direction.</param>
		/// <param name="nYShift">Source pixel shift in y-direction.</param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ResizeSqrPixel(NPPImage_8uC1 dst, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect dstRect = new NppiRect(dst.PointRoi, dst.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_8u_C1R_Ctx(_devPtr, _sizeRoi, _pitch, srcRect, dst.DevicePointer, dst.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image remap.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. </param>
		/// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. </param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Remap(NPPImage_8uC1 dst, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.Remap.nppiRemap_8u_C1R_Ctx(_devPtr, _sizeRoi, _pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Scale(NPPImage_16sC1 dst, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.Scale.nppiScale_8u16s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Scale(NPPImage_16uC1 dst, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.Scale.nppiScale_8u16u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u16u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
		/// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Scale(NPPImage_32fC1 dst, float nMin, float nMax, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.Scale.nppiScale_8u32f_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nMin, nMax, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Scale(NPPImage_32sC1 dst, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.Scale.nppiScale_8u32s_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u32s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 1 channel 8-bit unsigned image resize. This primitive matches the behavior of GraphicsMagick++.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nXFactor">Factor by which x dimension is changed.</param>
		/// <param name="nYFactor">Factor by which y dimension is changed.</param>
		/// <param name="eInterpolationMode">The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ResizeSqrPixelAdvanced(NPPImage_8uC1 dst, double nXFactor, double nYFactor, InterpolationMode eInterpolationMode, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ResizeAdvancedGetBufferHostSize(dst.SizeRoi, eInterpolationMode);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			NppiRect roiIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect roiOut = new NppiRect(dst._pointRoi, dst._sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(_devPtrRoi, _sizeOriginal, _pitch, roiIn, dst.DevicePointerRoi, dst.Pitch, roiOut, nXFactor, nYFactor, buffer.DevicePointer, eInterpolationMode, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_8u_C1R_Advanced_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 1 channel 8-bit unsigned image resize. This primitive matches the behavior of GraphicsMagick++.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nXFactor">Factor by which x dimension is changed.</param>
		/// <param name="nYFactor">Factor by which y dimension is changed.</param>
		/// <param name="eInterpolationMode">The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.</param>
		/// <param name="buffer">Device buffer that is used during calculations.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ResizeSqrPixelAdvanced(NPPImage_8uC1 dst, double nXFactor, double nYFactor, InterpolationMode eInterpolationMode, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ResizeAdvancedGetBufferHostSize(dst.SizeRoi, eInterpolationMode);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			NppiRect roiIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect roiOut = new NppiRect(dst._pointRoi, dst._sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(_devPtrRoi, _sizeOriginal, _pitch, roiIn, dst.DevicePointerRoi, dst.Pitch, roiOut, nXFactor, nYFactor, buffer.DevicePointer, eInterpolationMode, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_8u_C1R_Advanced_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMaxEveryNew
		/// <summary>
		/// image MinEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinEvery(NPPImage_8uC1 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MinMaxEvery.nppiMinEvery_8u_C1IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinEvery_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image MaxEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxEvery(NPPImage_8uC1 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MinMaxEvery.nppiMaxEvery_8u_C1IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxEvery_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MirrorNew


		/// <summary>
		/// Mirror image inplace.
		/// </summary>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mirror(NppiAxis flip, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiMirror_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, flip, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region DupNew
		/// <summary>
		/// source image duplicated in all 3 channels of destination image.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dup(NPPImage_8uC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Dup.nppiDup_8u_C1C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_8u_C1C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// source image duplicated in all 4 channels of destination image.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dup(NPPImage_8uC4 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Dup.nppiDup_8u_C1C4R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_8u_C1C4R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void DupA(NPPImage_8uC4 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Dup.nppiDup_8u_C1AC4R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_8u_C1AC4R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//New in Cuda 6.0

		#region FilterBorder
		/// <summary>
		/// Single channel 8-bit unsigned convolution filter with border control.<para/>
		/// General purpose 2D convolution filter with border control.<para/>
		/// Pixels under the mask are multiplied by the respective weights in the mask
		/// and the results are summed. Before writing the result pixel the sum is scaled
		/// back via division by nDivisor. If any portion of the mask overlaps the source
		/// image boundary the requested border type operation is applied to all mask pixels
		/// which fall outside of the source image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order</param>
		/// <param name="nKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided.
		/// If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBorder(NPPImage_8uC1 dest, CudaDeviceVariable<int> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, int nDivisor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBorder.nppiFilterBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, nDivisor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Single channel 8-bit unsigned convolution filter with border control.<para/>
		/// General purpose 2D convolution filter using floating-point weights with border control.<para/>
		/// Pixels under the mask are multiplied by the respective weights in the mask
		/// and the results are summed. Before writing the result pixel the sum is scaled
		/// back via division by nDivisor. If any portion of the mask overlaps the source
		/// image boundary the requested border type operation is applied to all mask pixels
		/// which fall outside of the source image. <para/>
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order</param>
		/// <param name="nKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBorder(NPPImage_8uC1 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBorder32f.nppiFilterBorder32f_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Single channel 8-bit unsigned to 16-bit signed convolution filter with border control.<para/>
		/// General purpose 2D convolution filter using floating-point weights with border control.<para/>
		/// Pixels under the mask are multiplied by the respective weights in the mask
		/// and the results are summed. Before writing the result pixel the sum is scaled
		/// back via division by nDivisor. If any portion of the mask overlaps the source
		/// image boundary the requested border type operation is applied to all mask pixels
		/// which fall outside of the source image. <para/>
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order</param>
		/// <param name="nKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBorder(NPPImage_16sC1 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBorder32f.nppiFilterBorder32f_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Filter Median
		/// <summary>
		/// Result pixel value is the median of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMedian(NPPImage_8uC1 dst, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize, nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.ImageMedianFilter.nppiFilterMedian_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the median of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMedian(NPPImage_8uC1 dst, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize, nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageMedianFilter.nppiFilterMedian_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for FilterMedian.
		/// </summary>
		/// <returns></returns>
		public int FilterMedianGetBufferHostSize(NppiSize oMaskSize, NppStreamContext nppStreamCtx)
		{
			uint bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageMedianFilter.nppiFilterMedianGetBufferSize_8u_C1R_Ctx(_sizeRoi, oMaskSize, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianGetBufferSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return (int)bufferSize; //We stay consistent with other GetBufferHostSize functions and convert to int.
		}
		#endregion

		#region MaxError
		/// <summary>
		/// image maximum error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region AverageError
		/// <summary>
		/// image average error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region MaximumRelative_Error
		/// <summary>
		/// image maximum relative error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaximumRelativeError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaximumRelativeError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region AverageRelative_Error
		/// <summary>
		/// image average relative error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageRelativeError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C1R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageRelativeError(NPPImage_8uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region FilterScharrBorder
		/// <summary>
		/// Filters the image using a horizontal Scharr filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterScharrHorizBorder(NPPImage_16sC1 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterScharrHorizBorder.nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrHorizBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Scharr filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterScharrVertBorder(NPPImage_16sC1 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterScharrVertBorder.nppiFilterScharrVertBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrVertBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterSobelBorder
		/// <summary>
		/// Filters the image using a horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelHorizBorder(NPPImage_8uC1 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelHorizBorder(NPPImage_16sC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVertBorder(NPPImage_8uC1 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelVertBorder.nppiFilterSobelVertBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVertBorder(NPPImage_16sC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelVertBorder.nppiFilterSobelVertBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a second derivative, horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelHorizSecondBorder(NPPImage_16sC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelHorizSecondBorder.nppiFilterSobelHorizSecondBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizSecondBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a second derivative, vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVertSecondBorder(NPPImage_16sC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelVertSecondBorder.nppiFilterSobelVertSecondBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertSecondBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a second cross derivative Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelCrossBorder(NPPImage_16sC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelCrossBorder.nppiFilterSobelCrossBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelCrossBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion


		#region ColorTwist
		/// <summary>
		/// An input color twist matrix with floating-point pixel values is applied
		/// within ROI.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ColorTwist(NPPImage_8uC1 dest, float[,] twistMatrix, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorProcessing.nppiColorTwist32f_8u_C1R_Ctx(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// in place color twist.
		/// 
		/// An input color twist matrix with floating-point coefficient values is applied
		/// within ROI.
		/// </summary>
		/// <param name="aTwist">The color twist matrix with floating-point coefficient values. [3,4]</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ColorTwist(float[,] aTwist, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorProcessing.nppiColorTwist32f_8u_C1IR_Ctx(_devPtr, _pitch, _sizeRoi, aTwist, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion


		#region FilterGauss
		/// <summary>Filters the image using a Gaussian filter kernel with border control:<para/>
		/// 1/16 2/16 1/16<para/>
		/// 2/16 4/16 2/16<para/>
		/// 1/16 2/16 1/16<para/>
		/// <para/> or <para/>
		/// 2/571 7/571 12/571 7/571 2/571<para/>
		/// 7/571 31/571 52/571 31/571 7/571<para/>
		/// 12/571 52/571 127/571 52/571 12/571<para/>
		/// 7/571 31/571 52/571 31/571 7/571<para/>
		/// 2/571 7/571 12/571 7/571 2/571<para/>
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussBorder(NPPImage_8uC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussBorder.nppiFilterGaussBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//New in Cuda 7.0
		#region FilterColumnBorder
		/// <summary>
		/// General purpose 1D convolution column filter with border control.<para/>
		/// Pixels under the mask are multiplied by the respective weights in the mask
		/// and the results are summed. Before writing the result pixel the sum is scaled
		/// back via division by nDivisor. If any portion of the mask overlaps the source
		/// image boundary the requested border type operation is applied to all mask pixels
		/// which fall outside of the source image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterColumnBorder(NPPImage_8uC1 dest, CudaDeviceVariable<int> Kernel, int nAnchor, int nDivisor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterColumnBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, nDivisor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// General purpose 1D convolution column filter with border control.<para/>
		/// Pixels under the mask are multiplied by the respective weights in the mask
		/// and the results are summed. If any portion of the mask overlaps the source
		/// image boundary the requested border type operation is applied to all mask pixels
		/// which fall outside of the source image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterColumnBorder(NPPImage_8uC1 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterColumnBorder32f_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterRow
		/// <summary>
		/// General purpose 1D convolution row filter with border control.<para/>
		/// Pixels under the mask are multiplied by the respective weights in the mask
		/// and the results are summed. If any portion of the mask overlaps the source
		/// image boundary the requested border type operation is applied to all mask pixels
		/// which fall outside of the source image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRowBorder(NPPImage_8uC1 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterRowBorder32f_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		#endregion

		#region SumWindow

		/// <summary>
		/// Apply Column Window Summation filter over a 1D mask region around each
		/// source pixel for 1-channel 8 bit/pixel input images with 32-bit floating point
		/// output.  
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring column pixel values in a mask region of the source image defined by
		/// nMaskSize and nAnchor. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SumWindowColumnBorder(NPPImage_32fC1 dest, int nMaskSize, int nAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.WindowSum1D.nppiSumWindowColumnBorder_8u32f_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowColumnBorder_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Apply Row Window Summation filter over a 1D mask region around each source
		/// pixel for 1-channel 8-bit pixel input images with 32-bit floating point output.  
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring row pixel values in a mask region of the source image defined
		/// by nKernelDim and nAnchorX. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SumWindowRowBorder(NPPImage_32fC1 dest, int nMaskSize, int nAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.WindowSum1D.nppiSumWindowRowBorder_8u32f_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowRowBorder_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterBox


		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBoxBorder(NPPImage_8uC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFixedFilters2D.nppiFilterBoxBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Filter Min/Max


		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMinBorder(NPPImage_8uC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMinBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMinBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMaxBorder(NPPImage_8uC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMaxBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMaxBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterOthers
		

		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterPrewittHorizBorder(NPPImage_8uC1 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittHorizBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHorizBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterPrewittVertBorder(NPPImage_8uC1 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittVertBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVertBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsDownBorder(NPPImage_8uC1 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsDownBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDownBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsUpBorder(NPPImage_8uC1 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsUpBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUpBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLaplaceBorder(NPPImage_8uC1 dst, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLaplaceBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplaceBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterHighPassBorder(NPPImage_8uC1 dst, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterHighPassBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPassBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLowPassBorder(NPPImage_8uC1 dst, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLowPassBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPassBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSharpenBorder(NPPImage_8uC1 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSharpenBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpenBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region Filter Unsharp

		/// <summary>
		/// Filters the image using a unsharp-mask sharpening filter kernel with border control.<para/>
		/// The algorithm involves the following steps:<para/>
		/// Smooth the original image with a Gaussian filter, with the width controlled by the nRadius.<para/>
		/// Subtract the smoothed image from the original to create a high-pass filtered image.<para/>
		/// Apply any clipping needed on the high-pass image, as controlled by the nThreshold.<para/>
		/// Add a certain percentage of the high-pass filtered image to the original image, 
		/// with the percentage controlled by the nWeight.
		/// In pseudocode this algorithm can be written as:<para/>
		/// HighPass = Image - Gaussian(Image)<para/>
		/// Result = Image + nWeight * HighPass * ( |HighPass| >= nThreshold ) <para/>
		/// where nWeight is the amount, nThreshold is the threshold, and >= indicates a Boolean operation, 1 if true, or 0 otherwise.
		/// <para/>
		/// If any portion of the mask overlaps the source image boundary, the requested border type 
		/// operation is applied to all mask pixels which fall outside of the source image.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
		/// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
		/// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
		/// <param name="nThreshold">The threshold needed to apply the difference amount.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="buffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterUnsharpBorder(NPPImage_8uC1 dst, float nRadius, float nSigma, float nWeight, float nThreshold, NppiBorderType eBorderType, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			if (buffer.Size < FilterUnsharpGetBufferSize(nRadius, nSigma))
				throw new NPPException("Provided buffer is too small."); 

			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterUnsharpBorder_8u_C1R_Ctx(_devPtr, _pitch, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nRadius, nSigma, nWeight, nThreshold, eBorderType, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Filter Gauss Advanced

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGauss(NPPImage_8uC1 dst, CudaDeviceVariable<float> Kernel, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterGaussAdvanced_8u_C1R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvanced_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussBorder(NPPImage_8uC1 dst, CudaDeviceVariable<float> Kernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussBorder.nppiFilterGaussAdvancedBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvancedBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		//New in Cuda 7.5
		#region DeBayer
		/// <summary>
		/// Grayscale Color Filter Array to RGB Color Debayer conversion. Generates one RGB color pixel for every grayscale source pixel.<para/>
		/// Source and destination images must have even width and height.  Missing pixel colors are generated using bilinear interpolation
		/// with chroma correlation of generated green values (eInterpolation MUST be set to 0). eGrid allows the user to specify the Bayer grid 
		/// registration position at source image location oSrcROI.x, oSrcROI.y relative to pSrc.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eGrid">enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.</param>
		/// <param name="eInterpolation">MUST be <see cref="InterpolationMode.Undefined"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CFAToRGB(NPPImage_8uC3 dest, NppiBayerGridPosition eGrid, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorDebayer.nppiCFAToRGB_8u_C1C3R_Ctx(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointerRoi, dest.Pitch, eGrid, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCFAToRGB_8u_C1C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Grayscale Color Filter Array to RGB Color Debayer conversion. Generates one RGB color pixel for every grayscale source pixel.<para/>
		/// Source and destination images must have even width and height.  Missing pixel colors are generated using bilinear interpolation
		/// with chroma correlation of generated green values (eInterpolation MUST be set to 0). eGrid allows the user to specify the Bayer grid 
		/// registration position at source image location oSrcROI.x, oSrcROI.y relative to pSrc.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eGrid">enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.</param>
		/// <param name="eInterpolation">MUST be <see cref="InterpolationMode.Undefined"/></param>
		/// <param name="nAlpha">constant alpha value to be written to each destination pixel</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CFAToRGB(NPPImage_8uC4 dest, NppiBayerGridPosition eGrid, InterpolationMode eInterpolation, byte nAlpha, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorDebayer.nppiCFAToRGBA_8u_C1AC4R_Ctx(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointerRoi, dest.Pitch, eGrid, eInterpolation, nAlpha, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCFAToRGBA_8u_C1AC4R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region FilterGaussAdvancedBorder

		/// <summary>
		/// Single channel 8-bit unsigned Gauss filter with downsampling and border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <param name="nFilterTaps">The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. </param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussPyramidLayerDownBorder(NPPImage_8uC1 dest, float nRate, int nFilterTaps, CudaDeviceVariable<float> pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussPyramid.nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nRate, nFilterTaps, pKernel.DevicePointer, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Single channel 8-bit unsigned Gauss filter with downsampling and border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <param name="nFilterTaps">The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. </param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussPyramidLayerUpBorder(NPPImage_8uC1 dest, float nRate, int nFilterTaps, CudaDeviceVariable<float> pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussPyramid.nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nRate, nFilterTaps, pKernel.DevicePointer, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region FilterBilateralGaussBorder


		/// <summary>
		/// Single channel 8-bit unsigned bilateral Gauss filter with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRadius">The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc. Radius values from 1 to 32 are supported.</param>
		/// <param name="nStepBetweenSrcPixels">The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. </param>
		/// <param name="nValSquareSigma">The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel and the source image pixel at the center of the filter kernel.</param>
		/// <param name="nPosSquareSigma">The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel and the source image pixel at the center of the filter kernel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBilateralGaussBorder(NPPImage_8uC1 dest, int nRadius, int nStepBetweenSrcPixels, float nValSquareSigma, float nPosSquareSigma, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBilateralGaussBorder.nppiFilterBilateralGaussBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nRadius, nStepBetweenSrcPixels, nValSquareSigma, nPosSquareSigma, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBilateralGaussBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region GradientVectorPrewittBorder

		/// <summary>
		/// 1 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
		/// and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
		/// </summary>
		/// <param name="destX">X vector destination_image_pointer</param>
		/// <param name="destY">Y vector destination_image_pointer.</param>
		/// <param name="destMag">magnitude destination_image_pointer.</param>
		/// <param name="destAngle">angle destination_image_pointer.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void GradientVectorPrewittBorder(NPPImage_16sC1 destX, NPPImage_16sC1 destY, NPPImage_16sC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GradientVectorPrewittBorder.nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GradientVectorScharrBorder

		/// <summary>
		/// 1 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
		/// and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
		/// </summary>
		/// <param name="destX">X vector destination_image_pointer</param>
		/// <param name="destY">Y vector destination_image_pointer.</param>
		/// <param name="destMag">magnitude destination_image_pointer.</param>
		/// <param name="destAngle">angle destination_image_pointer.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void GradientVectorScharrBorder(NPPImage_16sC1 destX, NPPImage_16sC1 destY, NPPImage_16sC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GradientVectorScharrBorder.nppiGradientVectorScharrBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorScharrBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GradientVectorSobelBorder

		/// <summary>
		/// 1 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
		/// and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
		/// </summary>
		/// <param name="destX">X vector destination_image_pointer</param>
		/// <param name="destY">Y vector destination_image_pointer.</param>
		/// <param name="destMag">magnitude destination_image_pointer.</param>
		/// <param name="destAngle">angle destination_image_pointer.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void GradientVectorSobelBorder(NPPImage_16sC1 destX, NPPImage_16sC1 destY, NPPImage_16sC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GradientVectorSobelBorder.nppiGradientVectorSobelBorder_8u16s_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorSobelBorder_8u16s_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region FilterCannyBorder

		/// <summary>
		/// 1 channel 8-bit unsigned grayscale to 1 channel 8-bit unsigned black (0x00) and white (0xFF) image with border control.
		/// </summary>
		/// <param name="dest">destination_image_pointer</param>
		/// <param name="eFilterType">selects between Sobel or Scharr filter type.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="nLowThreshold">low hysteresis threshold value.</param>
		/// <param name="nHighThreshold">high hysteresis threshold value.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="pDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterCannyBorderGetBufferSize() above)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterCannyBorder(NPPImage_8uC1 dest, DifferentialKernel eFilterType,
					 MaskSize eMaskSize, short nLowThreshold, short nHighThreshold, NppiNorm eNorm,
					 NppiBorderType eBorderType, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterCannyBorder.nppiFilterCannyBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eFilterType, eMaskSize, nLowThreshold, nHighThreshold, eNorm, eBorderType, pDeviceBuffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterCannyBorder_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region FilterHarrisCornersBorder

		/// <summary>
		/// 1 channel 8-bit unsigned grayscale to 1 channel 8-bit unsigned black (0x00) and white (0xFF) image with border control.
		/// </summary>
		/// <param name="dest">destination_image_pointer</param>
		/// <param name="eFilterType">selects between Sobel or Scharr filter type.</param>
		/// <param name="eMaskSize">fixed filter mask size to use (3x3 or 5x5 for Sobel).</param>
		/// <param name="eAvgWindowSize">fixed window mask size to use (3x3 or 5x5).</param>
		/// <param name="nK">Harris Corners constant (commonly used value is 0.04F).</param>
		/// <param name="nScale">output is scaled by this scale factor.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="pDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterHarrisCornersBorderGetBufferSize() above)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterHarrisCornersBorder(NPPImage_32fC1 dest, DifferentialKernel eFilterType,
										MaskSize eMaskSize, MaskSize eAvgWindowSize, float nK, float nScale,
										NppiBorderType eBorderType, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterHarrisCornersBorder.nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eFilterType, eMaskSize, eAvgWindowSize, nK, nScale, eBorderType, pDeviceBuffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region NewInCuda9.0


        /// <summary>
        /// Single channel 8-bit unsigned threshold adaptive box filter with border control.
        /// </summary>
        /// <param name="dest">destination_image_pointer</param>
        /// <param name="eFilterType">selects between Sobel or Scharr filter type.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation, Width and Height must be equal and odd.</param>
        /// <param name="nDelta">Neighborhood average adjustment value.</param>
        /// <param name="nValGT">Destination output value if source pixel is greater than average.</param>
        /// <param name="nValLE">Destination output value if source pixel is less than or equal to average.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void FilterThresholdAdaptiveBoxBorder(NPPImage_8uC1 dest, DifferentialKernel eFilterType,
                     NppiSize oMaskSize, float nDelta, byte nValGT, byte nValLE,
                     NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.FilterThresholdAdaptiveBoxBorder.nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, nDelta, nValGT, nValLE, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Wiener filter with border control.
        /// </summary>
        /// <param name="dest">destination_image_pointer</param>
        /// <param name="oMaskSize">Pixel Width and Height of the rectangular region of interest surrounding the source pixel.</param>
        /// <param name="oAnchor">Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.</param>
        /// <param name="aNoise">Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void FilterWienerBorder(NPPImage_8uC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, float aNoise, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.FilterWienerBorder.nppiFilterWienerBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, aNoise, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterWienerBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 8-bit unsigned binarized (0, 255) source feature (canny edges, etc.) source image to list of lines in point polar format
        /// representing the length (rho) and angle (theta) of each line from the origin of the normal to the line using the formula rho = x cos(theta) + y sin(theta).
        /// The level of discretization, nDelta, is specified as an input parameter. The performance and effectiveness of this function highly depends on
        /// this parameter with higher performance for larger numbers and more detailed results for lower numbers. nDelta must have the same values as
        /// those used in the nppiFilterHoughLineGetBufferSize() function call.
        /// </summary>
        /// <param name="nDelta">Discretization steps, range 0.0F &lt; radial increment nDelta.rho &lt; 3.0F, 1.0F recommended, range 0.25F &lt; angular increment nDelta.theta &lt; 3.0F, 1.0F recommended.</param>
        /// <param name="nThreshold">Minimum number of points to accept a line.</param>
        /// <param name="pDeviceLines">Device pointer to (nMaxLineCount * sizeof(NppPointPolar) line objects.</param>
        /// <param name="pDeviceLineCount">The number of lines detected by this function up to nMaxLineCount.</param>
        /// <param name="pDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterHoughLineGetBufferSize() above)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void FilterHoughLine(NppPointPolar nDelta, int nThreshold, CudaDeviceVariable<NppPointPolar> pDeviceLines, CudaDeviceVariable<int> pDeviceLineCount, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.FilterHoughLine.nppiFilterHoughLine_8u32f_C1R_Ctx(_devPtr, _pitch, _sizeRoi, nDelta, nThreshold, pDeviceLines.DevicePointer, pDeviceLines.Size, pDeviceLineCount.DevicePointer, pDeviceBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHoughLine_8u32f_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 8-bit unsigned binarized (0, 255) source feature (canny edges, etc.) source image to list of lines in point polar format
        /// representing the length (rho) and angle (theta) of each line from the origin of the normal to the line using the formula rho = x cos(theta) + y sin(theta).
        /// The level of discretization, nDelta, is specified as an input parameter. The performance and effectiveness of this function highly depends on
        /// this parameter with higher performance for larger numbers and more detailed results for lower numbers. nDelta must have the same values as
        /// those used in the nppiFilterHoughLineGetBufferSize() function call. The oDstROI region limits are used to limit accepted lines to those that fall within
        /// those limits.
        /// </summary>
        /// <param name="nDelta">Discretization steps, range 0.0F &lt; radial increment nDelta.rho &lt; 3.0F, 1.0F recommended, range 0.25F &lt; angular increment nDelta.theta &lt; 3.0F, 1.0F recommended.</param>
        /// <param name="nThreshold">Minimum number of points to accept a line.</param>
        /// <param name="pDeviceLines">Device pointer to (nMaxLineCount * sizeof(NppPointPolar) line objects.</param>
        /// <param name="oDstROI">Region limits with oDstROI[0].rho &lt;= accepted rho &lt;= oDstROI[1].rho and oDstROI[0].theta &lt;= accepted theta &lt;= oDstROI[1].theta.</param>
        /// <param name="pDeviceLineCount">The number of lines detected by this function up to nMaxLineCount.</param>
        /// <param name="pDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiFilterHoughLineGetBufferSize() above)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void FilterHoughLine(NppPointPolar nDelta, int nThreshold, CudaDeviceVariable<NppPointPolar> pDeviceLines, NppPointPolar[] oDstROI, CudaDeviceVariable<int> pDeviceLineCount, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.FilterHoughLine.nppiFilterHoughLineRegion_8u32f_C1R_Ctx(_devPtr, _pitch, _sizeRoi, nDelta, nThreshold, pDeviceLines.DevicePointer, oDstROI, pDeviceLines.Size, pDeviceLineCount.DevicePointer, pDeviceBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHoughLineRegion_8u32f_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 1 channel 8-bit unsigned grayscale per source image descriptor window location with source image border control 
        /// to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
        /// call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
        /// total size for nLocations of output histogram block descriptor windows.
        /// </summary>
        /// <param name="hpLocations">Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a </param>
        /// <param name="pDstWindowDescriptorBuffer">Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).</param>
        /// <param name="oHOGConfig">Requested HOG configuration parameters structure.</param>
        /// <param name="pScratchBuffer">Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void HistogramOfGradientsBorder(NppiPoint[] hpLocations, CudaDeviceVariable<byte> pDstWindowDescriptorBuffer, NppiHOGConfig oHOGConfig, CudaDeviceVariable<byte> pScratchBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.HistogramOfOrientedGradientsBorder.nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, hpLocations, hpLocations.Length, pDstWindowDescriptorBuffer.DevicePointer, _sizeRoi, oHOGConfig, pScratchBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// One-channel 8-bit unsigned image MSE.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pMSE">Pointer to the computed MSE of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiMSEGetBufferHostSize_8u_C1R to compute the required size (in bytes).</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MSE(NPPImage_8uC1 src2 , CudaDeviceVariable<float> pMSE, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiMSE_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pMSE.DevicePointer, pDeviceBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSE_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// One-channel 8-bit unsigned image PSNR.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pPSNR">Pointer to the computed PSNR of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiPSNRGetBufferHostSize_8u_C1R to compute the required size (in bytes).</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void PSNR(NPPImage_8uC1 src2, CudaDeviceVariable<float> pPSNR, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiPSNR_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pPSNR.DevicePointer, pDeviceBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiPSNR_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// One-channel 8-bit unsigned image SSIM.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pSSIM">Pointer to the computed SSIM of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiSSIMGetBufferHostSize_8u_C1R to compute the required size (in bytes).</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void SSIM(NPPImage_8uC1 src2, CudaDeviceVariable<float> pSSIM, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiSSIM_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pSSIM.DevicePointer, pDeviceBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSSIM_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// One-channel 8-bit unsigned image MS-SSIM.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pMSSSIM">Pointer to the computed SSIM of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiMSSSIMGetBufferHostSize_8u_C1R to compute the required size (in bytes).</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MSSSIM(NPPImage_8uC1 src2, CudaDeviceVariable<float> pMSSSIM, CudaDeviceVariable<byte> pDeviceBuffer, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiMSSSIM_8u_C1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pMSSSIM.DevicePointer, pDeviceBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSSSIM_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Buffer size for \ref nppiMSE_8u_C1R.
        /// </summary>
        public int MSEGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int ret = 0;
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiMSEGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref ret, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSEGetBufferHostSize_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// Buffer size for \ref nppiPSNR_8u_C1R.
        /// </summary>
        public int PSNRGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int ret = 0;
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiPSNRGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref ret, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiPSNRGetBufferHostSize_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// Buffer size for \ref nppiSSIM_8u_C1R.
        /// </summary>
        public int SSIMGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int ret = 0;
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiSSIMGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref ret, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSSIMGetBufferHostSize_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// Buffer size for \ref nppiMSSSIM_8u_C1R.
        /// </summary>
        public int MSSSIMGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int ret = 0;
            status = NPPNativeMethods_Ctx.NPPi.IQA.nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx(_sizeRoi, ref ret, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }




        /// <summary>
        /// Gray scale dilation with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="Mask">Pointer to the start address of the mask array.</param>
        /// <param name="aMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void GrayDilateBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.DilationWithBorderControl.nppiGrayDilateBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGrayDilateBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Gray scale erosion with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="Mask">Pointer to the start address of the mask array.</param>
        /// <param name="aMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void GrayErodeBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ErosionWithBorderControl.nppiGrayErodeBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGrayErodeBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 8-bit unsigned integer in place label markers image generation.
        /// </summary>
        /// <param name="nMinVal">Pixel values less than or equal to nMinVal will be excluded as members of any connected region and given a label ID of 0.</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersGetBufferSize call.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        /// <returns>The maximum generated marker label ID will be returned.</returns>
        public int LabelMarkers(byte nMinVal, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer, NppStreamContext nppStreamCtx)
        {
            int pNumber = 0;
            status = NPPNativeMethods_Ctx.NPPi.LabelMarkers.nppiLabelMarkers_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nMinVal, eNorm, ref pNumber, pBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabelMarkers_8u_C1IR_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return pNumber;
        }


        /// <summary>
        /// 1 channel 8-bit to 32-bit unsigned integer label markers image generation.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="nMinVal">Pixel values less than or equal to nMinVal will be excluded as members of any connected region and given a label ID of 0.</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersGetBufferSize call.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        /// <returns>The maximum generated marker label ID will be returned.</returns>
        public int LabelMarkers(NPPImage_32uC1 dest, byte nMinVal, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer, NppStreamContext nppStreamCtx)
        {
            int pNumber = 0;
            status = NPPNativeMethods_Ctx.NPPi.LabelMarkers.nppiLabelMarkers_8u32u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMinVal, eNorm, ref pNumber, pBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabelMarkers_8u32u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return pNumber;
        }

        /// <summary>
        /// 1 channel 8-bit unsigned integer in place connected region marker label renumbering with numbering sparseness elimination.
        /// </summary>
        /// <param name="nStartingNumber">The value returned from a previous call to the nppiLabelMarkers_8u32u function.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding CompressMarkerLabelsGetBufferSize call.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        /// <returns>The maximum renumbered marker label ID will be returned.</returns>
        public int CompressMarkerLabels(int nStartingNumber, CudaDeviceVariable<byte> pBuffer, NppStreamContext nppStreamCtx)
        {
            int pNewNumber = 0;
            status = NPPNativeMethods_Ctx.NPPi.LabelMarkers.nppiCompressMarkerLabels_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nStartingNumber, ref pNewNumber, pBuffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressMarkerLabels_8u_C1IR_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return pNewNumber;
        }


        /// <summary>
        /// 1 channel 8-bit unsigned integer in place region boundary border image generation.
        /// </summary>
        /// <param name="nBorderVal">Pixel value to be used at connected region boundary borders</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void BoundSegments(byte nBorderVal, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.LabelMarkers.nppiBoundSegments_8u_C1IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nBorderVal, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBoundSegments_8u_C1IR_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }




        /// <summary>
        /// 1 channel 8-bit unsigned integer morphological close with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphCloseBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphCloseBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphCloseBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 1 channel 8-bit unsigned integer morphological open with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphOpenBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphOpenBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphOpenBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 8-bit unsigned integer morphological top hat with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphTopHatBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphTopHatBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphTopHatBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 8-bit unsigned integer morphological black hat with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphBlackHatBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphBlackHatBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphBlackHatBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 8-bit unsigned integer morphological gradient with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphGradientBorder(NPPImage_8uC1 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphGradientBorder_8u_C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphGradientBorder_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        #endregion

        #region New in Cuda 10.0

        /// <summary>
        /// image resize batch.
        /// </summary>
        /// <param name="oSmallestSrcSize">Size in pixels of the entire smallest source image width and height, may be from different images.</param>
        /// <param name="oSrcRectROI">Region of interest in the source images (may overlap source image size width and height).</param>
        /// <param name="oSmallestDstSize">Size in pixels of the entire smallest destination image width and height, may be from different images.</param>
        /// <param name="oDstRectROI">Region of interest in the destination images (may overlap destination image size width and height).</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, or NPPI_INTER_SUPER. </param>
        /// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiResizeBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void ResizeBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiResizeBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
        {
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResizeBatch_8u_C1R_Ctx(oSmallestSrcSize, oSrcRectROI, oSmallestDstSize, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeBatch_8u_C1R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// image resize batch for variable ROI.
        /// </summary>
        /// <param name="nMaxWidth">Size in pixels of the entire smallest source image width and height, may be from different images.</param>
        /// <param name="nMaxHeight">Region of interest in the source images (may overlap source image size width and height).</param>
        /// <param name="pBatchSrc">Size in pixels of the entire smallest destination image width and height, may be from different images.</param>
        /// <param name="pBatchDst">Region of interest in the destination images (may overlap destination image size width and height).</param>
        /// <param name="nBatchSize">Device memory pointer to nBatchSize list of NppiResizeBatchCXR structures.</param>
        /// <param name="pBatchROI">Device pointer to NppiResizeBatchROI_Advanced list of per-image variable ROIs.User needs to initialize this structure and copy it to device.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void ResizeBatchAdvanced(int nMaxWidth, int nMaxHeight, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc, CudaDeviceVariable<NppiImageDescriptor> pBatchDst,
                                        CudaDeviceVariable<NppiResizeBatchROI_Advanced> pBatchROI, uint nBatchSize, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
        {
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResizeBatch_8u_C1R_Advanced_Ctx(nMaxWidth, nMaxHeight, pBatchSrc.DevicePointer, pBatchDst.DevicePointer,
                pBatchROI.DevicePointer, pBatchDst.Size, eInterpolation, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeBatch_8u_C1R_Advanced_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }
		#endregion

		#region New in Cuda 10.2

		/// <summary>
		/// image warp affine batch.
		/// </summary>
		/// <param name="oSmallestSrcSize">Size in pixels of the entire smallest source image width and height, may be from different images.</param>
		/// <param name="oSrcRectROI">Region of interest in the source images (may overlap source image size width and height).</param>
		/// <param name="oDstRectROI">Region of interest in the destination images (may overlap destination image size width and height).</param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, or NPPI_INTER_SUPER. </param>
		/// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiWarpAffineBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpAffineBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiWarpAffineBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiWarpAffineBatch_8u_C1R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBatch_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// image warp perspective batch.
		/// </summary>
		/// <param name="oSmallestSrcSize">Size in pixels of the entire smallest source image width and height, may be from different images.</param>
		/// <param name="oSrcRectROI">Region of interest in the source images (may overlap source image size width and height).</param>
		/// <param name="oDstRectROI">Region of interest in the destination images (may overlap destination image size width and height).</param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, or NPPI_INTER_SUPER. </param>
		/// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiWarpAffineBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpPerspectiveBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiWarpAffineBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiWarpPerspectiveBatch_8u_C1R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBatch_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 1 channel 8-bit to 32-bit unsigned integer label markers image generation.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
		/// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LabelMarkersUF(NPPImage_32uC1 dest, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LabelMarkers.nppiLabelMarkersUF_8u32u_C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eNorm, pBuffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabelMarkersUF_8u32u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// color twist batch
		/// An input color twist matrix with floating-point coefficient values is applied
		/// within the ROI for each image in batch. Color twist matrix can vary per image. The same ROI is applied to each image.
		/// </summary>
		/// <param name="nMin">Minimum clamp value.</param>
		/// <param name="nMax">Maximum saturation and clamp value.</param>
		/// <param name="oSizeROI"></param>
		/// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiColorTwistBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void ColorTwistBatch(float nMin, float nMax, NppiSize oSizeROI, CudaDeviceVariable<NppiColorTwistBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwistBatch.nppiColorTwistBatch32f_8u_C1R_Ctx(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch32f_8u_C1R_Ctx", status));
			NPPException.CheckNppStatus(status, pBatchList);
		}


		/// <summary>
		/// in place color twist batch
		/// An input color twist matrix with floating-point coefficient values is applied
		/// within the ROI for each image in batch. Color twist matrix can vary per image. The same ROI is applied to each image.
		/// </summary>
		/// <param name="nMin">Minimum clamp value.</param>
		/// <param name="nMax">Maximum saturation and clamp value.</param>
		/// <param name="oSizeROI"></param>
		/// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiColorTwistBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void ColorTwistBatchI(float nMin, float nMax, NppiSize oSizeROI, CudaDeviceVariable<NppiColorTwistBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwistBatch.nppiColorTwistBatch32f_8u_C1IR_Ctx(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch32f_8u_C1IR_Ctx", status));
			NPPException.CheckNppStatus(status, pBatchList);
		}
		#endregion
	}
}
