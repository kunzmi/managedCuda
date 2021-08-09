﻿//	Copyright (c) 2012, Michael Kunz. All rights reserved.
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
	public partial class NPPImage_8uC3 : NPPImageBase
	{
		#region Constructors
		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="nWidthPixels">Image width in pixels</param>
		/// <param name="nHeightPixels">Image height in pixels</param>
		public NPPImage_8uC3(int nWidthPixels, int nHeightPixels)
		{
			_sizeOriginal.width = nWidthPixels;
			_sizeOriginal.height = nHeightPixels;
			_sizeRoi.width = nWidthPixels;
			_sizeRoi.height = nHeightPixels;
			_channels = 3;
			_isOwner = true;
			_typeSize = sizeof(byte);

			_devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_8u_C3(nWidthPixels, nHeightPixels, ref _pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_8u_C3", res, _pitch, _channels));
			
			if (_devPtr.Pointer == 0)
			{
				throw new NPPException("Device allocation error", null);
			}
			_devPtrRoi = _devPtr;
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="width">Image width in pixels</param>
		/// <param name="height">Image height in pixels</param>
		/// <param name="pitch">Pitch / Line step</param>
		/// <param name="isOwner">If TRUE, devPtr is freed when disposing</param>
		public NPPImage_8uC3(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
		{
			_devPtr = devPtr;
			_devPtrRoi = _devPtr;
			_sizeOriginal.width = width;
			_sizeOriginal.height = height;
			_sizeRoi.width = width;
			_sizeRoi.height = height;
			_pitch = pitch;
			_channels = 3;
			_isOwner = isOwner;
			_typeSize = sizeof(byte);
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="width">Image width in pixels</param>
		/// <param name="height">Image height in pixels</param>
		/// <param name="pitch">Pitch / Line step</param>
		public NPPImage_8uC3(CUdeviceptr devPtr, int width, int height, int pitch)
			: this(devPtr, width, height, pitch, false)
		{

		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
		/// </summary>
		/// <param name="image">NPP image</param>
		public NPPImage_8uC3(NPPImageBase image)
			: this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
		{

		}

		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="size">Image size</param>
		public NPPImage_8uC3(NppiSize size)
			: this(size.width, size.height)
		{ 
			
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="size">Image size</param>
		/// <param name="pitch">Pitch / Line step</param>
		/// <param name="isOwner">If TRUE, devPtr is freed when disposing</param>
		public NPPImage_8uC3(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
			: this(devPtr, size.width, size.height, pitch, isOwner)
		{ 
			
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="size">Image size</param>
		/// <param name="pitch">Pitch / Line step</param>
		public NPPImage_8uC3(CUdeviceptr devPtr, NppiSize size, int pitch)
			: this(devPtr, size.width, size.height, pitch)
		{

		}

		/// <summary>
		/// For dispose
		/// </summary>
		~NPPImage_8uC3()
		{
			Dispose (false);
		}
		#endregion

		#region Converter operators

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		public CudaPitchedDeviceVariable<VectorTypes.uchar3> ToCudaPitchedDeviceVariable()
		{
			return new CudaPitchedDeviceVariable<VectorTypes.uchar3>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
		}

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		/// <param name="img">NPPImage</param>
		/// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
		public static implicit operator CudaPitchedDeviceVariable<VectorTypes.uchar3>(NPPImage_8uC3 img)
		{
			return img.ToCudaPitchedDeviceVariable();
		}

		/// <summary>
		/// Converts a CudaPitchedDeviceVariable to a NPPImage 
		/// </summary>
		/// <param name="img">CudaPitchedDeviceVariable</param>
		/// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
		public static implicit operator NPPImage_8uC3(CudaPitchedDeviceVariable<VectorTypes.uchar3> img)
		{
			return img.ToNPPImage();
		}
		#endregion

		#region Color conversion
		/// <summary>
		/// 3 channel 8-bit unsigned RGB to 2 channel chroma packed YCbCr422 color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void RGBToYCbCr(NPPImage_8uC2 dest)
		{
			status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr422_8u_C3C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr422_8u_C3C2R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to packed YCbCr color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void RGBToYCbCr(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to planar YCbCr420 color conversion.
		/// </summary>
		/// <param name="dst0">Destination image channel 0</param>
		/// <param name="dst1">Destination image channel 1</param>
		/// <param name="dst2">Destination image channel 2</param>
		public void RGBToYCbCr420(NPPImage_8uC1 dst0, NPPImage_8uC1 dst1, NPPImage_8uC1 dst2)
		{
			CUdeviceptr[] array = new CUdeviceptr[] { dst0.DevicePointerRoi, dst1.DevicePointerRoi, dst2.DevicePointerRoi };
			int[] arrayStep = new int[] { dst0.Pitch, dst1.Pitch, dst2.Pitch };
			status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr420_8u_C3P3R(_devPtrRoi, _pitch, array, arrayStep, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr420_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YCbCr to RGB color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void YCbCrToRGB(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGB_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGB_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel planar 8-bit unsigned RGB to YCbCr color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void RGBToYCbCr(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc  = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, dest0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCrToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc  = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGB_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, dest0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGB_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to packed RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCr420ToYCbCr411(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arraySrcStep = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr420ToRGB_8u_P3C3R(arraySrc, arraySrcStep, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToRGB_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr:420 to YCbCr:422 resampling.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr420ToYCbCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc  = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arraySrcStep = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			int[] arrayDestStep = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.SamplePatternConversion.nppiYCbCr420ToYCbCr422_8u_P3R(arraySrc, arraySrcStep, arrayDest, arrayDestStep, dest0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr422_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr:422 to YCbCr:411 resampling.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr422ToYCbCr411(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc  = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arraySrcStep = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			int[] arrayDestStep = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.SamplePatternConversion.nppiYCbCr422ToYCbCr411_8u_P3R(arraySrc, arraySrcStep, arrayDest, arrayDestStep, dest0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr411_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr:422 to YCbCr:420 resampling.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr422ToYCbCr420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc  = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arraySrcStep = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			int[] arrayDestStep = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.SamplePatternConversion.nppiYCbCr422ToYCbCr420_8u_P3R(arraySrc, arraySrcStep, arrayDest, arrayDestStep, dest0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr420_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr:420 to YCbCr:411 resampling.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel 0</param>
		/// <param name="destCbCr">Destination image channel 1</param>
		public static void YCbCr420ToYCbCr411(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc  = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arraySrcStep = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.SamplePatternConversion.nppiYCbCr420ToYCbCr411_8u_P3P2R(arraySrc, arraySrcStep, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, destY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr411_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		#endregion

		#region Copy
		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		public void Copy(NPPImage_8uC1 dst, int channel)
		{
			if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_8u_C3C1R(_devPtrRoi + channel * _typeSize, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C3C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 8-bit unsigned packed to planar image copy.
		/// </summary>
		/// <param name="dst0">Destination image channel 0</param>
		/// <param name="dst1">Destination image channel 1</param>
		/// <param name="dst2">Destination image channel 2</param>
		public void Copy(NPPImage_8uC1 dst0, NPPImage_8uC1 dst1, NPPImage_8uC1 dst2)
		{
			CUdeviceptr[] array = new CUdeviceptr[] { dst0.DevicePointerRoi, dst1.DevicePointerRoi, dst2.DevicePointerRoi };
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_8u_C3P3R(_devPtrRoi, _pitch, array, dst0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 8-bit unsigned planar to packed image copy.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void Copy(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] array = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_8u_P3C3R(array, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channelSrc">Channel number. This number is added to the src pointer</param>
		/// <param name="channelDst">Channel number. This number is added to the dst pointer</param>
		public void Copy(NPPImage_8uC3 dst, int channelSrc, int channelDst)
		{
			if (channelSrc < 0 | channelSrc >= _channels) throw new ArgumentOutOfRangeException("channelSrc", "channelSrc must be in range [0..2].");
			if (channelDst < 0 | channelDst >= dst.Channels) throw new ArgumentOutOfRangeException("channelDst", "channelDst must be in range [0..2].");
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_8u_C3CR(_devPtrRoi + channelSrc * _typeSize, _pitch, dst.DevicePointerRoi + channelDst * _typeSize, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C3CR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="mask">Mask image</param>
		public void Copy(NPPImage_8uC3 dst, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_8u_C3MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C3MR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Logical
		/// <summary>
		/// image bit shift by constant (left).
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 3)</param>
		/// <param name="dest">Destination image</param>
		public void LShiftC(uint[] nConstant, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_8u_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (left), inplace.
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 3)</param>
		public void LShiftC(uint[] nConstant)
		{
			status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_8u_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (right).
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 3)</param>
		/// <param name="dest">Destination image</param>
		public void RShiftC(uint[] nConstant, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_8u_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (right), inplace.
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 3)</param>
		public void RShiftC(uint[] nConstant)
		{
			status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_8u_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical and.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void And(NPPImage_8uC3 src2, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.And.nppiAnd_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical and.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void And(NPPImage_8uC3 src2)
		{
			status = NPPNativeMethods.NPPi.And.nppiAnd_8u_C3IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical and with constant.
		/// </summary>
		/// <param name="nConstant">Value (Array length = 3)</param>
		/// <param name="dest">Destination image</param>
		public void And(byte[] nConstant, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.AndConst.nppiAndC_8u_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical and with constant.
		/// </summary>
		/// <param name="nConstant">Value (Array length = 3)</param>
		public void And(byte[] nConstant)
		{
			status = NPPNativeMethods.NPPi.AndConst.nppiAndC_8u_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Or.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void Or(NPPImage_8uC3 src2, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.Or.nppiOr_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Or.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void Or(NPPImage_8uC3 src2)
		{
			status = NPPNativeMethods.NPPi.Or.nppiOr_8u_C3IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Or with constant.
		/// </summary>
		/// <param name="nConstant">Value (Array length = 3)</param>
		/// <param name="dest">Destination image</param>
		public void Or(byte[] nConstant, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.OrConst.nppiOrC_8u_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Or with constant.
		/// </summary>
		/// <param name="nConstant">Value (Array length = 3)</param>
		public void Or(byte[] nConstant)
		{
			status = NPPNativeMethods.NPPi.OrConst.nppiOrC_8u_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Xor.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void Xor(NPPImage_8uC3 src2, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.Xor.nppiXor_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Xor.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void Xor(NPPImage_8uC3 src2)
		{
			status = NPPNativeMethods.NPPi.Xor.nppiXor_8u_C3IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image logical Xor with constant.
		/// </summary>
		/// <param name="nConstant">Value (Array length = 3)</param>
		/// <param name="dest">Destination image</param>
		public void Xor(byte[] nConstant, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.XorConst.nppiXorC_8u_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Xor with constant.
		/// </summary>
		/// <param name="nConstant">Value (Array length = 3)</param>
		public void Xor(byte[] nConstant)
		{
			status = NPPNativeMethods.NPPi.XorConst.nppiXorC_8u_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Image logical Not.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Not(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.Not.nppiNot_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNot_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image logical Not.
		/// </summary>
		public void Not()
		{
			status = NPPNativeMethods.NPPi.Not.nppiNot_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNot_8u_C3IR", status));
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
		public void Add(NPPImage_8uC3 src2, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_8u_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Add(NPPImage_8uC3 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_8u_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Add(byte[] nConstant, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Add(byte[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_8u_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_8u_C3IRSfs", status));
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
		public void Sub(NPPImage_8uC3 src2, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_8u_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sub(NPPImage_8uC3 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_8u_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sub(byte[] nConstant, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sub(byte[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_8u_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_8u_C3IRSfs", status));
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
		public void Mul(NPPImage_8uC3 src2, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_8u_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Mul(NPPImage_8uC3 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_8u_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Mul(byte[] nConstant, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Mul(byte[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_8u_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image multiplication and scale by max bit width value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void Mul(NPPImage_8uC3 src2, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.MulScale.nppiMulScale_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulScale_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication and scale by max bit width value
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void Mul(NPPImage_8uC3 src2)
		{
			status = NPPNativeMethods.NPPi.MulScale.nppiMulScale_8u_C3IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulScale_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image and scale by max bit width value
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		public void Mul(byte[] nConstant, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.MulConstScale.nppiMulCScale_8u_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulCScale_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image and scale by max bit width value
		/// </summary>
		/// <param name="nConstant">Value</param>
		public void Mul(byte[] nConstant)
		{
			status = NPPNativeMethods.NPPi.MulConstScale.nppiMulCScale_8u_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulCScale_8u_C3IR", status));
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
		public void Div(NPPImage_8uC3 src2, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_8u_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(NPPImage_8uC3 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_8u_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(byte[] nConstant, NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(byte[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_8u_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(NPPImage_8uC3 src2, NPPImage_8uC3 dest, NppRoundMode rndMode, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_8u_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, rndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(NPPImage_8uC3 src2, NppRoundMode rndMode, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_8u_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, rndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Exp
		/// <summary>
		/// Exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Exp(NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Exp.nppiExp_8u_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Exp(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Exp.nppiExp_8u_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Ln
		/// <summary>
		/// Natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Ln(NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Ln.nppiLn_8u_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Ln(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Ln.nppiLn_8u_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqr
		/// <summary>
		/// Image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqr(NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_8u_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqr(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_8u_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqrt
		/// <summary>
		/// Image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqrt(NPPImage_8uC3 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_8u_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqrt(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_8u_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_8u_C3IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region LUT
		/// <summary>
		/// look-up-table color conversion.<para/>
		/// The LUT is derived from a set of user defined mapping points through linear interpolation.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="values0">array of user defined OUTPUT values, channel 0</param>
		/// <param name="levels0">array of user defined INPUT values, channel 0</param>
		/// <param name="values1">array of user defined OUTPUT values, channel 1</param>
		/// <param name="levels1">array of user defined INPUT values, channel 1</param>
		/// <param name="values2">array of user defined OUTPUT values, channel 2</param>
		/// <param name="levels2">array of user defined INPUT values, channel 2</param>
		public void Lut(NPPImage_8uC3 dest, CudaDeviceVariable<int> values0, CudaDeviceVariable<int> levels0, CudaDeviceVariable<int> values1, CudaDeviceVariable<int> levels1, CudaDeviceVariable<int> values2, CudaDeviceVariable<int> levels2)
		{
			if (values0.Size != levels0.Size) throw new ArgumentException("values0 and levels0 must have same size.");
			if (values1.Size != levels1.Size) throw new ArgumentException("values1 and levels1 must have same size.");
			if (values2.Size != levels2.Size) throw new ArgumentException("values2 and levels2 must have same size.");

			CUdeviceptr[] values = new CUdeviceptr[3];
			CUdeviceptr[] levels = new CUdeviceptr[3];
			int[] levelLengths = new int[3];

			values[0] = values0.DevicePointer;
			values[1] = values1.DevicePointer;
			values[2] = values2.DevicePointer;
			levels[0] = levels0.DevicePointer;
			levels[1] = levels1.DevicePointer;
			levels[2] = levels2.DevicePointer;

			levelLengths[0] = (int)levels0.Size;
			levelLengths[1] = (int)levels1.Size;
			levelLengths[2] = (int)levels2.Size;

			status = NPPNativeMethods.NPPi.ColorProcessing.nppiLUT_Linear_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, values, levels, levelLengths);
						
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Geometric Transforms

		/// <summary>
		/// Compute shape of rotated image.
		/// </summary>
		/// <param name="nAngle">The angle of rotation in degrees.</param>
		/// <param name="nShiftX">Shift along horizontal axis</param>
		/// <param name="nShiftY">Shift along vertical axis</param>
		public double[,] GetRotateQuad(double nAngle, double nShiftX, double nShiftY)
		{
			double[,] quad = new double[4, 2];
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiGetRotateQuad(new NppiRect(_pointRoi, _sizeRoi), quad, nAngle, nShiftX, nShiftY);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetRotateQuad", status));
			NPPException.CheckNppStatus(status, this);
			return quad;
		}

		/// <summary>
		/// Compute bounding-box of rotated image.
		/// </summary>
		/// <param name="nAngle">The angle of rotation in degrees.</param>
		/// <param name="nShiftX">Shift along horizontal axis</param>
		/// <param name="nShiftY">Shift along vertical axis</param>
		public double[,] GetRotateBound(double nAngle, double nShiftX, double nShiftY)
		{
			double[,] bbox = new double[2, 2];
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiGetRotateBound(new NppiRect(_pointRoi, _sizeRoi), bbox, nAngle, nShiftX, nShiftY);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetRotateBound", status));
			NPPException.CheckNppStatus(status, this);
			return bbox;
		}

		/// <summary>
		/// Rotate images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAngle">The angle of rotation in degrees.</param>
		/// <param name="nShiftX">Shift along horizontal axis</param>
		/// <param name="nShiftY">Shift along vertical axis</param>
		/// <param name="eInterpolation">Interpolation mode</param>
		public void Rotate(NPPImage_8uC3 dest, double nAngle, double nShiftX, double nShiftY, InterpolationMode eInterpolation)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiRotate_8u_C3R(_devPtr, _sizeRoi, _pitch, new NppiRect(_pointRoi, _sizeRoi),
				dest.DevicePointer, dest.Pitch, new NppiRect(dest.PointRoi, dest.SizeRoi), nAngle, nShiftX, nShiftY, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRotate_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Mirror image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		public void Mirror(NPPImage_8uC3 dest, NppiAxis flip)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Affine Transformations
		/// <summary>
		/// Calculates affine transform coefficients given source rectangular ROI and its destination quadrangle projection
		/// </summary>
		/// <param name="quad">Destination quadrangle [4,2]</param>
		/// <returns>Affine transform coefficients [2,3]</returns>
		public double[,] GetAffineTransform(double[,] quad)
		{
			double[,] coeffs = new double[2, 3];
			NppiRect rect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiGetAffineTransform(rect, quad, coeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetAffineTransform", status));
			NPPException.CheckNppStatus(status, this);
			return coeffs;
		}

		/// <summary>
		/// Calculates affine transform projection of given source rectangular ROI
		/// </summary>
		/// <param name="coeffs">Affine transform coefficients [2,3]</param>
		/// <returns>Destination quadrangle [4,2]</returns>
		public double[,] GetAffineQuad(double[,] coeffs)
		{
			double[,] quad = new double[4, 2];
			NppiRect rect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiGetAffineQuad(rect, quad, coeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetAffineQuad", status));
			NPPException.CheckNppStatus(status, this);
			return quad;
		}

		/// <summary>
		/// Calculates bounding box of the affine transform projection of the given source rectangular ROI
		/// </summary>
		/// <param name="coeffs">Affine transform coefficients [2,3]</param>
		/// <returns>Destination quadrangle [2,2]</returns>
		public double[,] GetAffineBound(double[,] coeffs)
		{
			double[,] bound = new double[2, 2];
			NppiRect rect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiGetAffineBound(rect, bound, coeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetAffineBound", status));
			NPPException.CheckNppStatus(status, this);
			return bound;
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
		public void WarpAffine(NPPImage_8uC3 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffine_8u_C3R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffine_8u_C3R", status));
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
		public void WarpAffineBack(NPPImage_8uC3 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffineBack_8u_C3R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBack_8u_C3R", status));
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
		public void WarpAffineQuad(double[,] srcQuad, NPPImage_8uC3 dest, double[,] dstQuad, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffineQuad_8u_C3R(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, dstQuad, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineQuad_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
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
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="coeffs">Affine transform coefficients [2,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		public static void WarpAffine(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffine_8u_P3R(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffine_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
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
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="coeffs">Affine transform coefficients [2,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		public static void WarpAffineBack(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffineBack_8u_P3R(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBack_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
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
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="srcQuad">Source quadrangle [4,2]</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="dstQuad">Destination quadrangle [4,2]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		public static void WarpAffineQuad(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, double[,] srcQuad, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double[,] dstQuad, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffineQuad_8u_P3R(src, src0.Size, src0.Pitch, rectIn, srcQuad, dst, dest0.Pitch, rectOut, dstQuad, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineQuad_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Perspective Transformations
		/// <summary>
		/// Calculates affine transform coefficients given source rectangular ROI and its destination quadrangle projection
		/// </summary>
		/// <param name="quad">Destination quadrangle [4,2]</param>
		/// <returns>Perspective transform coefficients [3,3]</returns>
		public double[,] GetPerspectiveTransform(double[,] quad)
		{
			double[,] coeffs = new double[3, 3];
			NppiRect rect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiGetPerspectiveTransform(rect, quad, coeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetPerspectiveTransform", status));
			NPPException.CheckNppStatus(status, this);
			return coeffs;
		}

		/// <summary>
		///Calculates perspective transform projection of given source rectangular ROI
		/// </summary>
		/// <param name="coeffs">Perspective transform coefficients [3,3]</param>
		/// <returns>Destination quadrangle [4,2]</returns>
		public double[,] GetPerspectiveQuad(double[,] coeffs)
		{
			double[,] quad = new double[4, 2];
			NppiRect rect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiGetPerspectiveQuad(rect, quad, coeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetPerspectiveQuad", status));
			NPPException.CheckNppStatus(status, this);
			return quad;
		}

		/// <summary>
		/// Calculates bounding box of the affine transform projection of the given source rectangular ROI
		/// </summary>
		/// <param name="coeffs">Perspective transform coefficients [3,3]</param>
		/// <returns>Destination quadrangle [2,2]</returns>
		public double[,] GetPerspectiveBound(double[,] coeffs)
		{
			double[,] bound = new double[2, 2];
			NppiRect rect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiGetPerspectiveBound(rect, bound, coeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetPerspectiveBound", status));
			NPPException.CheckNppStatus(status, this);
			return bound;
		}

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
		public void WarpPerspective(NPPImage_8uC3 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspective_8u_C3R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspective_8u_C3R", status));
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
		public void WarpPerspectiveBack(NPPImage_8uC3 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspectiveBack_8u_C3R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBack_8u_C3R", status));
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
		public void WarpPerspectiveQuad(double[,] srcQuad, NPPImage_8uC3 dest, double[,] destQuad, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspectiveQuad_8u_C3R(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, destQuad, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveQuad_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

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
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="coeffs">Perspective transform coefficients [3,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		public static void WarpPerspective(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspective_8u_P3R(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspective_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
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
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="coeffs">Perspective transform coefficients [3,3]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		public static void WarpPerspectiveBack(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspectiveBack_8u_P3R(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBack_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
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
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="srcQuad">Source quadrangle [4,2]</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="destQuad">Destination quadrangle [4,2]</param>
		/// <param name="eInterpolation">Interpolation mode: can be <see cref="InterpolationMode.NearestNeighbor"/>, <see cref="InterpolationMode.Linear"/> or <see cref="InterpolationMode.Cubic"/></param>
		public static void WarpPerspectiveQuad(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, double[,] srcQuad, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double[,] destQuad, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspectiveQuad_8u_P3R(src, src0.Size, src0.Pitch, rectIn, srcQuad, dst, dest0.Pitch, rectOut, destQuad, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveQuad_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
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
		public void AlphaComp(byte alpha1, NPPImage_8uC3 src2, byte alpha2, NPPImage_8uC3 dest, NppiAlphaOp nppAlphaOp)
		{
			status = NPPNativeMethods.NPPi.AlphaCompConst.nppiAlphaCompC_8u_C3R(_devPtrRoi, _pitch, alpha1, src2.DevicePointerRoi, src2.Pitch, alpha2, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaCompC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image premultiplication using constant alpha.
		/// </summary>
		/// <param name="alpha">alpha</param>
		/// <param name="dest">Destination image</param>
		public void AlphaPremul(byte alpha, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.AlphaPremulConst.nppiAlphaPremulC_8u_C3R(_devPtrRoi, _pitch, alpha, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaPremulC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// In place alpha premultiplication using constant alpha.
		/// </summary>
		/// <param name="alpha">alpha</param>
		public void AlphaPremul(byte alpha)
		{
			status = NPPNativeMethods.NPPi.AlphaPremulConst.nppiAlphaPremulC_8u_C3IR(alpha, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaPremulC_8u_C3IR", status));
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
		public void ColorTwist(NPPImage_8uC3 dest, float[,] twistMatrix)
		{
			status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist32f_8u_C3R(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel planar 8-bit unsigned color twist.
		/// An input color twist matrix with floating-point pixel values is applied
		/// within ROI.
		/// </summary>
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
		public static void ColorTwist(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, float[,] twistMatrix)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };

			NppStatus status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist32f_8u_P3R(src, src0.Pitch, dst, dest0.Pitch, src0.SizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel planar 8-bit unsigned inplace color twist.
		/// An input color twist matrix with floating-point pixel values is applied
		/// within ROI.
		/// </summary>
		/// <param name="srcDest0">Source / Destination image (Channel 0)</param>
		/// <param name="srcDest1">Source / Destinationimage (Channel 1)</param>
		/// <param name="srcDest2">Source / Destinationimage (Channel 2)</param>
		/// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
		public static void ColorTwist(NPPImage_8uC1 srcDest0, NPPImage_8uC1 srcDest1, NPPImage_8uC1 srcDest2, float[,] twistMatrix)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { srcDest0.DevicePointerRoi, srcDest1.DevicePointerRoi, srcDest2.DevicePointerRoi };

			NppStatus status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_8u_IP3R(src, srcDest0.Pitch, srcDest0.SizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_IP3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region AbsDiff
		/// <summary>
		/// Absolute difference of this minus src2.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void AbsDiff(NPPImage_8uC3 src2, NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.AbsDiff.nppiAbsDiff_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbsDiff_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region CompColorKey
		/// <summary>
		/// packed color complement color key replacement of source image 1 by source image 2
		/// </summary>
		/// <param name="src2">source2 packed pixel format image.</param>
		/// <param name="dest">Destination image</param>
		/// <param name="colorKeyConst">color key constants</param>
		public void CompColorKey(NPPImage_8uC3 src2, NPPImage_8uC3 dest, byte[] colorKeyConst)
		{
			status = NPPNativeMethods.NPPi.CompColorKey.nppiCompColorKey_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, colorKeyConst);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompColorKey_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Convert
		/// <summary>
		/// 8-bit unsigned to 16-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Convert(NPPImage_16uC3 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_8u16u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u16u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned to 16-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Convert(NPPImage_16sC3 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_8u16s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u16s_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned to 32-bit floating point conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Convert(NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_8u32f_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned to 32-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Convert(NPPImage_32sC3 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_8u32s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8u32s_C3R", status));
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
		public void Compare(NPPImage_8uC3 src2, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompare_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompare_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc's pixels with constant value.
		/// </summary>
		/// <param name="nConstant">constant values</param>
		/// <param name="dest">Destination image</param>
		/// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		public void Compare(byte[] nConstant, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompareC_8u_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareC_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region NormInf
		/// <summary>
		/// Scratch-buffer size for Norm inf.
		/// </summary>
		/// <returns></returns>
		public int NormInfGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormInf.nppiNormInfGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for Norm inf (masked).
		/// </summary>
		/// <returns></returns>
		public int NormInfMaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormInf.nppiNormInfGetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void NormInf(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfGetBufferHostSize()"/></param>
		public void NormInf(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void NormInf(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfMaskedGetBufferHostSize()"/></param>
		public void NormInf(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region NormL1
		/// <summary>
		/// Scratch-buffer size for Norm L1.
		/// </summary>
		/// <returns></returns>
		public int NormL1GetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormL1.nppiNormL1GetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for Norm L1 (masked).
		/// </summary>
		/// <returns></returns>
		public int NormL1MaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormL1.nppiNormL1GetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void NormL1(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1GetBufferHostSize()"/></param>
		public void NormL1(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void NormL1(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1MaskedGetBufferHostSize()"/></param>
		public void NormL1(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region NormL2
		/// <summary>
		/// Scratch-buffer size for Norm L2.
		/// </summary>
		/// <returns></returns>
		public int NormL2GetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormL2.nppiNormL2GetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for Norm L2 (masked).
		/// </summary>
		/// <returns></returns>
		public int NormL2MaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormL2.nppiNormL2GetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void NormL2(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2GetBufferHostSize()"/></param>
		public void NormL2(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void NormL2(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2MaskedGetBufferHostSize()"/></param>
		public void NormL2(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_8u_C3CMR", status));
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
		public void Threshold(NPPImage_8uC3 dest, byte[] nThreshold, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		public void Threshold(byte[] nThreshold, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_8u_C3IR", status));
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
		public void ThresholdGT(NPPImage_8uC3 dest, byte[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdGT(byte[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_8u_C3IR", status));
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
		public void ThresholdLT(NPPImage_8uC3 dest, byte[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdLT(byte[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_8u_C3IR", status));
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
		public void Threshold(NPPImage_8uC3 dest, byte[] nThreshold, byte[] nValue, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_8u_C3R", status));
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
		public void Threshold(byte[] nThreshold, byte[] nValue, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_8u_C3IR", status));
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
		public void ThresholdGT(NPPImage_8uC3 dest, byte[] nThreshold, byte[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdGT(byte[] nThreshold, byte[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_8u_C3IR", status));
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
		public void ThresholdLT(NPPImage_8uC3 dest, byte[] nThreshold, byte[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdLT(byte[] nThreshold, byte[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_8u_C3IR", status));
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
		public void ThresholdLTGT(NPPImage_8uC3 dest, byte[] nThresholdLT, byte[] nValueLT, byte[] nThresholdGT, byte[] nValueGT)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_8u_C3R", status));
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
		public void ThresholdLTGT(byte[] nThresholdLT, byte[] nValueLT, byte[] nThresholdGT, byte[] nValueGT)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Mean
		/// <summary>
		/// Scratch-buffer size for Mean.
		/// </summary>
		/// <returns></returns>
		public int MeanGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MeanNew.nppiMeanGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for Mean with mask.
		/// </summary>
		/// <returns></returns>
		public int MeanMaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MeanNew.nppiMeanGetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void Mean(CudaDeviceVariable<double> mean)
		{
			int bufferSize = MeanGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanGetBufferHostSize()"/></param>
		public void Mean(CudaDeviceVariable<double> mean, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void Mean(int coi, CudaDeviceVariable<double> mean, NPPImage_8uC1 mask)
		{
			int bufferSize = MeanMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanMaskedGetBufferHostSize()"/></param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void Mean(int coi, CudaDeviceVariable<double> mean, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MeanStdDev
		/// <summary>
		/// Scratch-buffer size for MeanStdDev.
		/// </summary>
		/// <returns></returns>
		public int MeanStdDevGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_8u_C3CR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_8u_C3CR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		/// <summary>
		/// Scratch-buffer size for MeanStdDev (masked).
		/// </summary>
		/// <returns></returns>
		public int MeanStdDevMaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev)
		{
			int bufferSize = MeanStdDevGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C3CR(_devPtrRoi, _pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C3CR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanStdDevGetBufferHostSize()"/></param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanStdDevGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C3CR(_devPtrRoi, _pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C3CR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanStdDevMaskedGetBufferHostSize()"/></param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sum
		/// <summary>
		/// Scratch-buffer size for nppiSum_8u_C3R.
		/// </summary>
		/// <returns></returns>
		public int SumDoubleGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Sum.nppiSumGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image sum with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void Sum(CudaDeviceVariable<double> result)
		{
			int bufferSize = SumDoubleGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Sum.nppiSum_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SumDoubleGetBufferHostSize()"/></param>
		public void Sum(CudaDeviceVariable<double> result, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = SumDoubleGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Sum.nppiSum_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Min
		/// <summary>
		/// Scratch-buffer size for Min.
		/// </summary>
		/// <returns></returns>
		public int MinGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Min.nppiMinGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		public void Min(CudaDeviceVariable<byte> min)
		{
			int bufferSize = MinGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Min.nppiMin_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinGetBufferHostSize()"/></param>
		public void Min(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Min.nppiMin_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinIndex
		/// <summary>
		/// Scratch-buffer size for MinIndex.
		/// </summary>
		/// <returns></returns>
		public int MinIndexGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndxGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndxGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		public void MinIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MinIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinIndexGetBufferHostSize()"/></param>
		public void MinIndex(CudaDeviceVariable<byte> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Max
		/// <summary>
		/// Scratch-buffer size for Max.
		/// </summary>
		/// <returns></returns>
		public int MaxGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Max.nppiMaxGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		public void Max(CudaDeviceVariable<byte> max)
		{
			int bufferSize = MaxGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Max.nppiMax_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxGetBufferHostSize()"/></param>
		public void Max(CudaDeviceVariable<byte> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Max.nppiMax_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MaxIndex
		/// <summary>
		/// Scratch-buffer size for MaxIndex.
		/// </summary>
		/// <returns></returns>
		public int MaxIndexGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndxGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndxGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		public void MaxIndex(CudaDeviceVariable<byte> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MaxIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxIndexGetBufferHostSize()"/></param>
		public void MaxIndex(CudaDeviceVariable<byte> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMax
		/// <summary>
		/// Scratch-buffer size for MinMax.
		/// </summary>
		/// <returns></returns>
		public int MinMaxGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMaxGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		public void MinMax(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max)
		{
			int bufferSize = MinMaxGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(byte)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxGetBufferHostSize()"/></param>
		public void MinMax(CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMaxIndex
		/// <summary>
		/// Scratch-buffer size for MinMaxIndex.
		/// </summary>
		/// <returns></returns>
		public int MinMaxIndexGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_8u_C3CR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_8u_C3CR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for MinMaxIndex with mask.
		/// </summary>
		/// <returns></returns>
		public int MinMaxIndexMaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C3CR(_devPtrRoi, _pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C3CR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexGetBufferHostSize()"/></param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C3CR(_devPtrRoi, _pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C3CR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(byte)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexMaskedGetBufferHostSize()"/></param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<byte> min, CudaDeviceVariable<byte> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_8u_C3CMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Gamma
		/// <summary>
		/// image forward gamma correction.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Gamma(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.Gamma.nppiGammaFwd_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaFwd_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image forward gamma correction.
		/// </summary>
		public void Gamma()
		{
			status = NPPNativeMethods.NPPi.Gamma.nppiGammaFwd_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaFwd_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar color not in place forward gamma correction.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void Gamma(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.Gamma.nppiGammaFwd_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaFwd_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar color in place forward gamma correction.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		public static void Gamma(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.Gamma.nppiGammaFwd_8u_IP3R(arraySrc, src0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaFwd_8u_IP3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region GammaInv
		/// <summary>
		/// image inverse gamma correction.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void GammaInv(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.Gamma.nppiGammaInv_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaInv_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image inverse gamma correction.
		/// </summary>
		public void GammaInv()
		{
			status = NPPNativeMethods.NPPi.Gamma.nppiGammaInv_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaInv_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar color not in place inverse gamma correction.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void GammaInv(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.Gamma.nppiGammaInv_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaInv_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar color in place inverse gamma correction.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		public static void GammaInv(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.Gamma.nppiGammaInv_8u_IP3R(arraySrc, src0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGammaInv_8u_IP3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Color Space conversion new in CUDA 5
		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void YUVToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YUVToRGB.nppiYUVToRGB_8u_P3C3R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToRGB_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCrToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGB_8u_P3C3R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGB_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCrToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGR_8u_P3C3R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGR_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCrToBGR_709CSC(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGR_709CSC_8u_P3C3R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGR_709CSC_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed HLS color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void BGRToHLS(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.BGRToHLS.nppiBGRToHLS_8u_P3C3R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToHLS_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void HLSToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.HLSToBGR.nppiHLSToBGR_8u_P3C3R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHLSToBGR_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void RGBToYCbCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr422_8u_P3C2R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void RGBToYCrCb422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCrCb.nppiRGBToYCrCb422_8u_P3C2R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCrCb422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>         
		public static void YCbCrToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest, byte nAval)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGB_8u_P3C4R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGB_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>       
		public static void YCbCrToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest, byte nAval)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGR_8u_P3C4R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGR_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR_709CSC color conversion with constant alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>         
		public static void YCbCrToBGR_709CSC(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest, byte nAval)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGR_709CSC_8u_P3C4R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGR_709CSC_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>     
		public static void RGBToYUV(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV.nppiRGBToYUV_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>     
		public static void YUVToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YUVToRGB.nppiYUVToRGB_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToRGB_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar HLS color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>     
		public static void BGRToHLS(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.BGRToHLS.nppiBGRToHLS_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToHLS_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned planar BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>     
		public static void HLSToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.HLSToBGR.nppiHLSToBGR_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHLSToBGR_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>     
		public static void RGBToYUV422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayDestPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV422.nppiRGBToYUV422_8u_P3R(arraySrc, src0.Pitch, arrayDest, arrayDestPitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV422_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>     
		public static void RGBToYUV420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayDestPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV420.nppiRGBToYUV420_8u_P3R(arraySrc, src0.Pitch, arrayDest, arrayDestPitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV420_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422_8u_P3C2R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr422ToYCrCb422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCrCb422_8u_P3C2R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCrCb422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCrCb420ToYCbCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb420ToYCbCr422_8u_P3C2R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb420ToYCbCr422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCrCb420ToCbYCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb420ToCbYCr422_8u_P3C2R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb420ToCbYCr422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr411ToYCbCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr422_8u_P3C2R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr411ToYCrCb422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCrCb422_8u_P3C2R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCrCb422_8u_P3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YUV422ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToRGB_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToRGB_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YUV420ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToRGB.nppiYUV420ToRGB_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToRGB_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha (0xFF).
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YUV420ToRGBA(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToRGB.nppiYUV420ToRGB_8u_P3C4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToRGB_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YUV420ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToBGR.nppiYUV420ToBGR_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToBGR_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha (0xFF).
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YUV420ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToBGR.nppiYUV420ToBGR_8u_P3C4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToBGR_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr422ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr422ToRGB_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGB_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr422ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr422ToBGR_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr420ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr420ToBGR_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGR_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr420ToBGR_709CSC(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr420ToBGR_709CSC_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGR_709CSC_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YCbCr411ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr411ToBGR_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToBGR_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV422 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YUV422ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToRGB_8u_P3AC4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToRGB_8u_P3AC4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		public static void YUV420ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToRGB.nppiYUV420ToRGB_8u_P3AC4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToRGB_8u_P3AC4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCrCb420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		/// <param name="nAval">8-bit unsigned alpha constant.</param> 
		public static void YCrCb420ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest, byte nAval)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCrCbToRGB.nppiYCrCb420ToRGB_8u_P3C4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb420ToRGB_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		/// <param name="nAval">8-bit unsigned alpha constant.</param> 
		public static void YCbCr420ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest, byte nAval)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr420ToBGR_8u_P3C4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGR_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		/// <param name="nAval">8-bit unsigned alpha constant.</param> 
		public static void YCbCr420ToBGR_709HDTV(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest, byte nAval)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr420ToBGR_709HDTV_8u_P3C4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGR_709HDTV_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>     
		/// <param name="nAval">8-bit unsigned alpha constant.</param> 
		public static void YCbCr411ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC4 dest, byte nAval)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr411ToBGR_8u_P3C4R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, src0.SizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToBGR_8u_P3C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned planar RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void YUV422ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToRGB_8u_P3R(arraySrc, arrayPitchSrc, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToRGB_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned planar RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void YUV420ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToRGB.nppiYUV420ToRGB_8u_P3R(arraySrc, arrayPitchSrc, arrayDest, dest0.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToRGB_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCrCb420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void YCrCb420ToYCbCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitchDest = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb420ToYCbCr422_8u_P3R(arraySrc, arrayPitchSrc, arrayDest, arrayPitchDest, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb420ToYCbCr422_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void YCbCr411ToYCbCr422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitchDest = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr422_8u_P3R(arraySrc, arrayPitchSrc, arrayDest, arrayPitchDest, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr422_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb422 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void YCbCr411ToYCrCb422(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitchDest = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCrCb422_8u_P3R(arraySrc, arrayPitchSrc, arrayDest, arrayPitchDest, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCrCb422_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param>  
		/// <param name="dest2">Destination image channel 2</param>   
		public static void YCbCr411ToYCbCr420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitchDest = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr420_8u_P3R(arraySrc, arrayPitchSrc, arrayDest, arrayPitchDest, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr420_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel Y</param>  
		/// <param name="destCbCr">Destination image channel CbCr</param> 
		public static void YCbCr422ToYCbCr420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr420_8u_P3P2R(arraySrc, arrayPitchSrc, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr420_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel Y</param>  
		/// <param name="destCbCr">Destination image channel CbCr</param> 
		public static void YCbCr422ToYCbCr411(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr411_8u_P3P2R(arraySrc, arrayPitchSrc, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr411_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel Y</param>  
		/// <param name="destCbCr">Destination image channel CbCr</param> 
		public static void YCbCr420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420_8u_P3P2R(arraySrc, arrayPitchSrc, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel Y</param>  
		/// <param name="destCbCr">Destination image channel CbCr</param> 
		public static void YCrCb420ToYCbCr420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb420ToYCbCr420_8u_P3P2R(arraySrc, arrayPitchSrc, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb420ToYCbCr420_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel Y</param>  
		/// <param name="destCbCr">Destination image channel CbCr</param> 
		public static void YCrCb420ToYCbCr411(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb420ToYCbCr411_8u_P3P2R(arraySrc, arrayPitchSrc, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb420ToYCbCr411_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel Y</param>  
		/// <param name="destCbCr">Destination image channel CbCr</param> 
		public static void YCbCr411(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411_8u_P3P2R(arraySrc, arrayPitchSrc, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="destY">Destination image channel Y</param>  
		/// <param name="destCbCr">Destination image channel CbCr</param> 
		public static void YCbCr411ToYCbCr420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitchSrc = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr420_8u_P3P2R(arraySrc, arrayPitchSrc, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, src0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr420_8u_P3P2R", status));
			NPPException.CheckNppStatus(status, null);
		}





		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void RGBToYUV(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV.nppiRGBToYUV_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit packed YCbCr color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void RGBToYCbCr(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar HLS color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void BGRToHLS(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.BGRToHLS.nppiBGRToHLS_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToHLS_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned planar BGR color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void HLSToBGR(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.HLSToBGR.nppiHLSToBGR_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHLSToBGR_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void RGBToYUV422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV422.nppiRGBToYUV422_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV422_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void RGBToYUV420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV420.nppiRGBToYUV420_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV420_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void RGBToYCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr422_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr422_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void BGRToYCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCbCr.nppiBGRToYCbCr422_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr422_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void BGRToYCbCr420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCbCr.nppiBGRToYCbCr420_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr420_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void BGRToYCbCr420_709CSC(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCbCr.nppiBGRToYCbCr420_709CSC_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr420_709CSC_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void BGRToYCrCb420_709CSC(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCrCb.nppiBGRToYCrCb420_709CSC_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCrCb420_709CSC_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void BGRToYCrCb420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCrCb.nppiBGRToYCrCb420_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCrCb420_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>  
		/// <param name="dest1">Destination image channel 1</param> 
		/// <param name="dest2">Destination image channel 2</param>  
		public void BGRToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCbCr.nppiBGRToYCbCr411_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr411_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YUV422 color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToYUV422(NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV422.nppiRGBToYUV422_8u_C3C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV422_8u_C3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToYCrCb422(NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCrCb.nppiRGBToYCrCb422_8u_C3C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCrCb422_8u_C3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed YCbCr422 color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void BGRToYCbCr422(NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCbCr.nppiBGRToYCbCr422_8u_C3C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr422_8u_C3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToCbYCr422(NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToCbYCr.nppiRGBToCbYCr422_8u_C3C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToCbYCr422_8u_C3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB first gets forward gamma corrected then converted to 2 channel 8-bit unsigned packed CbYCr422 color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToCbYCr422Gamma(NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToCbYCr.nppiRGBToCbYCr422Gamma_8u_C3C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToCbYCr422Gamma_8u_C3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void BGRToCbYCr422_709HDTV(NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.BGRToCbYCr.nppiBGRToCbYCr422_709HDTV_8u_C3C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToCbYCr422_709HDTV_8u_C3C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YUV color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToYUV(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToYUV.nppiRGBToYUV_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYUV_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void YUVToRGB(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.YUVToRGB.nppiYUVToRGB_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToRGB_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed XYZ color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToXYZ(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToXYZ.nppiRGBToXYZ_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToXYZ_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed XYZ to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void XYZToRGB(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.XYZToRGB.nppiXYZToRGB_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXYZToRGB_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed LUV color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToLUV(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToLUV.nppiRGBToLUV_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToLUV_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed LUV to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void LUVToRGB(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.LUVToRGB.nppiLUVToRGB_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUVToRGB_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed Lab color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void BGRToLab(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.BGRToLab.nppiBGRToLab_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToLab_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed Lab to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void LabToBGR(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.LabToBGR.nppiLabToBGR_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabToBGR_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YCC color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToYCC(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCC.nppiRGBToYCC_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCC_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YCC to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void YCCToRGB(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCCToRGB.nppiYCCToRGB_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCCToRGB_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HLS color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToHLS(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToHLS.nppiRGBToHLS_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToHLS_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void HLSToRGB(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.HLSToRGB.nppiHLSToRGB_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHLSToRGB_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HSV color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void RGBToHSV(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.RGBToHSV.nppiRGBToHSV_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToHSV_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed HSV to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>  
		public void HSVToRGB(NPPImage_8uC3 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.HSVToRGB.nppiHSVToRGB_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHSVToRGB_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Histogram
		/// <summary>
		/// Scratch-buffer size for HistogramEven.
		/// </summary>
		/// <param name="nLevels"></param>
		/// <returns></returns>
		public int HistogramEvenGetBufferSize(int[] nLevels)
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEvenGetBufferSize_8u_C3R(_sizeRoi, nLevels, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEvenGetBufferSize_8u_C3R", status));
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
		/// <returns>An array of size nLevels which receives the levels being computed.</returns>
		public int[] EvenLevels(int nLevels, int nLowerBound, int nUpperBound)
		{
			int[] Levels = new int[nLevels];
			status = NPPNativeMethods.NPPi.Histogram.nppiEvenLevelsHost_32s(Levels, nLevels, nLowerBound, nUpperBound);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiEvenLevelsHost_32s", status));
			NPPException.CheckNppStatus(status, this);
			return Levels;
		}

		/// <summary>
		/// Histogram with evenly distributed bins. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="histogram">Allocated device memory of size nLevels (3 Variables)</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]. Size = 3</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]. Size = 3</param>
		public void HistogramEven(CudaDeviceVariable<int>[] histogram, int[] nLowerLevel, int[] nUpperLevel)
		{
			int[] size = new int[] { (int)histogram[0].Size + 1, (int)histogram[1].Size + 1, (int)histogram[2].Size + 1 };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };


			int bufferSize = HistogramEvenGetBufferSize(size);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEven_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, size, nLowerLevel, nUpperLevel, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with evenly distributed bins. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">Allocated device memory of size nLevels (3 Variables)</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]. Size = 3</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]. Size = 3</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramEvenGetBufferSize(int[])"/></param>
		public void HistogramEven(CudaDeviceVariable<int>[] histogram, int[] nLowerLevel, int[] nUpperLevel, CudaDeviceVariable<byte> buffer)
		{
			int[] size = new int[] { (int)histogram[0].Size + 1, (int)histogram[1].Size + 1, (int)histogram[2].Size + 1 };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };

			int bufferSize = HistogramEvenGetBufferSize(size);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEven_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, size, nLowerLevel, nUpperLevel, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Scratch-buffer size for HistogramRange.
		/// </summary>
		/// <param name="nLevels"></param>
		/// <returns></returns>
		public int HistogramRangeGetBufferSize(int[] nLevels)
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRangeGetBufferSize_8u_C3R(_sizeRoi, nLevels, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRangeGetBufferSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 3</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 3</param>
		public void HistogramRange(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels)
		{
			int[] size = new int[] { (int)histogram[0].Size, (int)histogram[1].Size, (int)histogram[2].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSize(size);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 3</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 3</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramRangeGetBufferSize(int[])"/></param>
		public void HistogramRange(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels, CudaDeviceVariable<byte> buffer)
		{
			int[] size = new int[] { (int)histogram[0].Size, (int)histogram[1].Size, (int)histogram[2].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSize(size);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//new in Cuda 5.5
		#region DotProduct
		/// <summary>
		/// Device scratch buffer size (in bytes) for nppiDotProd_8u64f_C3R.
		/// </summary>
		/// <returns></returns>
		public int DotProdGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.DotProd.nppiDotProdGetBufferHostSize_8u64f_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProdGetBufferHostSize_8u64f_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Three-channel 8-bit unsigned image DotProd.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="DotProdGetBufferHostSize()"/></param>
		public void DotProduct(NPPImage_8uC3 src2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = DotProdGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_8u64f_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_8u64f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 8-bit unsigned image DotProd. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (3 * sizeof(double))</param>
		public void DotProduct(NPPImage_8uC3 src2, CudaDeviceVariable<double> pDp)
		{
			int bufferSize = DotProdGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_8u64f_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_8u64f_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region LUT
		/// <summary>
		/// Three channel 8-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 8-bit unsigned destination output with alpha.
		/// The LUT is derived from a set of user defined mapping points in a palette and 
		/// source pixels are then processed using a restricted bit range when looking up palette values.
		/// This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nAlphaValue">Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.</param>
		/// <param name="pTables0">Host pointer to an array of 3 device memory pointers, channel 0, pointing to user defined OUTPUT palette values.
		/// <para/>Alpha values &lt; 0 or &gt; 255 will cause destination pixel alpha channel values to be unmodified.</param>
		/// <param name="pTables1">Host pointer to an array of 3 device memory pointers, channel 1, pointing to user defined OUTPUT palette values.
		/// <para/>Alpha values &lt; 0 or &gt; 255 will cause destination pixel alpha channel values to be unmodified.</param>
		/// <param name="pTables2">Host pointer to an array of 3 device memory pointers, channel 2, pointing to user defined OUTPUT palette values.
		/// <para/>Alpha values &lt; 0 or &gt; 255 will cause destination pixel alpha channel values to be unmodified.</param>
		/// <param name="nBitSize">Number of least significant bits (must be &gt; 0 and &lt;= 8) of each source pixel value to use as index into palette table during conversion.</param>
		public void LUTPaletteSwap(NPPImage_8uC4 dst, int nAlphaValue, CudaDeviceVariable<byte> pTables0, CudaDeviceVariable<byte> pTables1, CudaDeviceVariable<byte> pTables2, int nBitSize)
		{
			CUdeviceptr[] ptrs = new CUdeviceptr[] { pTables0.DevicePointer, pTables1.DevicePointer, pTables2.DevicePointer };
			status = NPPNativeMethods.NPPi.ColorLUTPalette.nppiLUTPaletteSwap_8u_C3A0C4R(_devPtrRoi, _pitch, nAlphaValue, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrs, nBitSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUTPaletteSwap_8u_C3A0C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUT(NPPImage_8uC3 dst, CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { (int)pLevels[0].Size, (int)pLevels[1].Size, (int)pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubic(NPPImage_8uC3 dst, CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { (int)pLevels[0].Size, (int)pLevels[1].Size, (int)pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// range restricted palette look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points in a palette and 
		/// source pixels are then processed using a restricted bit range when looking up palette values.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pTable">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.</param>
		/// <param name="nBitSize">Number of least significant bits (must be &gt; 0 and &lt;= 8) of each source pixel value to use as index into palette table during conversion.</param>
		public void LUTPalette(NPPImage_8uC3 dst, CudaDeviceVariable<byte>[] pTable, int nBitSize)
		{
			CUdeviceptr[] ptrsT = new CUdeviceptr[] { pTable[0].DevicePointer, pTable[1].DevicePointer, pTable[2].DevicePointer };
			status = NPPNativeMethods.NPPi.ColorLUTPalette.nppiLUTPalette_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsT, nBitSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUTPalette_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUT(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { (int)pLevels[0].Size, (int)pLevels[1].Size, (int)pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubic(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { (int)pLevels[0].Size, (int)pLevels[1].Size, (int)pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace linear interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTLinear(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { (int)pLevels[0].Size, (int)pLevels[1].Size, (int)pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUTLinear.nppiLUT_Linear_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Transpose
		/// <summary>
		/// image transpose
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Transpose(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.Transpose.nppiTranspose_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Color...New
		
		/// <summary>
		/// Swap color channels
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. <para/>The n-th entry of the array
		/// contains the number of the channel that is stored in the n-th channel of the output image. <para/>E.g.
		/// Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
		public void SwapChannels(NPPImage_8uC3 dest, int[] aDstOrder)
		{
			status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Swap color channels
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. <para/>The n-th entry of the array
		/// contains the number of the channel that is stored in the n-th channel of the output image. <para/>E.g.
		/// Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR channel order.</param>
		/// <param name="nValue">(V) Single channel constant value that can be replicated in one or more of the 4 destination channels.<para/>
		/// nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
		/// channel. <para/>An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
		/// particular destination channel value unmodified.</param>
		public void SwapChannels(NPPImage_8uC4 dest, int[] aDstOrder, byte nValue)
		{
			status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_8u_C3C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_8u_C3C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		
		/// <summary>
		/// Swap color channels inplace
		/// </summary>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. <para/>The n-th entry of the array
		/// contains the number of the channel that is stored in the n-th channel of the output image. <para/>E.g.
		/// Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
		public void SwapChannels(int[] aDstOrder)
		{
			status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, aDstOrder);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		
		/// <summary>
		/// RGB to Gray conversion
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void RGBToGray(NPPImage_8uC1 dest)
		{
			status = NPPNativeMethods.NPPi.RGBToGray.nppiRGBToGray_8u_C3C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToGray_8u_C3C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Color to Gray conversion
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
		public void ColorToGray(NPPImage_8uC1 dest, float[] aCoeffs)
		{
			status = NPPNativeMethods.NPPi.ColorToGray.nppiColorToGray_8u_C3C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aCoeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorToGray_8u_C3C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// in place color twist.
		/// 
		/// An input color twist matrix with floating-point coefficient values is applied
		/// within ROI.
		/// </summary>
		/// <param name="aTwist">The color twist matrix with floating-point coefficient values. [3,4]</param>
		public void ColorTwist(float[,] aTwist)
		{
			status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_8u_C3IR(_devPtr, _pitch, _sizeRoi, aTwist);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set (Array size = 3)</param>
		public void Set(byte[] nValue)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_8u_C3R(nValue, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Set pixel values to nValue. <para/>
		/// The 8-bit mask image affects setting of the respective pixels in the destination image. <para/>
		/// If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
		/// destination pixel is set to specified value.
		/// </summary>
		/// <param name="nValue">Value to be set (Array size = 3)</param>
		/// <param name="mask">Mask image</param>
		public void Set(byte[] nValue, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_8u_C3MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8u_C3MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Set pixel values to nValue. <para/>
		/// The 8-bit mask image affects setting of the respective pixels in the destination image. <para/>
		/// If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
		/// destination pixel is set to specified value.
		/// </summary>
		/// <param name="nValue">Value to be set</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		public void Set(byte nValue, int channel)
		{
			if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_8u_C3CR(nValue, _devPtrRoi + channel * _typeSize, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8u_C3CR", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion
		
		#region Copy

		/// <summary>
		/// image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Copy(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8u_C3R", status));
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
		public void Copy(NPPImage_8uC3 dst, int nTopBorderHeight, int nLeftBorderWidth, byte[] nValue)
		{
			status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// image copy with nearest source image pixel color.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nTopBorderHeight">Height (in pixels) of the top border. The height of the border at the bottom of
		/// the destination ROI is implicitly defined by the size of the source ROI: nBottomBorderHeight =
		/// oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
		/// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		public void CopyReplicateBorder(NPPImage_8uC3 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_8u_C3R", status));
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
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		public void CopyWrapBorder(NPPImage_8uC3 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// linearly interpolated source image subpixel coordinate color copy.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nDx">Fractional part of source image X coordinate.</param>
		/// <param name="nDy">Fractional part of source image Y coordinate.</param>
		public void CopySubpix(NPPImage_8uC3 dst, float nDx, float nDy)
		{
			status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MorphologicalNew
		/// <summary>
		/// Dilation computes the output pixel as the maximum pixel value of the pixels under the mask. Pixels who’s
		/// corresponding mask values are zero to not participate in the maximum search.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Mask">Pointer to the start address of the mask array.</param>
		/// <param name="aMaskSize">Width and Height mask array.</param>
		/// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
		public void Dilate(NPPImage_8uC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiDilate_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate_8u_C3R", status));
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
		public void Erode(NPPImage_8uC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiErode_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 3x3 dilation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Dilate3x3(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiDilate3x3_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Erode3x3(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiErode3x3_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3_8u_C3R", status));
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
		public void DilateBorder(NPPImage_8uC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.DilationWithBorderControl.nppiDilateBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilateBorder_8u_C3R", status));
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
		public void ErodeBorder(NPPImage_8uC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.ErosionWithBorderControl.nppiErodeBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErodeBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 dilation with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void Dilate3x3Border(NPPImage_8uC3 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.Dilate3x3Border.nppiDilate3x3Border_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3Border_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void Erode3x3Border(NPPImage_8uC3 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.Erode3x3Border.nppiErode3x3Border_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3Border_8u_C3R", status));
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
		public void Filter(NPPImage_8uC3 dest, CudaDeviceVariable<int> Kernel, NppiSize aKernelSize, NppiPoint oAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, aKernelSize, oAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter_8u_C3R", status));
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
		public void FilterColumn(NPPImage_8uC3 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumn_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn_8u_C3R", status));
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
		public void FilterRow(NPPImage_8uC3 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRow_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow_8u_C3R", status));
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
		public void FilterRowBorder(NPPImage_8uC3 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRowBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterBox(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBox_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBox_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMin(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMin_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMin_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMax(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMax_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMax_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 1D column convolution.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterColumn(NPPImage_8uC3 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumn32f_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, (int)pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn32f_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 1D row convolution.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterRow(NPPImage_8uC3 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRow32f_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, (int)pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow32f_8u_C3R", status));
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
		public void Filter(NPPImage_8uC3 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter32f_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_8u_C3R", status));
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
		public void Filter(NPPImage_16sC3 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter32f_8u16s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_8u16s_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittHoriz(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHoriz_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHoriz_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittVert(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVert_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVert_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void SobelHoriz(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelHoriz_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHoriz_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSobelVert(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelVert_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVert_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsDown(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDown_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDown_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter..
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsUp(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUp_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUp_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLaplace(NPPImage_8uC3 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplace_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplace_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Gauss filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterGauss(NPPImage_8uC3 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGauss_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGauss_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterHighPass(NPPImage_8uC3 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPass_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPass_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLowPass(NPPImage_8uC3 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPass_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPass_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSharpen(NPPImage_8uC3 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpen_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpen_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region NormNew
		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormDiffInfGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfGetBufferHostSize()"/></param>
		public void NormDiff_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L1.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL1GetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1GetBufferHostSize()"/></param>
		public void NormDiff_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L2.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL2GetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2GetBufferHostSize()"/></param>
		public void NormDiff_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormRelInfGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfGetBufferHostSize()"/></param>
		public void NormRel_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L1.
		/// </summary>
		/// <returns></returns>
		public int NormRelL1GetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1GetBufferHostSize()"/></param>
		public void NormRel_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L2.
		/// </summary>
		/// <returns></returns>
		public int NormRelL2GetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2GetBufferHostSize()"/></param>
		public void NormRel_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_8u_C3R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}





		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrFull_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int FullNormLevelGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_8u32f_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrFull_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="FullNormLevelGetBufferHostSize()"/></param>
		public void CrossCorrFull_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = FullNormLevelGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		public void CrossCorrFull_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			int bufferSize = FullNormLevelGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u32f_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrSame_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int SameNormLevelGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_8u32f_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrSame_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SameNormLevelGetBufferHostSize()"/></param>
		public void CrossCorrSame_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = SameNormLevelGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		public void CrossCorrSame_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			int bufferSize = SameNormLevelGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u32f_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrValid_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int ValidNormLevelGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_8u32f_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrValid_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="ValidNormLevelGetBufferHostSize()"/></param>
		public void CrossCorrValid_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		public void CrossCorrValid_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u32f_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}






		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrFull_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int FullNormLevelScaledGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_8u_C3RSfs(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_8u_C3RSfs", status));
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
		public void CrossCorrFull_NormLevel(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = FullNormLevelScaledGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void CrossCorrFull_NormLevel(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			int bufferSize = FullNormLevelScaledGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_8u_C3RSfs", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrSame_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int SameNormLevelScaledGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_8u_C3RSfs(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_8u_C3RSfs", status));
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
		public void CrossCorrSame_NormLevel(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = SameNormLevelScaledGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void CrossCorrSame_NormLevel(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			int bufferSize = SameNormLevelScaledGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_8u_C3RSfs", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// Device scratch buffer size (in bytes) for CrossCorrValid_NormLevel.
		/// </summary>
		/// <returns></returns>
		public int ValidNormLevelScaledGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_8u_C3RSfs(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_8u_C3RSfs", status));
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
		public void CrossCorrValid_NormLevel(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = ValidNormLevelScaledGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void CrossCorrValid_NormLevel(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			int bufferSize = ValidNormLevelScaledGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, nScaleFactor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_8u_C3RSfs", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}






		/// <summary>
		/// image SqrDistanceFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void SqrDistanceFull_Norm(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceFull_Norm_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceFull_Norm_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void SqrDistanceSame_Norm(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceSame_Norm_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceSame_Norm_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void SqrDistanceValid_Norm(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceValid_Norm_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceValid_Norm_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// image SqrDistanceFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void SqrDistanceFull_Norm(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceFull_Norm_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceFull_Norm_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void SqrDistanceSame_Norm(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceSame_Norm_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceSame_Norm_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void SqrDistanceValid_Norm(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceValid_Norm_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceValid_Norm_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// image CrossCorrFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void CrossCorrFull_Norm(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_Norm_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void CrossCorrSame_Norm(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_Norm_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void CrossCorrValid_Norm(NPPImage_8uC3 tpl, NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_Norm_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// image CrossCorrFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void CrossCorrFull_Norm(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_Norm_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void CrossCorrSame_Norm(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_Norm_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public void CrossCorrValid_Norm(NPPImage_8uC3 tpl, NPPImage_8uC3 dst, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_Norm_8u_C3RSfs(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_8u_C3RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region NormMaskedNew
		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormDiffInfMaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfMaskedGetBufferHostSize()"/></param>
		public void NormDiff_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		public void NormDiff_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L1.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL1MaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1MaskedGetBufferHostSize()"/></param>
		public void NormDiff_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		public void NormDiff_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L2.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL2MaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2MaskedGetBufferHostSize()"/></param>
		public void NormDiff_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		public void NormDiff_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_Inf.
		/// </summary>
		/// <returns></returns>
		public int NormRelInfMaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfMaskedGetBufferHostSize()"/></param>
		public void NormRel_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		public void NormRel_Inf(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L1.
		/// </summary>
		/// <returns></returns>
		public int NormRelL1MaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1MaskedGetBufferHostSize()"/></param>
		public void NormRel_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		public void NormRel_L1(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L2.
		/// </summary>
		/// <returns></returns>
		public int NormRelL2MaskedGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_8u_C3CMR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2MaskedGetBufferHostSize()"/></param>
		public void NormRel_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C3CMR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		public void NormRel_L2(NPPImage_8uC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_8u_C3CMR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_8u_C3CMR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}






		#endregion

		#region MinMaxEveryNew
		/// <summary>
		/// image MinEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MinEvery(NPPImage_8uC3 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMinEvery_8u_C3IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinEvery_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image MaxEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MaxEvery(NPPImage_8uC3 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMaxEvery_8u_C3IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxEvery_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MirrorNew


		/// <summary>
		/// Mirror image inplace.
		/// </summary>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		public void Mirror(NppiAxis flip)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_8u_C3IR(_devPtrRoi, _pitch, _sizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region CountInRange
		/// <summary>
		/// Device scratch buffer size (in bytes) for CountInRange.
		/// </summary>
		/// <returns></returns>
		public int CountInRangeGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.CountInRange.nppiCountInRangeGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRangeGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image CountInRange.
		/// </summary>
		/// <param name="pCounts">Pointer to the number of pixels that fall into the specified range. (3 * sizeof(int))</param>
		/// <param name="nLowerBound">Fixed size array of the lower bound of the specified range, one per channel.</param>
		/// <param name="nUpperBound">Fixed size array of the upper bound of the specified range, one per channel.</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="CountInRangeGetBufferHostSize()"/></param>
		public void CountInRange(CudaDeviceVariable<int> pCounts, byte[] nLowerBound, byte[] nUpperBound, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = CountInRangeGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.CountInRange.nppiCountInRange_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image CountInRange.
		/// </summary>
		/// <param name="pCounts">Pointer to the number of pixels that fall into the specified range. (3 * sizeof(int))</param>
		/// <param name="nLowerBound">Fixed size array of the lower bound of the specified range, one per channel.</param>
		/// <param name="nUpperBound">Fixed size array of the upper bound of the specified range, one per channel.</param>
		public void CountInRange(CudaDeviceVariable<int> pCounts, byte[] nLowerBound, byte[] nUpperBound)
		{
			int bufferSize = CountInRangeGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.CountInRange.nppiCountInRange_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region QualityIndex
		/// <summary>
		/// Device scratch buffer size (in bytes) for QualityIndex.
		/// </summary>
		/// <returns></returns>
		public int QualityIndexGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.QualityIndex.nppiQualityIndexGetBufferHostSize_8u32f_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndexGetBufferHostSize_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (3 * sizeof(float))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="QualityIndexGetBufferHostSize()"/></param>
		public void QualityIndex(NPPImage_8uC3 src2, CudaDeviceVariable<float> dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = QualityIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.QualityIndex.nppiQualityIndex_8u32f_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (3 * sizeof(float))</param>
		public void QualityIndex(NPPImage_8uC3 src2, CudaDeviceVariable<float> dst)
		{
			int bufferSize = QualityIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.QualityIndex.nppiQualityIndex_8u32f_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_8u32f_C3R", status));
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
		public void ResizeSqrPixel(NPPImage_8uC3 dst, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect dstRect = new NppiRect(dst.PointRoi, dst.SizeRoi);
			status = NPPNativeMethods.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_8u_C3R(_devPtr, _sizeRoi, _pitch, srcRect, dst.DevicePointer, dst.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image remap.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. </param>
		/// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. </param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public void Remap(NPPImage_8uC3 dst, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Remap.nppiRemap_8u_C3R(_devPtr, _sizeRoi, _pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Scale(NPPImage_16sC3 dst)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Scale.nppiScale_8u16s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u16s_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Scale(NPPImage_16uC3 dst)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Scale.nppiScale_8u16u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u16u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
		/// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
		public void Scale(NPPImage_32fC3 dst, float nMin, float nMax)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Scale.nppiScale_8u32f_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nMin, nMax);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Scale(NPPImage_32sC3 dst)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Scale.nppiScale_8u32s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_8u32s_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
        ///// <summary>
        ///// Resizes images.
        ///// </summary>
        ///// <param name="dest">Destination image</param>
        ///// <param name="xFactor">X scaling factor</param>
        ///// <param name="yFactor">Y scaling factor</param>
        ///// <param name="eInterpolation">Interpolation mode</param>
        //public void Resize(NPPImage_8uC3 dest, double xFactor, double yFactor, InterpolationMode eInterpolation)
        //{
        //	status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_8u_C3R(_devPtr, _sizeOriginal, _pitch, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, xFactor, yFactor, eInterpolation);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_8u_C3R", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        /// <summary>
        /// Resizes images.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="eInterpolation">Interpolation mode</param>
        public void Resize(NPPImage_8uC3 dest, InterpolationMode eInterpolation)
        {
            status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_8u_C3R(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointer, dest.Pitch, dest.Size, new NppiRect(dest.PointRoi, dest.SizeRoi), eInterpolation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        ///// <summary>
        ///// resizes planar images.
        ///// </summary>
        ///// <param name="src0">Source image (Channel 0)</param>
        ///// <param name="src1">Source image (Channel 1)</param>
        ///// <param name="src2">Source image (Channel 2)</param>
        ///// <param name="dest0">Destination image (Channel 0)</param>
        ///// <param name="dest1">Destination image (Channel 1)</param>
        ///// <param name="dest2">Destination image (Channel 2)</param>
        ///// <param name="xFactor">X scaling factor</param>
        ///// <param name="yFactor">Y scaling factor</param>
        ///// <param name="eInterpolation">Interpolation mode</param>
        //public static void Resize(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double xFactor, double yFactor, InterpolationMode eInterpolation)
        //{
        //	CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
        //	CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
        //	NppStatus status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_8u_P3R(src, src0.Size, src0.Pitch, new NppiRect(src0.PointRoi, src0.SizeRoi), dst, dest0.Pitch, dest0.SizeRoi, xFactor, yFactor, eInterpolation);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_8u_P3R", status));
        //	NPPException.CheckNppStatus(status, null);
        //}

        /// <summary>
        /// resizes planar images.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        /// <param name="eInterpolation">Interpolation mode</param>
        public static void Resize(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, InterpolationMode eInterpolation)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_8u_P3R(src, src0.Pitch, src0.Size, new NppiRect(src0.PointRoi, src0.SizeRoi), dst, dest0.Pitch, dest0.Size, new NppiRect(dest0.PointRoi, dest0.SizeRoi), eInterpolation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }



        /// <summary>
        /// planar image resize.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        /// <param name="nXFactor">Factor by which x dimension is changed. </param>
        /// <param name="nYFactor">Factor by which y dimension is changed. </param>
        /// <param name="nXShift">Source pixel shift in x-direction.</param>
        /// <param name="nYShift">Source pixel shift in y-direction.</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
        public static void ResizeSqrPixel(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };
			NppiRect srcRect = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect dstRect = new NppiRect(dest0.PointRoi, dest0.SizeRoi);
			NppStatus status = NPPNativeMethods.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_8u_P3R(src, src0.SizeRoi, src0.Pitch, srcRect, dst, dest0.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// planar image remap.
		/// </summary>
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. </param>
		/// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. </param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public static void Remap(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppiRect srcRect = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppStatus status = NPPNativeMethods.NPPi.Remap.nppiRemap_8u_P3R(src, src0.SizeRoi, src0.Pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst, dest0.Pitch, dest0.SizeRoi, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		#endregion

		#region Color Space conversion new in CUDA 6
		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed YUV color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void BGRToYUV(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.BGRToYUV.nppiBGRToYUV_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYUV_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YUV color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void BGRToYUV(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYUV.nppiBGRToYUV_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, dest0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYUV_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YUV color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void BGRToYUV(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYUV.nppiBGRToYUV_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYUV_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR color conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void YUVToBGR(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.YUVToBGR.nppiYUVToBGR_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToBGR_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YUVToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YUVToBGR.nppiYUVToBGR_8u_P3R(arraySrc, src0.Pitch, arrayDest, dest0.Pitch, dest0.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToBGR_8u_P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar BGR color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void YUVToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YUVToBGR.nppiYUVToBGR_8u_P3C3R(arraySrc, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToBGR_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}


		/// <summary>
		/// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void BGRToYCbCr(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.BGRToYCbCr.nppiBGRToYCbCr_8u_C3P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region SumWindow
		/// <summary>
		/// 8-bit unsigned 1D (column) sum to 32f.
		/// Apply Column Window Summation filter over a 1D mask region around each
		/// source pixel for 3-channel 8 bit/pixel input images with 32-bit floating point
		/// output.  <para/>
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring column pixel values in a mask region of the source image defined by
		/// nMaskSize and nAnchor. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void SumWindowColumn(NPPImage_32fC3 dest, int nMaskSize, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowColumn_8u32f_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowColumn_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 8-bit unsigned 1D (row) sum to 32f.<para/>
		/// Apply Row Window Summation filter over a 1D mask region around each source
		/// pixel for 3-channel 8-bit pixel input images with 32-bit floating point output.  
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring row pixel values in a mask region of the source image defined
		/// by nKernelDim and nAnchorX. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void SumWindowRow(NPPImage_32fC3 dest, int nMaskSize, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowRow_8u32f_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowRow_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterBorder
		/// <summary>
		/// Three channel 8-bit unsigned convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_8uC3 dest, CudaDeviceVariable<int> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder.nppiFilterBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Three channel 8-bit unsigned convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_8uC3 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder32f.nppiFilterBorder32f_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Three channel 8-bit unsigned to 16-bit signed convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_16sC3 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder32f.nppiFilterBorder32f_8u16s_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_8u16s_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterSobelBorder
		/// <summary>
		/// Filters the image using a horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelHorizBorder(NPPImage_8uC3 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelVertBorder(NPPImage_8uC3 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelVertBorder.nppiFilterSobelVertBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertBorder_8u_C3R", status));
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
		public void FilterMedian(NPPImage_8uC3 dst, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_8u_C3R", status));
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
		public void FilterMedian(NPPImage_8uC3 dst, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for FilterMedian.
		/// </summary>
		/// <returns></returns>
		public int FilterMedianGetBufferHostSize(NppiSize oMaskSize)
		{
			uint bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianGetBufferSize_8u_C3R(_sizeRoi, oMaskSize, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianGetBufferSize_8u_C3R", status));
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
		public void MaxError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
		public void MaxError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_8u_C3R", status));
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
		public void AverageError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
		public void AverageError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_8u_C3R", status));
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
		public void MaximumRelativeError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
		public void MaximumRelativeError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_8u_C3R", status));
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
		public void AverageRelativeError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C3R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
		public void AverageRelativeError(NPPImage_8uC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_8u_C3R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
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
		public void FilterGaussBorder(NPPImage_8uC3 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussBorder_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussBorder_8u_C3R", status));
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
		public void FilterColumnBorder(NPPImage_8uC3 dest, CudaDeviceVariable<int> Kernel, int nAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumnBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder_8u_C3R", status));
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
		public void FilterColumnBorder(NPPImage_8uC3 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumnBorder32f_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder32f_8u_C3R", status));
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
		public void FilterRowBorder(NPPImage_8uC3 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRowBorder32f_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder32f_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}


		#endregion

		#region SumWindow

		/// <summary>
		/// Apply Column Window Summation filter over a 1D mask region around each
		/// source pixel for 3-channel 8 bit/pixel input images with 32-bit floating point
		/// output.  
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring column pixel values in a mask region of the source image defined by
		/// nMaskSize and nAnchor. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void SumWindowColumnBorder(NPPImage_32fC3 dest, int nMaskSize, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowColumnBorder_8u32f_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowColumnBorder_8u32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Apply Row Window Summation filter over a 1D mask region around each source
		/// pixel for 3-channel 8-bit pixel input images with 32-bit floating point output.  
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring row pixel values in a mask region of the source image defined
		/// by nKernelDim and nAnchorX. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void SumWindowRowBorder(NPPImage_32fC3 dest, int nMaskSize, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowRowBorder_8u32f_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowRowBorder_8u32f_C3R", status));
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
		public void FilterBoxBorder(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorder_8u_C3R", status));
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
		public void FilterMinBorder(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMinBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMinBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterMaxBorder(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMaxBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMaxBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterOthers


		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittHorizBorder(NPPImage_8uC3 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHorizBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHorizBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittVertBorder(NPPImage_8uC3 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVertBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVertBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsDownBorder(NPPImage_8uC3 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDownBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDownBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsUpBorder(NPPImage_8uC3 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUpBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUpBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLaplaceBorder(NPPImage_8uC3 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplaceBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplaceBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterHighPassBorder(NPPImage_8uC3 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPassBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPassBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLowPassBorder(NPPImage_8uC3 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPassBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPassBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSharpenBorder(NPPImage_8uC3 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpenBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpenBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Filter Unsharp

		/// <summary>
		/// Scratch-buffer size for unsharp filter.
		/// </summary>
		/// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
		/// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
		/// <returns></returns>
		public int FilterUnsharpGetBufferSize(float nRadius, float nSigma)
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpGetBufferSize_8u_C3R(nRadius, nSigma, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpGetBufferSize_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

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
		public void FilterUnsharpBorder(NPPImage_8uC3 dst, float nRadius, float nSigma, float nWeight, float nThreshold, NppiBorderType eBorderType, CudaDeviceVariable<byte> buffer)
		{
			if (buffer.Size < FilterUnsharpGetBufferSize(nRadius, nSigma))
				throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpBorder_8u_C3R(_devPtr, _pitch, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nRadius, nSigma, nWeight, nThreshold, eBorderType, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Filter Gauss Advanced

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		public void FilterGauss(NPPImage_8uC3 dst, CudaDeviceVariable<float> Kernel)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGaussAdvanced_8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvanced_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterGaussBorder(NPPImage_8uC3 dst, CudaDeviceVariable<float> Kernel, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussAdvancedBorder_8u_C3R(_devPtrRoi, _pitch, _sizeRoi, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvancedBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		//New in Cuda 8.0
		#region ColorConversion new in Cuda 8.0


		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr411 color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void RGBToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr.nppiRGBToYCbCr411_8u_C3P3R(DevicePointerRoi, Pitch, arrayDest, arrayPitch, SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr411_8u_C3P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCr411ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
		{
			CUdeviceptr[] arraySrc = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			int[] arrayPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr411ToRGB_8u_P3C3R(arraySrc, arrayPitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToRGB_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region GradientColorToGray


		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray Gradient conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eNorm">Gradient distance method to use.</param>
		public void GradientColorToGray(NPPImage_8uC1 dest, NppiNorm eNorm)
		{
			NppStatus status = NPPNativeMethods.NPPi.GradientColorToGray.nppiGradientColorToGray_8u_C3C1R(DevicePointerRoi, Pitch, dest.DevicePointerRoi, dest.Pitch, SizeRoi, eNorm);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientColorToGray_8u_C3C1R", status));
			NPPException.CheckNppStatus(status, null);
		}

		#endregion

		#region FilterGaussAdvancedBorder


		/// <summary>
		/// Calculate destination image SizeROI width and height from source image ROI width and height and downsampling rate.
		/// It is highly recommended that this function be use to determine the destination image ROI for consistent results.
		/// </summary>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <returns>
		/// the destination image roi_specification.
		/// </returns>
		public NppiSize GetFilterGaussPyramidLayerDownBorderDstROI(float nRate)
		{
			NppiSize retSize = new NppiSize();
			status = NPPNativeMethods.NPPi.FilterGaussPyramid.nppiGetFilterGaussPyramidLayerDownBorderDstROI(_sizeRoi.width, _sizeRoi.height, ref retSize, nRate);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetFilterGaussPyramidLayerDownBorderDstROI", status));
			NPPException.CheckNppStatus(status, this);
			return retSize;
		}

		/// <summary>
		/// Calculate destination image SizeROI width and height from source image ROI width and height and downsampling rate.
		/// It is highly recommended that this function be use to determine the destination image ROI for consistent results.
		/// </summary>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <param name="pDstSizeROIMin">Minimum recommended destination image roi_specification.</param>
		/// <param name="pDstSizeROIMax">Maximum recommended destination image roi_specification.</param>
		public void GetFilterGaussPyramidLayerUpBorderDstROI(float nRate, out NppiSize pDstSizeROIMin, out NppiSize pDstSizeROIMax)
		{
			pDstSizeROIMin = new NppiSize();
			pDstSizeROIMax = new NppiSize();
			status = NPPNativeMethods.NPPi.FilterGaussPyramid.nppiGetFilterGaussPyramidLayerUpBorderDstROI(_sizeRoi.width, _sizeRoi.height, ref pDstSizeROIMin, ref pDstSizeROIMax, nRate);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGetFilterGaussPyramidLayerUpBorderDstROI", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three channel 8-bit unsigned Gauss filter with downsampling and border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <param name="nFilterTaps">The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. </param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterGaussPyramidLayerDownBorder(NPPImage_8uC3 dest, float nRate, int nFilterTaps, CudaDeviceVariable<float> pKernel, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussPyramid.nppiFilterGaussPyramidLayerDownBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nRate, nFilterTaps, pKernel.DevicePointer, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussPyramidLayerDownBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three channel 8-bit unsigned Gauss filter with downsampling and border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <param name="nFilterTaps">The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. </param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterGaussPyramidLayerUpBorder(NPPImage_8uC3 dest, float nRate, int nFilterTaps, CudaDeviceVariable<float> pKernel, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussPyramid.nppiFilterGaussPyramidLayerUpBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nRate, nFilterTaps, pKernel.DevicePointer, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussPyramidLayerUpBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region FilterBilateralGaussBorder


		/// <summary>
		/// Three channel 8-bit unsigned bilateral Gauss filter with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRadius">The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc. Radius values from 1 to 32 are supported.</param>
		/// <param name="nStepBetweenSrcPixels">The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. </param>
		/// <param name="nValSquareSigma">The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel and the source image pixel at the center of the filter kernel.</param>
		/// <param name="nPosSquareSigma">The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel and the source image pixel at the center of the filter kernel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterBilateralGaussBorder(NPPImage_8uC3 dest, int nRadius, int nStepBetweenSrcPixels, float nValSquareSigma, float nPosSquareSigma, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBilateralGaussBorder.nppiFilterBilateralGaussBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nRadius, nStepBetweenSrcPixels, nValSquareSigma, nPosSquareSigma, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBilateralGaussBorder_8u_C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region GradientVectorPrewittBorder

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
		/// and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
		/// </summary>
		/// <param name="destX">X vector destination_image_pointer</param>
		/// <param name="destY">Y vector destination_image_pointer.</param>
		/// <param name="destMag">magnitude destination_image_pointer.</param>
		/// <param name="destAngle">angle destination_image_pointer.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void GradientVectorPrewittBorder(NPPImage_16sC1 destX, NPPImage_16sC1 destY, NPPImage_16sC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.GradientVectorPrewittBorder.nppiGradientVectorPrewittBorder_8u16s_C3C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorPrewittBorder_8u16s_C3C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GradientVectorScharrBorder

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
		/// and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
		/// </summary>
		/// <param name="destX">X vector destination_image_pointer</param>
		/// <param name="destY">Y vector destination_image_pointer.</param>
		/// <param name="destMag">magnitude destination_image_pointer.</param>
		/// <param name="destAngle">angle destination_image_pointer.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void GradientVectorScharrBorder(NPPImage_16sC1 destX, NPPImage_16sC1 destY, NPPImage_16sC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.GradientVectorScharrBorder.nppiGradientVectorScharrBorder_8u16s_C3C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorScharrBorder_8u16s_C3C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GradientVectorSobelBorder

		/// <summary>
		/// 3 channel 8-bit unsigned packed RGB to optional 1 channel 16-bit signed X (vertical), Y (horizontal), magnitude, 
		/// and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
		/// </summary>
		/// <param name="destX">X vector destination_image_pointer</param>
		/// <param name="destY">Y vector destination_image_pointer.</param>
		/// <param name="destMag">magnitude destination_image_pointer.</param>
		/// <param name="destAngle">angle destination_image_pointer.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void GradientVectorSobelBorder(NPPImage_16sC1 destX, NPPImage_16sC1 destY, NPPImage_16sC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.GradientVectorSobelBorder.nppiGradientVectorSobelBorder_8u16s_C3C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorSobelBorder_8u16s_C3C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
        #endregion

        //New in Cuda 9.0
        #region New Cuda9
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void RGBToYCbCr420_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr420_JPEG_8u_P3R(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr420_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void RGBToYCbCr422_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr422_JPEG_8u_P3R(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr422_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void RGBToYCbCr411_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr411_JPEG_8u_P3R(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr411_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void RGBToYCbCr444_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr444_JPEG_8u_P3R(src, src0.Pitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr444_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }





        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void RGBToYCbCr420_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr420_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr420_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void RGBToYCbCr422_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr422_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr422_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void RGBToYCbCr411_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr411_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr411_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void RGBToYCbCr444_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiRGBToYCbCr444_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToYCbCr444_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }




        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void BGRToYCbCr420_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr420_JPEG_8u_P3R(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr420_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void BGRToYCbCr422_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr422_JPEG_8u_P3R(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr422_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void BGRToYCbCr411_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr411_JPEG_8u_P3R(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr411_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void BGRToYCbCr444_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr444_JPEG_8u_P3R(src, src0.Pitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr444_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }





        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void BGRToYCbCr420_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr420_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr420_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void BGRToYCbCr422_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr422_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr422_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void BGRToYCbCr411_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr411_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr411_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public void BGRToYCbCr444_JPEG(NPPImage_8uC3 src, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiBGRToYCbCr444_JPEG_8u_C3P3R(src.DevicePointerRoi, src.Pitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiBGRToYCbCr444_JPEG_8u_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }





        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr420ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr420ToRGB_JPEG_8u_P3R(src, srcPitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToRGB_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr422ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr422ToRGB_JPEG_8u_P3R(src, srcPitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGB_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr411ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr411ToRGB_JPEG_8u_P3R(src, srcPitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToRGB_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr444ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr444ToRGB_JPEG_8u_P3R(src, src0.Pitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr444ToRGB_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }




        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned packed YCbCr420 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr420ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr420ToRGB_JPEG_8u_P3C3R(src, srcPitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToRGB_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned packed YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr422ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr422ToRGB_JPEG_8u_P3C3R(src, srcPitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGB_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned packed YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr411ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr411ToRGB_JPEG_8u_P3C3R(src, srcPitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToRGB_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned packed YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr444ToRGB_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr444ToRGB_JPEG_8u_P3C3R(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr444ToRGB_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }







        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr420ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr420ToBGR_JPEG_8u_P3R(src, srcPitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGR_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr422ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr422ToBGR_JPEG_8u_P3R(src, srcPitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr411ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr411ToBGR_JPEG_8u_P3R(src, srcPitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToBGR_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void YCbCr444ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr444ToBGR_JPEG_8u_P3R(src, src0.Pitch, dst, dest0.Pitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr444ToBGR_JPEG_8u_P3R", status));
            NPPException.CheckNppStatus(status, null);
        }








        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed YCbCr420 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr420ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr420ToBGR_JPEG_8u_P3C3R(src, srcPitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGR_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr422ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr422ToBGR_JPEG_8u_P3C3R(src, srcPitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr411ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            int[] srcPitch = new int[] { src0.Pitch, src1.Pitch, src2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr411ToBGR_JPEG_8u_P3C3R(src, srcPitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToBGR_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed YCbCr422 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="dest">Destination image</param>
        public static void YCbCr444ToBGR_JPEG(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 src2, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.RGBToYCbCr_JPEG.nppiYCbCr444ToBGR_JPEG_8u_P3C3R(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr444ToBGR_JPEG_8u_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }




        /// <summary>
        /// Wiener filter with border control.
        /// </summary>
        /// <param name="dest">destination_image_pointer</param>
        /// <param name="oMaskSize">Pixel Width and Height of the rectangular region of interest surrounding the source pixel.</param>
        /// <param name="oAnchor">Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.</param>
        /// <param name="aNoise">Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void FilterWienerBorder(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, float[] aNoise, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.FilterWienerBorder.nppiFilterWienerBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, aNoise, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterWienerBorder_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }





        /// <summary>
        /// 3 channel 8-bit unsigned color per source image descriptor window location with source image border control 
        /// to per descriptor window destination floating point histogram of gradients. Requires first calling nppiHistogramOfGradientsBorderGetBufferSize function
        /// call to get required scratch (host) working buffer size and nppiHistogramOfGradientsBorderGetDescriptorsSize() function call to get
        /// total size for nLocations of output histogram block descriptor windows.
        /// </summary>
        /// <param name="hpLocations">Host pointer to array of NppiPoint source pixel starting locations of requested descriptor windows. Important: hpLocations is a </param>
        /// <param name="pDstWindowDescriptorBuffer">Output device memory buffer pointer of size hpDescriptorsSize bytes to first of nLoc descriptor windows (see nppiHistogramOfGradientsBorderGetDescriptorsSize() above).</param>
        /// <param name="oHOGConfig">Requested HOG configuration parameters structure.</param>
        /// <param name="pScratchBuffer">Device memory buffer pointer of size hpBufferSize bytes to scratch memory buffer (see nppiHistogramOfGradientsBorderGetBufferSize() above).</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void HistogramOfGradientsBorder(NppiPoint[] hpLocations, CudaDeviceVariable<byte> pDstWindowDescriptorBuffer, NppiHOGConfig oHOGConfig, CudaDeviceVariable<byte> pScratchBuffer, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.HistogramOfOrientedGradientsBorder.nppiHistogramOfGradientsBorder_8u32f_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, hpLocations, hpLocations.Length, pDstWindowDescriptorBuffer.DevicePointer, _sizeRoi, oHOGConfig, pScratchBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramOfGradientsBorder_8u32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region new in Cuda 9.1




        /// <summary>
        /// Calculate scratch buffer size needed for 3 channel 8-bit unsigned integer MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder, 
        /// MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
        /// </summary>
        /// <returns>Required buffer size in bytes.</returns>
        public int MorphGetBufferSize()
        {
            int ret = 0;
            status = NPPNativeMethods.NPPi.ComplexImageMorphology.nppiMorphGetBufferSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphGetBufferSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }




        /// <summary>
        /// 3 channel 8-bit unsigned integer morphological close with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void MorphCloseBorder(NPPImage_8uC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.ComplexImageMorphology.nppiMorphCloseBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphCloseBorder_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 3 channel 8-bit unsigned integer morphological open with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void MorphOpenBorder(NPPImage_8uC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.ComplexImageMorphology.nppiMorphOpenBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphOpenBorder_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 3 channel 8-bit unsigned integer morphological top hat with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void MorphTopHatBorder(NPPImage_8uC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.ComplexImageMorphology.nppiMorphTopHatBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphTopHatBorder_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 3 channel 8-bit unsigned integer morphological black hat with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void MorphBlackHatBorder(NPPImage_8uC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.ComplexImageMorphology.nppiMorphBlackHatBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphBlackHatBorder_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 3 channel 8-bit unsigned integer morphological gradient with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void MorphGradientBorder(NPPImage_8uC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.ComplexImageMorphology.nppiMorphGradientBorder_8u_C3R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphGradientBorder_8u_C3R", status));
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
        public static void ResizeBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiResizeBatchCXR> pBatchList)
        {
            NppStatus status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResizeBatch_8u_C3R(oSmallestSrcSize, oSrcRectROI, oSmallestDstSize, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeBatch_8u_C3R", status));
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
        public static void ResizeBatchAdvanced(int nMaxWidth, int nMaxHeight, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc, CudaDeviceVariable<NppiImageDescriptor> pBatchDst,
                                        CudaDeviceVariable<NppiResizeBatchROI_Advanced> pBatchROI, uint nBatchSize, InterpolationMode eInterpolation)
        {
            NppStatus status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResizeBatch_8u_C3R_Advanced(nMaxWidth, nMaxHeight, pBatchSrc.DevicePointer, pBatchDst.DevicePointer,
                pBatchROI.DevicePointer, pBatchDst.Size, eInterpolation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeBatch_8u_C3R_Advanced", status));
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
		public static void WarpAffineBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiWarpAffineBatchCXR> pBatchList)
		{
			NppStatus status = NPPNativeMethods.NPPi.GeometricTransforms.nppiWarpAffineBatch_8u_C3R(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBatch_8u_C3R", status));
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
		public static void WarpPerspectiveBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiWarpAffineBatchCXR> pBatchList)
		{
			NppStatus status = NPPNativeMethods.NPPi.GeometricTransforms.nppiWarpPerspectiveBatch_8u_C3R(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBatch_8u_C3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
		/// arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUVToRGBBatch(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YUVToRGB.nppiYUVToRGBBatch_8u_C3R(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToRGBBatch_8u_C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
		/// arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUVToBGRBatch(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YUVToBGR.nppiYUVToBGRBatch_8u_C3R(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToBGRBatch_8u_C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair of input/output images has own ROI.
		/// Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList.API user must ensure that
		/// ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCrToRGBBatch(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGBBatch_8u_C3R(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGBBatch_8u_C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input and output images passed in pSrcBatchList and pSrcBatchList
		/// arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCrToBGRBatch(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGRBatch_8u_C3R(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGRBatch_8u_C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair of input/output images has own ROI.
		/// Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList.API user must ensure that
		/// ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUVToRGBBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YUVToRGB.nppiYUVToRGBBatch_8u_C3R_Advanced(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToRGBBatch_8u_C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair of input/output images has own ROI.
		/// Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList.API user must ensure that
		/// ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCrToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGRBatch_8u_C3R_Advanced(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGRBatch_8u_C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair of input/output images has own ROI.
		/// Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList.API user must ensure that
		/// ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUVToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YUVToBGR.nppiYUVToBGRBatch_8u_C3R_Advanced(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToBGRBatch_8u_C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair of input/output images has own ROI.
		/// Provided oMaxSizeROI must contain the maximum width and the maximum height of all ROIs defined in pDstBatchList.API user must ensure that
		/// ROI from pDstBatchList for each pair of input and output images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">source_batch_images_pointer</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCrToRGBBatch_8u_C3R_Advanced(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGBBatch_8u_C3R_Advanced(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGBBatch_8u_C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUVToRGBBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUVToRGB.nppiYUVToRGBBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToRGBBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUVToBGRBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUVToBGR.nppiYUVToBGRBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToBGRBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUV422ToRGBBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToRGBBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToRGBBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUV422ToBGRBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToBGRBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToBGRBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUV420ToRGBBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToRGB.nppiYUV420ToRGBBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToRGBBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YUV420ToBGRBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToBGR.nppiYUV420ToBGRBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToBGRBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCrToRGBBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGBBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGBBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCrToBGRBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGRBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGRBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCr422ToRGBBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr422ToRGBBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGBBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCr422ToBGRBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr422ToBGRBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGRBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.	
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCr420ToRGBBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr420ToRGBBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToRGBBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR batch color conversion for a single ROI.
		/// Provided oSizeROI will be used for all pairs of input planes making input images and output packed images passed in
		/// pSrcBatchList and pSrcBatchList arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the
		/// borders of any of provided images.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oSizeROI">Region-of-Interest (ROI).</param>
		public static void YCbCr420ToBGRBatch(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr420ToBGRBatch_8u_P3C3R(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGRBatch_8u_P3C3R", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}




		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUVToRGBBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUVToRGB.nppiYUVToRGBBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToRGBBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUVToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUVToBGR.nppiYUVToBGRBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUVToBGRBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUV422ToRGBBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToRGBBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToRGBBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUV422ToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToBGRBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToBGRBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUV420ToRGBBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToRGB.nppiYUV420ToRGBBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToRGBBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YUV420ToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YUV420ToBGR.nppiYUV420ToBGRBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV420ToBGRBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCrToRGBBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCrToRGBBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToRGBBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCrToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCrToBGRBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCrToBGRBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCr422ToRGBBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCr422ToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCr420ToRGBBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
		}

		/// <summary>
		/// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR batch color conversion where each pair
		/// of input/output images has own ROI.Provided oMaxSizeROI must contain the maximum width and the maximum height of all
		/// ROIs defined in pDstBatchList.API user must ensure that ROI from pDstBatchList for each pair of input and output
		/// images does not go beyond the borders of images in each pair.
		/// </summary>
		/// <param name="pSrcBatchList">An array where each element is a batch of images representing one of planes in planar images, 
		/// \ref source_batch_images_pointer.The first element of array (pSrcBatchList[0]) represents a batch of Y planes. 
		/// The second element of array (pSrcBatchList[1]) represents a batch of Cb planes.The third element of array
		///  (pSrcBatchList[2]) represents a batch of Cr planes.</param>
		/// <param name="pDstBatchList">destination_batch_images_pointer</param>
		/// <param name="oMaxSizeROI">Region-of-Interest (ROI), must contain the maximum width and the maximum height from all destination ROIs used for processing data.</param>
		public static void YCbCr420ToBGRBatchAdvanced(CudaDeviceVariable<NppiImageDescriptor>[] pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList, NppiSize oMaxSizeROI)
		{
			CUdeviceptr[] srcList = new CUdeviceptr[] { pSrcBatchList[0].DevicePointer, pSrcBatchList[1].DevicePointer, pSrcBatchList[2].DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced(srcList, pDstBatchList.DevicePointer, pDstBatchList.Size, oMaxSizeROI);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced", status));
			NPPException.CheckNppStatus(status, pSrcBatchList);
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
		public static void ColorTwistBatch(float nMin, float nMax, NppiSize oSizeROI, CudaDeviceVariable<NppiColorTwistBatchCXR> pBatchList)
		{
			NppStatus status = NPPNativeMethods.NPPi.ColorTwistBatch.nppiColorTwistBatch32f_8u_C3R(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch32f_8u_C3R", status));
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
		public static void ColorTwistBatchI(float nMin, float nMax, NppiSize oSizeROI, CudaDeviceVariable<NppiColorTwistBatchCXR> pBatchList)
		{
			NppStatus status = NPPNativeMethods.NPPi.ColorTwistBatch.nppiColorTwistBatch32f_8u_C3IR(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch32f_8u_C3IR", status));
			NPPException.CheckNppStatus(status, pBatchList);
		}
		#endregion

		#region New in Cuda 11.1


		/// <summary>
		/// in place flood fill.
		/// </summary>
		/// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
		/// <param name="aNewValues">Image pixel values to be used to replace matching pixels.</param>
		/// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
		/// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
		public NppiConnectedRegion FloodFill(NppiPoint oSeed, byte[] aNewValues, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
		{
			NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
			status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFill_8u_C3IR(_devPtrRoi, _pitch, oSeed, aNewValues, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFill_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
			return pConnectedRegion;
		}

		/// <summary>
		/// in place flood fill.
		/// </summary>
		/// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
		/// <param name="aNewValues">Image pixel values to be used to replace matching pixels.</param>
		/// <param name="nBoundaryValue">Image pixel values to be used for region boundary. </param>
		/// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
		/// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
		public NppiConnectedRegion FloodFill(NppiPoint oSeed, byte aNewValues, byte nBoundaryValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
		{
			NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
			status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillBoundary_8u_C3IR(_devPtrRoi, _pitch, oSeed, aNewValues, nBoundaryValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillBoundary_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
			return pConnectedRegion;
		}

		/// <summary>
		/// in place flood fill.
		/// </summary>
		/// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
		/// <param name="aMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
		/// <param name="aMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
		/// <param name="aNewValues">Image pixel values to be used to replace matching pixels.</param>
		/// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
		/// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
		public NppiConnectedRegion FloodFill(NppiPoint oSeed, byte[] aMin, byte[] aMax, byte[] aNewValues, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
		{
			NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
			status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillRange_8u_C3IR(_devPtrRoi, _pitch, oSeed, aMin, aMax, aNewValues, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillRange_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
			return pConnectedRegion;
		}

		/// <summary>
		/// in place flood fill.
		/// </summary>
		/// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
		/// <param name="aMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
		/// <param name="aMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
		/// <param name="aNewValues">Image pixel values to be used to replace matching pixels.</param>
		/// <param name="nBoundaryValue">Image pixel values to be used for region boundary. </param>
		/// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
		/// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
		public NppiConnectedRegion FloodFill(NppiPoint oSeed, byte[] aMin, byte[] aMax, byte[] aNewValues, byte[] nBoundaryValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
		{
			NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
			status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillRangeBoundary_8u_C3IR(_devPtrRoi, _pitch, oSeed, aMin, aMax, aNewValues, nBoundaryValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillRangeBoundary_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
			return pConnectedRegion;
		}





		/// <summary>
		/// in place flood fill.
		/// </summary>
		/// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
		/// <param name="aMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
		/// <param name="aMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
		/// <param name="aNewValues">Image pixel values to be used to replace matching pixels.</param>
		/// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
		/// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
		public NppiConnectedRegion FloodFillGradient(NppiPoint oSeed, byte[] aMin, byte[] aMax, byte[] aNewValues, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
		{
			NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
			status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillGradient_8u_C3IR(_devPtrRoi, _pitch, oSeed, aMin, aMax, aNewValues, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillGradient_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
			return pConnectedRegion;
		}
		/// <summary>
		/// in place flood fill.
		/// </summary>
		/// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
		/// <param name="aMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
		/// <param name="aMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
		/// <param name="aNewValues">Image pixel values to be used to replace matching pixels.</param>
		/// <param name="aBoundaryValues">Image pixel values to be used for region boundary. </param>
		/// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
		/// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
		public NppiConnectedRegion FloodFillGradient(NppiPoint oSeed, byte[] aMin, byte[] aMax, byte[] aNewValues, byte[] aBoundaryValues, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
		{
			NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
			status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillGradientBoundary_8u_C3IR(_devPtrRoi, _pitch, oSeed, aMin, aMax, aNewValues, aBoundaryValues, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillGradientBoundary_8u_C3IR", status));
			NPPException.CheckNppStatus(status, this);
			return pConnectedRegion;
		}




		#endregion
	}
}
