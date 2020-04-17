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
	public partial class NPPImage_8uC2 : NPPImageBase
	{
		#region Constructors
		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="nWidthPixels">Image width in pixels</param>
		/// <param name="nHeightPixels">Image height in pixels</param>
		public NPPImage_8uC2(int nWidthPixels, int nHeightPixels)
		{
			_sizeOriginal.width = nWidthPixels;
			_sizeOriginal.height = nHeightPixels;
			_sizeRoi.width = nWidthPixels;
			_sizeRoi.height = nHeightPixels;
			_channels = 1;
			_isOwner = true;
			_typeSize = sizeof(byte);

			_devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_8u_C2(nWidthPixels, nHeightPixels, ref _pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_8u_C2", res, _pitch, _channels));
			
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
		public NPPImage_8uC2(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
		{
			_devPtr = devPtr;
			_devPtrRoi = _devPtr;
			_sizeOriginal.width = width;
			_sizeOriginal.height = height;
			_sizeRoi.width = width;
			_sizeRoi.height = height;
			_pitch = pitch;
			_channels = 1;
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
		public NPPImage_8uC2(CUdeviceptr devPtr, int width, int height, int pitch)
			: this(devPtr, width, height, pitch, false)
		{

		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
		/// </summary>
		/// <param name="image">NPP image</param>
		public NPPImage_8uC2(NPPImageBase image)
			: this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
		{

		}

		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="size">Image size</param>
		public NPPImage_8uC2(NppiSize size)
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
		public NPPImage_8uC2(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
			: this(devPtr, size.width, size.height, pitch, isOwner)
		{ 
			
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="size">Image size</param>
		/// <param name="pitch">Pitch / Line step</param>
		public NPPImage_8uC2(CUdeviceptr devPtr, NppiSize size, int pitch)
			: this(devPtr, size.width, size.height, pitch)
		{

		}

		/// <summary>
		/// For dispose
		/// </summary>
		~NPPImage_8uC2()
		{
			Dispose (false);
		}
		#endregion

		#region Converter operators

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		public CudaPitchedDeviceVariable<VectorTypes.uchar2> ToCudaPitchedDeviceVariable()
		{
			return new CudaPitchedDeviceVariable<VectorTypes.uchar2>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
		}

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		/// <param name="img">NPPImage</param>
		/// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
		public static implicit operator CudaPitchedDeviceVariable<VectorTypes.uchar2>(NPPImage_8uC2 img)
		{
			return img.ToCudaPitchedDeviceVariable();
		}

		/// <summary>
		/// Converts a CudaPitchedDeviceVariable to a NPPImage 
		/// </summary>
		/// <param name="img">CudaPitchedDeviceVariable</param>
		/// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
		public static implicit operator NPPImage_8uC2(CudaPitchedDeviceVariable<VectorTypes.uchar2> img)
		{
			return img.ToNPPImage();
		}
		#endregion

		#region Color space conversion
		/// <summary>
		/// 2 channel 8-bit unsigned YCbCr422 to 3 channel packed RGB color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">estination image</param>
		public void YCbCr422ToRGB(NPPImage_8uC3 dest)
		{ 
			status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr422ToRGB_8u_C2C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGB_8u_C2C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image composition using image alpha values (0 - max channel pixel value).<para/>
		/// Also the function is called *AC1R, it is a two channel image with second channel as alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppAlphaOp">alpha compositing operation</param>
		public void AlphaComp(NPPImage_8uC2 src2, NPPImage_8uC2 dest, NppiAlphaOp nppAlphaOp)
		{
			status = NPPNativeMethods.NPPi.AlphaComp.nppiAlphaComp_8u_AC1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaComp_8u_AC1R", status));
			NPPException.CheckNppStatus(status, this);
		}
				
		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr420ToYCbCr411(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCbCr411_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr411_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr420ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCbCr422_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr422_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCr420ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCbCr422_8u_P2C2R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr422_8u_P2C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCr420ToCbYCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToCbYCr422_8u_P2C2R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToCbYCr422_8u_P2C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr420ToYCrCb420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCrCb420_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCrCb420_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr411(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr411ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr422_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr422_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest">Destination image</param>
		public static void YCbCr411ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC2 dest)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr422_8u_P2C2R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr422_8u_P2C2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr411ToYCbCr420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr420_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr420_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public static void YCbCr411ToYCrCb420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCrCb420_8u_P2P3R(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCrCb420_8u_P2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void CbYCr422ToYCbCr422(NPPImage_8uC2 dest)
		{
			status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr422_8u_C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr422_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void YCbCr422ToYCrCb422(NPPImage_8uC2 dest)
		{
			status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCrCb422_8u_C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCrCb422_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void YCbCr422ToCbYCr422(NPPImage_8uC2 dest)
		{
			status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToCbYCr422_8u_C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToCbYCr422_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void YCrCb422ToRGB(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCrCb422ToRGB_8u_C2C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToRGB_8u_C2C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed BGR color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void YCbCr422ToBGR(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr422ToBGR_8u_C2C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_8u_C2C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCrC22 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void CbYCr422ToRGB(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.CbYCrToRGB.nppiCbYCr422ToRGB_8u_C2C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToRGB_8u_C2C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void CbYCr422ToBGR_709HDTV(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.CbYCrToBGR.nppiCbYCr422ToBGR_709HDTV_8u_C2C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToBGR_709HDTV_8u_C2C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void YUV422ToRGB(NPPImage_8uC3 dest)
		{
			status = NPPNativeMethods.NPPi.YUV422ToRGB.nppiYUV422ToRGB_8u_C2C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToRGB_8u_C2C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar RGB color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCrCb422ToRGB(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCrCb422ToRGB_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToRGB_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCbCr422ToRGB(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToRGB.nppiYCbCr422ToRGB_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGB_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void CbYCr422ToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr411_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr411_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCbCr422ToYCbCr420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr420_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr420_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCbCr422ToYCrCb420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCrCb420_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCrCb420_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCbCr422ToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr411_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr411_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCrCb422ToYCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb422ToYCbCr422_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToYCbCr422_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCrCb422ToYCbCr420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb422ToYCbCr420_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToYCbCr420_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void YCrCb422ToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb422ToYCbCr411_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToYCbCr411_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void CbYCr422ToYCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr422_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr422_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void CbYCr422ToYCbCr420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr420_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr420_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		public void CbYCr422ToYCrCb420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCrCb420_8u_C2P3R(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCrCb420_8u_C2P3R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="destY">Destination image channel 0</param>
		/// <param name="destCbCr">Destination image channel 1</param>
		public void YCbCr422ToYCbCr420(NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr420_8u_C2P2R(_devPtrRoi, _pitch, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr420_8u_C2P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="destY">Destination image channel 0</param>
		/// <param name="destCbCr">Destination image channel 1</param>
		public void YCbCr422ToYCbCr411(NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr411_8u_C2P2R(_devPtrRoi, _pitch, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr411_8u_C2P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="destY">Destination image channel 0</param>
		/// <param name="destCbCr">Destination image channel 1</param>
		public void CbYCr422ToYCbCr420(NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr420_8u_C2P2R(_devPtrRoi, _pitch, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr420_8u_C2P2R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>
		public void CbYCr422ToYCbCr420(NPPImage_8uC4 dest, byte nAval)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr422ToBGR_8u_C2C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_8u_C2C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>
		public void YCbCr422ToBGR(NPPImage_8uC4 dest, byte nAval)
		{
			NppStatus status = NPPNativeMethods.NPPi.YCbCrToBGR.nppiYCbCr422ToBGR_8u_C2C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_8u_C2C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>
		public void CbYCr422ToBGR_709HDTV(NPPImage_8uC4 dest, byte nAval)
		{
			NppStatus status = NPPNativeMethods.NPPi.CbYCrToBGR.nppiCbYCr422ToBGR_709HDTV_8u_C2C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAval);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToBGR_709HDTV_8u_C2C4R", status));
			NPPException.CheckNppStatus(status, null);
		}



		/// <summary>
		/// 2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed ARGB color conversion with constant alpha (0xFF).
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcVU">Source image channel VU</param>
		/// <param name="dest">Destination image</param>
		public static void NV21ToRGB(NPPImage_8uC1 srcY, NPPImage_8uC1 srcVU, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] devptrs = new CUdeviceptr[] { srcY.DevicePointer, srcVU.DevicePointer};
			NppStatus status = NPPNativeMethods.NPPi.NV21ToRGB.nppiNV21ToRGB_8u_P2C4R(devptrs, srcY.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV21ToRGB_8u_P2C4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed BGRA color conversion with constant alpha (0xFF).
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcVU">Source image channel VU</param>
		/// <param name="dest">Destination image</param>
		public static void NV21ToBGR(NPPImage_8uC1 srcY, NPPImage_8uC1 srcVU, NPPImage_8uC2 dest)
		{
			CUdeviceptr[] devptrs = new CUdeviceptr[] { srcY.DevicePointer, srcVU.DevicePointer };
			NppStatus status = NPPNativeMethods.NPPi.NV21ToBGR.nppiNV21ToBGR_8u_P2C4R(devptrs, srcY.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV21ToBGR_8u_P2C4R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set</param>
		public void Set(byte[] nValue)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_8u_C2R(nValue, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MaxError
		/// <summary>
		/// image maximum error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		public void MaxError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C2R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
		public void MaxError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_8u_C2R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_8u_C2R", status));
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
		public void AverageError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C2R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
		public void AverageError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_8u_C2R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_8u_C2R", status));
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
		public void MaximumRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C2R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
		public void MaximumRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_8u_C2R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_8u_C2R", status));
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
		public void AverageRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C2R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
		public void AverageRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_8u_C2R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion


		#region ColorTwist
		/// <summary>
		/// An input color twist matrix with floating-point pixel values is applied
		/// within ROI.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
		public void ColorTwist(NPPImage_8uC2 dest, float[,] twistMatrix)
		{
			status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist32f_8u_C2R(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C2R", status));
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
			status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist32f_8u_C2IR(_devPtr, _pitch, _sizeRoi, aTwist);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C2IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion



		#region FilterBorder
		/// <summary>
		/// Two channel 8-bit unsigned convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_8uC2 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder32f.nppiFilterBorder32f_8u_C2R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Filter

		/// <summary>
		/// convolution filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array.<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference</param>
		public void Filter(NPPImage_8uC2 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter32f_8u_C2R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_8u_C2R", status));
			NPPException.CheckNppStatus(status, this);
		}
        #endregion

        #region ColorConversion new in Cuda 9

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned planar YUV420 color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        public static void NV12ToYUV420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods.NPPi.NV12ToYUV420.nppiNV12ToYUV420_8u_P2P3R(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToYUV420_8u_P2P3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
        public static void NV12ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.NV12ToRGB.nppiNV12ToRGB_8u_P2C3R(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToRGB_8u_P2C3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB 709 HDTV full color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
        public static void NV12ToRGB_709HDTV(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.NV12ToRGB.nppiNV12ToRGB_709HDTV_8u_P2C3R(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToRGB_709HDTV_8u_P2C3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
        public static void NV12ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.NV12ToBGR.nppiNV12ToBGR_8u_P2C3R(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToBGR_8u_P2C3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed BGR 709 HDTV full color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
        public static void NV12ToBGR_709HDTV(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.NV12ToBGR.nppiNV12ToBGR_709HDTV_8u_P2C3R(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToBGR_709HDTV_8u_P2C3R", status));
            NPPException.CheckNppStatus(status, null);
        }
        #endregion
    }
}
