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
		#region Color space conversion
		/// <summary>
		/// 2 channel 8-bit unsigned YCbCr422 to 3 channel packed RGB color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">estination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToRGB(NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
		{ 
			status = NPPNativeMethods_Ctx.NPPi.YCbCrToRGB.nppiYCbCr422ToRGB_8u_C2C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGB_8u_C2C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image composition using image alpha values (0 - max channel pixel value).<para/>
		/// Also the function is called *AC1R, it is a two channel image with second channel as alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppAlphaOp">alpha compositing operation</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AlphaComp(NPPImage_8uC2 src2, NPPImage_8uC2 dest, NppiAlphaOp nppAlphaOp, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AlphaComp.nppiAlphaComp_8u_AC1R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaComp_8u_AC1R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr420ToYCbCr411(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420_8u_P2P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr420ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr420ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr420ToCbYCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr420ToYCrCb420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr411(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411_8u_P2P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr411ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr422_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr422_8u_P2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcCbCr">Source image channel CbCr</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr411ToYCbCr422(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr411ToYCbCr420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCbCr420_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCbCr420_8u_P2P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void YCbCr411ToYCrCb420(NPPImage_8uC1 srcY, NPPImage_8uC1 srcCbCr, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx(srcY.DevicePointerRoi, srcY.Pitch, srcCbCr.DevicePointerRoi, srcCbCr.Pitch, arrayDest, arrayPitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToYCbCr422(NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr422_8u_C2R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr422_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToYCrCb422(NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCrCb422_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToCbYCr422(NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToCbYCr422_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCrCb422ToRGB(NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.YCbCrToRGB.nppiYCrCb422ToRGB_8u_C2C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToRGB_8u_C2C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed BGR color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToBGR(NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.YCbCrToBGR.nppiYCbCr422ToBGR_8u_C2C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_8u_C2C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCrC22 to 3 channel 8-bit unsigned packed RGB color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToRGB(NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CbYCrToRGB.nppiCbYCr422ToRGB_8u_C2C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToRGB_8u_C2C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToBGR_709HDTV(NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CbYCrToBGR.nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion.
		/// images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YUV422ToRGB(NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.YUV422ToRGB.nppiYUV422ToRGB_8u_C2C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYUV422ToRGB_8u_C2C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar RGB color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCrCb422ToRGB(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrToRGB.nppiYCrCb422ToRGB_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToRGB_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToRGB(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrToRGB.nppiYCbCr422ToRGB_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, dest0.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToRGB_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr411_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr411_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToYCbCr420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr420_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr420_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToYCrCb420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCrCb420_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCrCb420_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCrCb422ToYCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCrCb422ToYCbCr420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCrCb422ToYCbCr411(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCrCb422ToYCbCr411_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCrCb422ToYCbCr411_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToYCbCr422(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr422_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr422_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToYCbCr420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
		/// </summary>
		/// <param name="dest0">Destination image channel 0</param>
		/// <param name="dest1">Destination image channel 1</param>
		/// <param name="dest2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToYCrCb420(NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] arrayDest = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			int[] arrayPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCrCb420_8u_C2P3R_Ctx(_devPtrRoi, _pitch, arrayDest, arrayPitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCrCb420_8u_C2P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="destY">Destination image channel 0</param>
		/// <param name="destCbCr">Destination image channel 1</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToYCbCr420(NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx(_devPtrRoi, _pitch, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
		/// </summary>
		/// <param name="destY">Destination image channel 0</param>
		/// <param name="destCbCr">Destination image channel 1</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToYCbCr411(NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx(_devPtrRoi, _pitch, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
		/// </summary>
		/// <param name="destY">Destination image channel 0</param>
		/// <param name="destCbCr">Destination image channel 1</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToYCbCr420(NPPImage_8uC1 destY, NPPImage_8uC1 destCbCr, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrAndACrCbAndOther.nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx(_devPtrRoi, _pitch, destY.DevicePointerRoi, destY.Pitch, destCbCr.DevicePointerRoi, destCbCr.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToYCbCr420(NPPImage_8uC4 dest, byte nAval, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrToBGR.nppiYCbCr422ToBGR_8u_C2C4R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAval, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_8u_C2C4R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void YCbCr422ToBGR(NPPImage_8uC4 dest, byte nAval, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.YCbCrToBGR.nppiYCbCr422ToBGR_8u_C2C4R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAval, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiYCbCr422ToBGR_8u_C2C4R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAval">8-bit unsigned alpha constant.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CbYCr422ToBGR_709HDTV(NPPImage_8uC4 dest, byte nAval, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.CbYCrToBGR.nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAval, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}



		/// <summary>
		/// 2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed ARGB color conversion with constant alpha (0xFF).
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcVU">Source image channel VU</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void NV21ToRGB(NPPImage_8uC1 srcY, NPPImage_8uC1 srcVU, NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] devptrs = new CUdeviceptr[] { srcY.DevicePointer, srcVU.DevicePointer};
			NppStatus status = NPPNativeMethods_Ctx.NPPi.NV21ToRGB.nppiNV21ToRGB_8u_P2C4R_Ctx(devptrs, srcY.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV21ToRGB_8u_P2C4R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// 2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed BGRA color conversion with constant alpha (0xFF).
		/// </summary>
		/// <param name="srcY">Source image channel Y</param>
		/// <param name="srcVU">Source image channel VU</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void NV21ToBGR(NPPImage_8uC1 srcY, NPPImage_8uC1 srcVU, NPPImage_8uC2 dest, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] devptrs = new CUdeviceptr[] { srcY.DevicePointer, srcVU.DevicePointer };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.NV21ToBGR.nppiNV21ToBGR_8u_P2C4R_Ctx(devptrs, srcY.Pitch, dest.DevicePointerRoi, dest.Pitch, srcY.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV21ToBGR_8u_P2C4R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Set(byte[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_8u_C2R_Ctx(nValue, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MaxError
		/// <summary>
		/// image maximum error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C2R_Ctx", status));
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
		public void MaxError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_8u_C2R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_8u_C2R_Ctx", status));
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
		public void AverageError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C2R_Ctx", status));
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
		public void AverageError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_8u_C2R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_8u_C2R_Ctx", status));
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
		public void MaximumRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C2R_Ctx", status));
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
		public void MaximumRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_8u_C2R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_8u_C2R_Ctx", status));
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
		public void AverageRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C2R_Ctx", status));
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
		public void AverageRelativeError(NPPImage_8uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_8u_C2R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_8u_C2R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_8u_C2R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_8u_C2R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ColorTwist(NPPImage_8uC2 dest, float[,] twistMatrix, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorProcessing.nppiColorTwist32f_8u_C2R_Ctx(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C2R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.ColorProcessing.nppiColorTwist32f_8u_C2IR_Ctx(_devPtr, _pitch, _sizeRoi, aTwist, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_8u_C2IR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBorder(NPPImage_8uC2 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBorder32f.nppiFilterBorder32f_8u_C2R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_8u_C2R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Filter(NPPImage_8uC2 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Convolution.nppiFilter32f_8u_C2R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_8u_C2R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void NV12ToYUV420(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC1 dest0, NPPImage_8uC1 dest1, NPPImage_8uC1 dest2, NppStreamContext nppStreamCtx)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
            int[] dstPitch = new int[] { dest0.Pitch, dest1.Pitch, dest2.Pitch };
            NppStatus status = NPPNativeMethods_Ctx.NPPi.NV12ToYUV420.nppiNV12ToYUV420_8u_P2P3R_Ctx(src, src0.Pitch, dst, dstPitch, dest0.SizeRoi, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToYUV420_8u_P2P3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void NV12ToRGB(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods_Ctx.NPPi.NV12ToRGB.nppiNV12ToRGB_8u_P2C3R_Ctx(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToRGB_8u_P2C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB 709 HDTV full color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void NV12ToRGB_709HDTV(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods_Ctx.NPPi.NV12ToRGB.nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void NV12ToBGR(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods_Ctx.NPPi.NV12ToBGR.nppiNV12ToBGR_8u_P2C3R_Ctx(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToBGR_8u_P2C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed BGR 709 HDTV full color conversion.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void NV12ToBGR_709HDTV(NPPImage_8uC1 src0, NPPImage_8uC1 src1, NPPImage_8uC3 dest, NppStreamContext nppStreamCtx)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer };
            NppStatus status = NPPNativeMethods_Ctx.NPPi.NV12ToBGR.nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(src, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }
        #endregion
    }
}
