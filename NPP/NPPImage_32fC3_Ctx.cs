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
	public partial class NPPImage_32fC3 : NPPImageBase
	{
		#region Copy
		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_32fC1 dst, int channel, NppStreamContext nppStreamCtx)
		{
			if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_32f_C3C1R_Ctx(_devPtrRoi + channel * _typeSize, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C3C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 8-bit unsigned packed to planar image copy.
		/// </summary>
		/// <param name="dst0">Destination image channel 0</param>
		/// <param name="dst1">Destination image channel 1</param>
		/// <param name="dst2">Destination image channel 2</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_32fC1 dst0, NPPImage_32fC1 dst1, NPPImage_32fC1 dst2, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] array = new CUdeviceptr[] { dst0.DevicePointerRoi, dst1.DevicePointerRoi, dst2.DevicePointerRoi };
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_32f_C3P3R_Ctx(_devPtrRoi, _pitch, array, dst0.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C3P3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 8-bit unsigned planar to packed image copy.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void Copy(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] array = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			NppStatus status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_32f_P3C3R_Ctx(array, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_P3C3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channelSrc">Channel number. This number is added to the src pointer</param>
		/// <param name="channelDst">Channel number. This number is added to the dst pointer</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_32fC3 dst, int channelSrc, int channelDst, NppStreamContext nppStreamCtx)
		{
			if (channelSrc < 0 | channelSrc >= _channels) throw new ArgumentOutOfRangeException("channelSrc", "channelSrc must be in range [0..2].");
			if (channelDst < 0 | channelDst >= dst.Channels) throw new ArgumentOutOfRangeException("channelDst", "channelDst must be in range [0..2].");
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_32f_C3CR_Ctx(_devPtrRoi + channelSrc * _typeSize, _pitch, dst.DevicePointerRoi + channelDst * _typeSize, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C3CR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="mask">Mask image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_32fC3 dst, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_32f_C3MR_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C3MR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Abs
		/// <summary>
		/// Image absolute value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Abs(NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Abs.nppiAbs_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image absolute value. In place.
		/// </summary>
		public void Abs(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Abs.nppiAbs_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Add
		/// <summary>
		/// Image addition.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(NPPImage_32fC3 src2, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(NPPImage_32fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_32f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(float[] nConstant, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_32f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_32f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sub
		/// <summary>
		/// Image subtraction.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(NPPImage_32fC3 src2, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(NPPImage_32fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_32f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(float[] nConstant, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_32f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_32f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Mul
		/// <summary>
		/// Image multiplication.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_32fC3 src2, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_32fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_32f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(float[] nConstant, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_32f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_32f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Div
		/// <summary>
		/// Image division.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_32fC3 src2, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_32fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_32f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(float[] nConstant, NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_32f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_32f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Exp
		/// <summary>
		/// Exponential.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Exp(NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Exp.nppiExp_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace exponential.
		/// </summary>
		public void Exp(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Exp.nppiExp_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Ln
		/// <summary>
		/// Natural logarithm.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Ln(NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Ln.nppiLn_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Natural logarithm.
		/// </summary>
		public void Ln(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Ln.nppiLn_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqr
		/// <summary>
		/// Image squared.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqr(NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqr.nppiSqr_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image squared.
		/// </summary>
		public void Sqr(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqr.nppiSqr_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqrt
		/// <summary>
		/// Image square root.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqrt(NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqrt.nppiSqrt_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image square root.
		/// </summary>
		public void Sqrt(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqrt.nppiSqrt_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region Geometric Transforms
		/// <summary>
		/// Rotate images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAngle">The angle of rotation in degrees.</param>
		/// <param name="nShiftX">Shift along horizontal axis</param>
		/// <param name="nShiftY">Shift along vertical axis</param>
		/// <param name="eInterpolation">Interpolation mode</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Rotate(NPPImage_32fC3 dest, double nAngle, double nShiftX, double nShiftY, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiRotate_32f_C3R_Ctx(_devPtr, _sizeRoi, _pitch, new NppiRect(_pointRoi, _sizeRoi),
				dest.DevicePointer, dest.Pitch, new NppiRect(dest.PointRoi, dest.SizeRoi), nAngle, nShiftX, nShiftY, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRotate_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Mirror image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mirror(NPPImage_32fC3 dest, NppiAxis flip, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiMirror_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, flip, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		#region Affine Transformations

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
		public void WarpAffine(NPPImage_32fC3 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffine_32f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffine_32f_C3R_Ctx", status));
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
		public void WarpAffineBack(NPPImage_32fC3 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffineBack_32f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBack_32f_C3R_Ctx", status));
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
		public void WarpAffineQuad(double[,] srcQuad, NPPImage_32fC3 dest, double[,] dstQuad, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffineQuad_32f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, dstQuad, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineQuad_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpAffine(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffine_32f_P3R_Ctx(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffine_32f_P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpAffineBack(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffineBack_32f_P3R_Ctx(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBack_32f_P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpAffineQuad(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, double[,] srcQuad, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double[,] dstQuad, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffineQuad_32f_P3R_Ctx(src, src0.Size, src0.Pitch, rectIn, srcQuad, dst, dest0.Pitch, rectOut, dstQuad, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineQuad_32f_P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
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
		public void WarpPerspective(NPPImage_32fC3 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspective_32f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspective_32f_C3R_Ctx", status));
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
		public void WarpPerspectiveBack(NPPImage_32fC3 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspectiveBack_32f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBack_32f_C3R_Ctx", status));
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
		public void WarpPerspectiveQuad(double[,] srcQuad, NPPImage_32fC3 dest, double[,] destQuad, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspectiveQuad_32f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, destQuad, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveQuad_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpPerspective(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspective_32f_P3R_Ctx(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspective_32f_P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpPerspectiveBack(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspectiveBack_32f_P3R_Ctx(src, src0.Size, src0.Pitch, rectIn, dst, dest0.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBack_32f_P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void WarpPerspectiveQuad(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, double[,] srcQuad, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double[,] destQuad, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect rectOut = new NppiRect(dest0.PointRoi, dest0.SizeRoi);

			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspectiveQuad_32f_P3R_Ctx(src, src0.Size, src0.Pitch, rectIn, srcQuad, dst, dest0.Pitch, rectOut, destQuad, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveQuad_32f_P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Convert
		/// <summary>
		/// 32-bit floating point to 8-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_8uC3 dst, NppRoundMode roundMode, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_32f8u_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f8u_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 32-bit floating point to 8-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_8sC3 dst, NppRoundMode roundMode, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_32f8s_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f8s_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 32-bit floating point to 16-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_16uC3 dst, NppRoundMode roundMode, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_32f16u_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f16u_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 32-bit floating point to 16-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_16sC3 dst, NppRoundMode roundMode, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_32f16s_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f16s_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 32-bit floating point to 16-bit floating point conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_16fC3 dst, NppRoundMode roundMode, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_32f16f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sum
		/// <summary>
		/// Scratch-buffer size for nppiSum_32f_C3R.
		/// </summary>
		/// <returns></returns>
		public int SumGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSumGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image sum with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sum(CudaDeviceVariable<double> result, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SumGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSum_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SumGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sum(CudaDeviceVariable<double> result, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SumGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Sum.nppiSum_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.Min.nppiMinGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Min(CudaDeviceVariable<float> min, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Min.nppiMin_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Min(CudaDeviceVariable<float> min, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Min.nppiMin_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MinIdx.nppiMinIndxGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndxGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinIdx.nppiMinIndx_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinIdx.nppiMinIndx_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.Max.nppiMaxGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Max(CudaDeviceVariable<float> max, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Max.nppiMax_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Max(CudaDeviceVariable<float> max, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Max.nppiMax_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MaxIdx.nppiMaxIndxGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndxGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxIndex(CudaDeviceVariable<float> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MaxIdx.nppiMaxIndx_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxIndex(CudaDeviceVariable<float> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaxIdx.nppiMaxIndx_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MinMaxNew.nppiMinMaxGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMax(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinMaxNew.nppiMinMax_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(float)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMax(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinMaxNew.nppiMinMax_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_32f_C3CR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_32f_C3CR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C3CR_Ctx(_devPtrRoi, _pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C3CR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C3CR_Ctx(_devPtrRoi, _pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C3CR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C3CMR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexMaskedGetBufferHostSize()"/></param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinMaxIndex(int coi, CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MinMaxIndexMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMeanGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMeanGetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(CudaDeviceVariable<double> mean, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(CudaDeviceVariable<double> mean, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(int coi, CudaDeviceVariable<double> mean, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C3CMR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mean(int coi, CudaDeviceVariable<double> mean, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanNew.nppiMean_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_32f_C3CR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_32f_C3CR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C3CR_Ctx(_devPtrRoi, _pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C3CR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanStdDevGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C3CR_Ctx(_devPtrRoi, _pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C3CR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C3CMR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanStdDevMaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MeanStdDev(int coi, CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNormInfGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNormInfGetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(CudaDeviceVariable<double> norm, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C3CMR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfMaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormInf(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormInf.nppiNorm_Inf_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNormL1GetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNormL1GetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(CudaDeviceVariable<double> norm, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C3CMR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL1(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL1.nppiNorm_L1_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNormL2GetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNormL2GetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(CudaDeviceVariable<double> norm, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C3CMR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="coi">Channel of interest (0, 1 or 2)</param>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2MaskedGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormL2(int coi, CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormL2.nppiNorm_L2_32f_C3CMR_Ctx(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, coi, norm.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C3CMR_Ctx", status));
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
		public void Threshold(NPPImage_32fC3 dest, float[] nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_32f_C3R_Ctx", status));
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
		public void Threshold(float[] nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_32f_C3IR_Ctx", status));
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
		public void ThresholdGT(NPPImage_32fC3 dest, float[] nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GT_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdGT(float[] nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GT_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_32f_C3IR_Ctx", status));
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
		public void ThresholdLT(NPPImage_32fC3 dest, float[] nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LT_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ThresholdLT(float[] nThreshold, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LT_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_32f_C3IR_Ctx", status));
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
		public void Threshold(NPPImage_32fC3 dest, float[] nThreshold, float[] nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_Val_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_32f_C3R_Ctx", status));
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
		public void Threshold(float[] nThreshold, float[] nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_Val_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_32f_C3IR_Ctx", status));
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
		public void ThresholdGT(NPPImage_32fC3 dest, float[] nThreshold, float[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GTVal_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_32f_C3R_Ctx", status));
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
		public void ThresholdGT(float[] nThreshold, float[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_GTVal_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_32f_C3IR_Ctx", status));
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
		public void ThresholdLT(NPPImage_32fC3 dest, float[] nThreshold, float[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTVal_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_32f_C3R_Ctx", status));
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
		public void ThresholdLT(float[] nThreshold, float[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTVal_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_32f_C3IR_Ctx", status));
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
		public void ThresholdLTGT(NPPImage_32fC3 dest, float[] nThresholdLT, float[] nValueLT, float[] nThresholdGT, float[] nValueGT, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTValGTVal_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_32f_C3R_Ctx", status));
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
		public void ThresholdLTGT(float[] nThresholdLT, float[] nValueLT, float[] nThresholdGT, float[] nValueGT, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Threshold.nppiThreshold_LTValGTVal_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_32f_C3IR_Ctx", status));
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
		public void Compare(NPPImage_32fC3 src2, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Compare.nppiCompare_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompare_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
        /// <summary>
        /// Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="nConstant">list of constants, one per color channel.</param>
        /// <param name="dest">Destination image</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void Compare(float[] nConstant, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Compare.nppiCompareC_32f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareC_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region CompareEqualEps
		/// <summary>
		/// Compare pSrc1's pixels with corresponding pixels in pSrc2.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="epsilon">epsilon tolerance value to compare to pixel absolute differences.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CompareEqualEps(NPPImage_32fC3 src2, NPPImage_8uC1 dest, float epsilon, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Compare.nppiCompareEqualEps_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, epsilon, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareEqualEps_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc's pixels with constant value.
		/// </summary>
		/// <param name="nConstant">list of constants, one per color channel.</param>
		/// <param name="dest">Destination image</param>
		/// <param name="epsilon">epsilon tolerance value to compare to pixel absolute differences.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CompareEqualEps(float[] nConstant, NPPImage_8uC1 dest, float epsilon, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Compare.nppiCompareEqualEpsC_32f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, epsilon, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareEqualEpsC_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Histogram
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
		/// Scratch-buffer size for HistogramRange.
		/// </summary>
		/// <param name="nLevels"></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// <returns></returns>
		public int HistogramRangeGetBufferSize(int[] nLevels, NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramRangeGetBufferSize_32f_C3R_Ctx(_sizeRoi, nLevels, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRangeGetBufferSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 3</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 3</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void HistogramRange(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels, NppStreamContext nppStreamCtx)
		{
			int[] size = new int[] { histogram[0].Size, histogram[1].Size, histogram[2].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSize(size, nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramRange_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 3</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 3</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramRangeGetBufferSize(int[])"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void HistogramRange(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int[] size = new int[] { histogram[0].Size, histogram[1].Size, histogram[2].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSize(size, nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.Histogram.nppiHistogramRange_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//new in Cuda 5.5
		#region DotProduct
		/// <summary>
		/// Device scratch buffer size (in bytes) for nppiDotProd_32f64f_C3R.
		/// </summary>
		/// <returns></returns>
		public int DotProdGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.DotProd.nppiDotProdGetBufferHostSize_32f64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProdGetBufferHostSize_32f64f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Three-channel 32-bit floating point image DotProd.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="DotProdGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void DotProduct(NPPImage_32fC3 src2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = DotProdGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.DotProd.nppiDotProd_32f64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_32f64f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 32-bit floating point image DotProd. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (3 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void DotProduct(NPPImage_32fC3 src2, CudaDeviceVariable<double> pDp, NppStreamContext nppStreamCtx)
		{
			int bufferSize = DotProdGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.DotProd.nppiDotProd_32f64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_32f64f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region LUT

		/// <summary>
		/// look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUT(NPPImage_32fC3 dst, CudaDeviceVariable<float>[] pValues, CudaDeviceVariable<float>[] pLevels, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods_Ctx.NPPi.ColorLUT.nppiLUT_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTCubic(NPPImage_32fC3 dst, CudaDeviceVariable<float>[] pValues, CudaDeviceVariable<float>[] pLevels, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTCubic.nppiLUT_Cubic_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Inplace look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUT(CudaDeviceVariable<float>[] pValues, CudaDeviceVariable<float>[] pLevels, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods_Ctx.NPPi.ColorLUT.nppiLUT_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTCubic(CudaDeviceVariable<float>[] pValues, CudaDeviceVariable<float>[] pLevels, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTCubic.nppiLUT_Cubic_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace linear interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void LUTLinear(CudaDeviceVariable<float>[] pValues, CudaDeviceVariable<float>[] pLevels, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods_Ctx.NPPi.ColorLUTLinear.nppiLUT_Linear_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Lut(NPPImage_32fC3 dest, CudaDeviceVariable<float> values0, CudaDeviceVariable<float> levels0, CudaDeviceVariable<float> values1, CudaDeviceVariable<float> levels1, CudaDeviceVariable<float> values2, CudaDeviceVariable<float> levels2, NppStreamContext nppStreamCtx)
		{

			CUdeviceptr[] values = new CUdeviceptr[3];
			CUdeviceptr[] levels = new CUdeviceptr[3];
			int[] levelLengths = new int[3];

			values[0] = values0.DevicePointer;
			values[1] = values1.DevicePointer;
			values[2] = values2.DevicePointer;
			levels[0] = levels0.DevicePointer;
			levels[1] = levels1.DevicePointer;
			levels[2] = levels2.DevicePointer;

			levelLengths[0] = levels0.Size;
			levelLengths[1] = levels1.Size;
			levelLengths[2] = levels2.Size;

			status = NPPNativeMethods_Ctx.NPPi.ColorLUTLinear.nppiLUT_Linear_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, values, levels, levelLengths, nppStreamCtx);

			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Transpose
		/// <summary>
		/// image transpose
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Transpose(NPPImage_32fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Transpose.nppiTranspose_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Color...New

		/// <summary>
		/// Swap color channels
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. <para/>The n-th entry of the array
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// contains the number of the channel that is stored in the n-th channel of the output image. <para/>E.g.
		/// Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
		public void SwapChannels(NPPImage_32fC3 dest, int[] aDstOrder, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SwapChannel.nppiSwapChannels_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
		/// channel. <para/>An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
		/// particular destination channel value unmodified.</param>
		public void SwapChannels(NPPImage_32fC4 dest, int[] aDstOrder, float nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SwapChannel.nppiSwapChannels_32f_C3C4R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_32f_C3C4R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Swap color channels inplace
		/// </summary>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. <para/>The n-th entry of the array
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// contains the number of the channel that is stored in the n-th channel of the output image. <para/>E.g.
		/// Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
		public void SwapChannels(int[] aDstOrder, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SwapChannel.nppiSwapChannels_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, aDstOrder, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// RGB to Gray conversion
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void RGBToGray(NPPImage_32fC1 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RGBToGray.nppiRGBToGray_32f_C3C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToGray_32f_C3C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Color to Gray conversion
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ColorToGray(NPPImage_32fC1 dest, float[] aCoeffs, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorToGray.nppiColorToGray_32f_C3C1R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aCoeffs, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorToGray_32f_C3C1R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist_32f_C3IR_Ctx(_devPtr, _pitch, _sizeRoi, aTwist, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// An input color twist matrix with floating-point pixel values is applied
		/// within ROI.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ColorTwist(NPPImage_32fC3 dest, float[,] twistMatrix, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist_32f_C3R_Ctx(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void ColorTwist(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, float[,] twistMatrix, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist_32f_P3R_Ctx(src, src0.Pitch, dst, dest0.Pitch, src0.SizeRoi, twistMatrix, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist_32f_P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void ColorTwist(NPPImage_32fC1 srcDest0, NPPImage_32fC1 srcDest1, NPPImage_32fC1 srcDest2, float[,] twistMatrix, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { srcDest0.DevicePointerRoi, srcDest1.DevicePointerRoi, srcDest2.DevicePointerRoi };

			NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist_32f_IP3R_Ctx(src, srcDest0.Pitch, srcDest0.SizeRoi, twistMatrix, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist_32f_IP3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set (Array size = 3)</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Set(float[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_32f_C3R_Ctx(nValue, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Set(float[] nValue, NPPImage_8uC1 mask, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_32f_C3MR_Ctx(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32f_C3MR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Set(float nValue, int channel, NppStreamContext nppStreamCtx)
		{
			if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_32f_C3CR_Ctx(nValue, _devPtrRoi + channel * _typeSize, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32f_C3CR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Copy

		/// <summary>
		/// image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C3R_Ctx", status));
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
		public void Copy(NPPImage_32fC3 dst, int nTopBorderHeight, int nLeftBorderWidth, float[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CopyConstBorder.nppiCopyConstBorder_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		public void CopyReplicateBorder(NPPImage_32fC3 dst, int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_32f_C3R_Ctx", status));
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
		public void CopyWrapBorder(NPPImage_32fC3 dst, int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CopyWrapBorder.nppiCopyWrapBorder_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// linearly interpolated source image subpixel coordinate color copy.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nDx">Fractional part of source image X coordinate.</param>
		/// <param name="nDy">Fractional part of source image Y coordinate.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CopySubpix(NPPImage_32fC3 dst, float nDx, float nDy, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.CopySubpix.nppiCopySubpix_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dilate(NPPImage_32fC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiDilate_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate_32f_C3R_Ctx", status));
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
		public void Erode(NPPImage_32fC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiErode_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 3x3 dilation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dilate3x3(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiDilate3x3_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Erode3x3(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MorphologyFilter2D.nppiErode3x3_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3_32f_C3R_Ctx", status));
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
		public void DilateBorder(NPPImage_32fC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DilationWithBorderControl.nppiDilateBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilateBorder_32f_C3R_Ctx", status));
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
		public void ErodeBorder(NPPImage_32fC3 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ErosionWithBorderControl.nppiErodeBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErodeBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 dilation with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Dilate3x3Border(NPPImage_32fC3 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Dilate3x3Border.nppiDilate3x3Border_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3Border_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Erode3x3Border(NPPImage_32fC3 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Erode3x3Border.nppiErode3x3Border_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3Border_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Filter(NPPImage_32fC3 dest, CudaDeviceVariable<float> Kernel, NppiSize aKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Convolution.nppiFilter_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, aKernelSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterColumn(NPPImage_32fC3 dest, CudaDeviceVariable<float> Kernel, int nKernelSize, int nAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterColumn_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRow(NPPImage_32fC3 dest, CudaDeviceVariable<float> Kernel, int nKernelSize, int nAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterRow_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 1D row convolution with border control. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRowBorder(NPPImage_32fC3 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterRowBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBox(NPPImage_32fC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFixedFilters2D.nppiFilterBox_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBox_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMin(NPPImage_32fC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMin_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMin_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterMax(NPPImage_32fC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMax_32f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMax_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterPrewittHoriz(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittHoriz_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHoriz_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterPrewittVert(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittVert_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVert_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SobelHoriz(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelHoriz_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHoriz_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVert(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSobelVert_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVert_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsDown(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsDown_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDown_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter..
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsUp(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsUp_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUp_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLaplace(NPPImage_32fC3 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLaplace_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplace_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Gauss filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGauss(NPPImage_32fC3 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterGauss_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGauss_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterHighPass(NPPImage_32fC3 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterHighPass_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPass_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLowPass(NPPImage_32fC3 dst, MaskSize eMaskSize, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLowPass_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPass_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSharpen(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSharpen_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpen_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2GetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2GetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2GetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_32f_C3R_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_32f_C3R_Ctx", status));
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
		public void CrossCorrFull_NormLevel(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FullNormLevelGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_NormLevel(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FullNormLevelGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_32f_C3R_Ctx", status));
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
		public void CrossCorrSame_NormLevel(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SameNormLevelGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_NormLevel(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = SameNormLevelGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_32f_C3R_Ctx", status));
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
		public void CrossCorrValid_NormLevel(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_NormLevel(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_32f_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}










		/// <summary>
		/// image SqrDistanceFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceFull_Norm(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceFull_Norm_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceFull_Norm_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceSame_Norm(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceSame_Norm_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceSame_Norm_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void SqrDistanceValid_Norm(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSqrDistanceValid_Norm_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceValid_Norm_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}





		/// <summary>
		/// image CrossCorrFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrFull_Norm(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_Norm_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrSame_Norm(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_Norm_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CrossCorrValid_Norm(NPPImage_32fC3 tpl, NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_Norm_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_32f_C3CMR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_Inf_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_32f_C3CMR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L1_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_32f_C3CMR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormDiff_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormDiff, int nCOI, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormDiff.nppiNormDiff_L2_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormDiff.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_32f_C3CMR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_Inf(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_Inf_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_32f_C3CMR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L1(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L1_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C3CMR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_32f_C3CMR_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_32f_C3CMR_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C3CMR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="nCOI">channel of interest.</param>
		/// <param name="pMask">Mask image.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void NormRel_L2(NPPImage_32fC3 tpl, CudaDeviceVariable<double> pNormRel, int nCOI, NPPImage_8uC1 pMask, NppStreamContext nppStreamCtx)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.NormRel.nppiNormRel_L2_32f_C3CMR_Ctx(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, nCOI, pNormRel.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C3CMR_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}






		#endregion

		#region MinMaxEveryNew
		/// <summary>
		/// image MinEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MinEvery(NPPImage_32fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MinMaxEvery.nppiMinEvery_32f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinEvery_32f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image MaxEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxEvery(NPPImage_32fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MinMaxEvery.nppiMaxEvery_32f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxEvery_32f_C3IR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiMirror_32f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, flip, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_32f_C3IR_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.CountInRange.nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CountInRange(CudaDeviceVariable<int> pCounts, float[] nLowerBound, float[] nUpperBound, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = CountInRangeGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.CountInRange.nppiCountInRange_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image CountInRange.
		/// </summary>
		/// <param name="pCounts">Pointer to the number of pixels that fall into the specified range. (3 * sizeof(int))</param>
		/// <param name="nLowerBound">Fixed size array of the lower bound of the specified range, one per channel.</param>
		/// <param name="nUpperBound">Fixed size array of the upper bound of the specified range, one per channel.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void CountInRange(CudaDeviceVariable<int> pCounts, float[] nLowerBound, float[] nUpperBound, NppStreamContext nppStreamCtx)
		{
			int bufferSize = CountInRangeGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.CountInRange.nppiCountInRange_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_32f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.QualityIndex.nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndexGetBufferHostSize_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (3 * sizeof(float))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="QualityIndexGetBufferHostSize()"/></param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void QualityIndex(NPPImage_32fC3 src2, CudaDeviceVariable<float> dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = QualityIndexGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.QualityIndex.nppiQualityIndex_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (3 * sizeof(float))</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void QualityIndex(NPPImage_32fC3 src2, CudaDeviceVariable<float> dst, NppStreamContext nppStreamCtx)
		{
			int bufferSize = QualityIndexGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods_Ctx.NPPi.QualityIndex.nppiQualityIndex_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_32f_C3R_Ctx", status));
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
		public void ResizeSqrPixel(NPPImage_32fC3 dst, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect dstRect = new NppiRect(dst.PointRoi, dst.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_32f_C3R_Ctx(_devPtr, _sizeRoi, _pitch, srcRect, dst.DevicePointer, dst.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_32f_C3R_Ctx", status));
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
		public void Remap(NPPImage_32fC3 dst, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.Remap.nppiRemap_32f_C3R_Ctx(_devPtr, _sizeRoi, _pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
		/// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Scale(NPPImage_32fC3 dst, float nMin, float nMax, NppStreamContext nppStreamCtx)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.Scale.nppiScale_32f8u_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nMin, nMax, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_32f8u_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

        ///// <summary>
        ///// Resizes images.
        ///// </summary>
        ///// <param name="dest">Destination image</param>
        ///// <param name="xFactor">X scaling factor</param>
        ///// <param name="yFactor">Y scaling factor</param>
        ///// <param name="eInterpolation">Interpolation mode</param>
        //public void Resize(NPPImage_32fC3 dest, double xFactor, double yFactor, InterpolationMode eInterpolation)
        //{
        //	status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResize_32f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, xFactor, yFactor, eInterpolation, nppStreamCtx);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_32f_C3R_Ctx", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

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
        //public static void Resize(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double xFactor, double yFactor, InterpolationMode eInterpolation)
        //{
        //	CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
        //	CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
        //	NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResize_32f_P3R_Ctx(src, src0.Size, src0.Pitch, new NppiRect(src0.PointRoi, src0.SizeRoi), dst, dest0.Pitch, dest0.SizeRoi, xFactor, yFactor, eInterpolation, nppStreamCtx);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_32f_P3R_Ctx", status));
        //	NPPException.CheckNppStatus(status, null);
        //}


        /// <summary>
        /// Resizes images.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="eInterpolation">Interpolation mode</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void Resize(NPPImage_32fC3 dest, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResize_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointer, dest.Pitch, dest.Size, new NppiRect(dest.PointRoi, dest.SizeRoi), eInterpolation, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_32f_C3R_Ctx_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }


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
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void Resize(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResize_32f_P3R_Ctx(src, src0.Pitch, src0.Size, new NppiRect(src0.PointRoi, src0.SizeRoi), dst, dest0.Pitch, dest0.Size, new NppiRect(dest0.PointRoi, dest0.SizeRoi), eInterpolation, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_32f_P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void ResizeSqrPixel(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer };
			NppiRect srcRect = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect dstRect = new NppiRect(dest0.PointRoi, dest0.SizeRoi);
			NppStatus status = NPPNativeMethods_Ctx.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_32f_P3R_Ctx(src, src0.SizeRoi, src0.Pitch, srcRect, dst, dest0.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_32f_P3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public static void Remap(NPPImage_32fC1 src0, NPPImage_32fC1 src1, NPPImage_32fC1 src2, NPPImage_32fC1 dest0, NPPImage_32fC1 dest1, NPPImage_32fC1 dest2, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };
			NppiRect srcRect = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppStatus status = NPPNativeMethods_Ctx.NPPi.Remap.nppiRemap_32f_P3R_Ctx(src, src0.SizeRoi, src0.Pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst, dest0.Pitch, dest0.SizeRoi, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_32f_P3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		#endregion

		//New in Cuda 6.0


		#region FilterBorder
		/// <summary>
		/// Three channel 32-bit float convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_32fC3 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBorder.nppiFilterBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder_32f_C3R_Ctx", status));
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
		public void FilterSobelHorizBorder(NPPImage_32fC3 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSobelVertBorder(NPPImage_32fC3 dest, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterSobelVertBorder.nppiFilterSobelVertBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertBorder_32f_C3R_Ctx", status));
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
		public void FilterMedian(NPPImage_32fC3 dst, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize, nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.ImageMedianFilter.nppiFilterMedian_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_32f_C3R_Ctx", status));
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
		public void FilterMedian(NPPImage_32fC3 dst, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize, nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.ImageMedianFilter.nppiFilterMedian_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for FilterMedian.
		/// </summary>
		/// <returns></returns>
		public int FilterMedianGetBufferHostSize(NppiSize oMaskSize, NppStreamContext nppStreamCtx)
		{
			uint bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.ImageMedianFilter.nppiFilterMedianGetBufferSize_32f_C3R_Ctx(_sizeRoi, oMaskSize, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianGetBufferSize_32f_C3R_Ctx", status));
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
		public void MaxError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_32f_C3R_Ctx", status));
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
		public void MaxError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_32f_C3R_Ctx", status));
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
		public void AverageError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32f_C3R_Ctx", status));
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
		public void AverageError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_32f_C3R_Ctx", status));
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
		public void MaximumRelativeError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32f_C3R_Ctx", status));
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
		public void MaximumRelativeError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_32f_C3R_Ctx", status));
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
		public void AverageRelativeError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32f_C3R_Ctx", status));
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
		public void AverageRelativeError(NPPImage_32fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_32f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_32f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_32f_C3R_Ctx", status));
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
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussBorder(NPPImage_32fC3 dest, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussBorder.nppiFilterGaussBorder_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion


		//New in Cuda 7.0
		#region FilterColumnBorder

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
		public void FilterColumnBorder(NPPImage_32fC3 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFilter1D.nppiFilterColumnBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder_32f_C3R_Ctx", status));
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
		public void FilterBoxBorder(NPPImage_32fC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.LinearFixedFilters2D.nppiFilterBoxBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorder_32f_C3R_Ctx", status));
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
		public void FilterMinBorder(NPPImage_32fC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMinBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMinBorder_32f_C3R_Ctx", status));
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
		public void FilterMaxBorder(NPPImage_32fC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.RankFilters.nppiFilterMaxBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMaxBorder_32f_C3R_Ctx", status));
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
		public void FilterPrewittHorizBorder(NPPImage_32fC3 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittHorizBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHorizBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterPrewittVertBorder(NPPImage_32fC3 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterPrewittVertBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVertBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsDownBorder(NPPImage_32fC3 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsDownBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDownBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterRobertsUpBorder(NPPImage_32fC3 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterRobertsUpBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUpBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLaplaceBorder(NPPImage_32fC3 dst, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLaplaceBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplaceBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterHighPassBorder(NPPImage_32fC3 dst, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterHighPassBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPassBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterLowPassBorder(NPPImage_32fC3 dst, MaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterLowPassBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPassBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterSharpenBorder(NPPImage_32fC3 dst, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterSharpenBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpenBorder_32f_C3R_Ctx", status));
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
		public void FilterUnsharpBorder(NPPImage_32fC3 dst, float nRadius, float nSigma, float nWeight, float nThreshold, NppiBorderType eBorderType, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			if (buffer.Size < FilterUnsharpGetBufferSize(nRadius, nSigma))
				throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterUnsharpBorder_32f_C3R_Ctx(_devPtr, _pitch, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nRadius, nSigma, nWeight, nThreshold, eBorderType, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpBorder_32f_C3R_Ctx", status));
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
		public void FilterGauss(NPPImage_32fC3 dst, CudaDeviceVariable<float> Kernel, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FixedFilters.nppiFilterGaussAdvanced_32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvanced_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussBorder(NPPImage_32fC3 dst, CudaDeviceVariable<float> Kernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussBorder.nppiFilterGaussAdvancedBorder_32f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvancedBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region GradientColorToGray


		/// <summary>
		/// 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray Gradient conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eNorm">Gradient distance method to use.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void GradientColorToGray(NPPImage_32fC1 dest, NppiNorm eNorm, NppStreamContext nppStreamCtx)
		{
			NppStatus status = NPPNativeMethods_Ctx.NPPi.GradientColorToGray.nppiGradientColorToGray_32f_C3C1R_Ctx(DevicePointerRoi, Pitch, dest.DevicePointerRoi, dest.Pitch, SizeRoi, eNorm, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientColorToGray_32f_C3C1R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion
		#region FilterGaussAdvancedBorder

		/// <summary>
		/// Three channel 32-bit floating-point Gauss filter with downsampling and border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <param name="nFilterTaps">The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. </param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussPyramidLayerDownBorder(NPPImage_32fC3 dest, float nRate, int nFilterTaps, CudaDeviceVariable<float> pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussPyramid.nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nRate, nFilterTaps, pKernel.DevicePointer, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three channel 32-bit floating-point Gauss filter with downsampling and border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRate">The downsampling rate to be used.  For integer equivalent rates unnecessary source pixels are just skipped. For non-integer rates the source image is bilinear interpolated. nRate must be > 1.0F and &lt;=  10.0F. </param>
		/// <param name="nFilterTaps">The number of filter taps where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F. </param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterGaussPyramidLayerUpBorder(NPPImage_32fC3 dest, float nRate, int nFilterTaps, CudaDeviceVariable<float> pKernel, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterGaussPyramid.nppiFilterGaussPyramidLayerUpBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, nRate, nFilterTaps, pKernel.DevicePointer, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussPyramidLayerUpBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region FilterBilateralGaussBorder


		/// <summary>
		/// Three channel 32-bit floating-point bilateral Gauss filter with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nRadius">The radius of the round filter kernel to be used.  A radius of 1 indicates a filter kernel size of 3 by 3, 2 indicates 5 by 5, etc. Radius values from 1 to 32 are supported.</param>
		/// <param name="nStepBetweenSrcPixels">The step size between adjacent source image pixels processed by the filter kernel, most commonly 1. </param>
		/// <param name="nValSquareSigma">The square of the sigma for the relative intensity distance between a source image pixel in the filter kernel and the source image pixel at the center of the filter kernel.</param>
		/// <param name="nPosSquareSigma">The square of the sigma for the relative geometric distance between a source image pixel in the filter kernel and the source image pixel at the center of the filter kernel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBilateralGaussBorder(NPPImage_32fC3 dest, int nRadius, int nStepBetweenSrcPixels, float nValSquareSigma, float nPosSquareSigma, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBilateralGaussBorder.nppiFilterBilateralGaussBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nRadius, nStepBetweenSrcPixels, nValSquareSigma, nPosSquareSigma, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBilateralGaussBorder_32f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region GradientVectorPrewittBorder

		/// <summary>
		/// 3 channel 32-bit floatring-point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
		/// and/or 32-bit floating point angle gradient vectors with user selectable fixed mask size and distance method with border control.
		/// </summary>
		/// <param name="destX">X vector destination_image_pointer</param>
		/// <param name="destY">Y vector destination_image_pointer.</param>
		/// <param name="destMag">magnitude destination_image_pointer.</param>
		/// <param name="destAngle">angle destination_image_pointer.</param>
		/// <param name="eNorm">gradient distance method to use.</param>
		/// <param name="eMaskSize">fixed filter mask size to use.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void GradientVectorPrewittBorder(NPPImage_32fC1 destX, NPPImage_32fC1 destY, NPPImage_32fC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GradientVectorPrewittBorder.nppiGradientVectorPrewittBorder_32f_C3C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorPrewittBorder_32f_C3C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GradientVectorScharrBorder

		/// <summary>
		/// 3 channel 32-bit floatring-point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
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
		public void GradientVectorScharrBorder(NPPImage_32fC1 destX, NPPImage_32fC1 destY, NPPImage_32fC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GradientVectorScharrBorder.nppiGradientVectorScharrBorder_32f_C3C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorScharrBorder_32f_C3C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GradientVectorSobelBorder

		/// <summary>
		/// 3 channel 32-bit floatring-point packed RGB to optional 1 channel 32-bit floating point X (vertical), Y (horizontal), magnitude, 
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
		public void GradientVectorSobelBorder(NPPImage_32fC1 destX, NPPImage_32fC1 destY, NPPImage_32fC1 destMag, NPPImage_32fC1 destAngle, MaskSize eMaskSize, NppiNorm eNorm, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GradientVectorSobelBorder.nppiGradientVectorSobelBorder_32f_C3C1R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, destX.DevicePointerRoi, destX.Pitch, destY.DevicePointerRoi, destY.Pitch, destMag.DevicePointerRoi, destMag.Pitch, destAngle.DevicePointerRoi, destAngle.Pitch, _sizeRoi, eMaskSize, eNorm, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGradientVectorSobelBorder_32f_C3C1R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
        #endregion


        //New in Cuda 9.0
        #region New in Cuda 9.0
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
            NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwistBatch.nppiColorTwistBatch_32f_C3R_Ctx(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, pBatchList);
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
        public static void ColorTwistBatchI(float nMin, float nMax, NppiSize oSizeROI, CudaDeviceVariable<NppiColorTwistBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
        {
            NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwistBatch.nppiColorTwistBatch_32f_C3IR_Ctx(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch_32f_C3IR_Ctx", status));
            NPPException.CheckNppStatus(status, pBatchList);
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
        public void FilterWienerBorder(NPPImage_32fC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, float[] aNoise, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.FilterWienerBorder.nppiFilterWienerBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, aNoise, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterWienerBorder_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 3 channel 32-bit floating point color per source image descriptor window location with source image border control 
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
            status = NPPNativeMethods_Ctx.NPPi.HistogramOfOrientedGradientsBorder.nppiHistogramOfGradientsBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, hpLocations, hpLocations.Length, pDstWindowDescriptorBuffer.DevicePointer, _sizeRoi, oHOGConfig, pScratchBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramOfGradientsBorder_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }



        /// <summary>
        /// floating point image resize batch.
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
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResizeBatch_32f_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oSmallestDstSize, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeBatch_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// floating point image resize batch.
        /// </summary>
        /// <param name="oSizeROI">Size in pixel.</param>
        /// <param name="flip">Specifies the axis about which the images are to be mirrored..</param>
        /// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiMirrorBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void MirrorBatch(NppiSize oSizeROI, NppiAxis flip, CudaDeviceVariable<NppiMirrorBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
        {
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiMirrorBatch_32f_C3R_Ctx(oSizeROI, flip, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirrorBatch_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// floating point in place image resize batch.
        /// </summary>
        /// <param name="oSizeROI">Size in pixel.</param>
        /// <param name="flip">Specifies the axis about which the images are to be mirrored..</param>
        /// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiMirrorBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void MirrorBatchI(NppiSize oSizeROI, NppiAxis flip, CudaDeviceVariable<NppiMirrorBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
        {
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiMirrorBatch_32f_C3IR_Ctx(oSizeROI, flip, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirrorBatch_32f_C3IR_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// floating point image warp affine batch.
        /// </summary>
        /// <param name="oSmallestSrcSize">Size in pixels of the entire smallest source image width and height, may be from different images.</param>
        /// <param name="oSrcRectROI">Region of interest in the source images (may overlap source image size width and height).</param>
        /// <param name="oDstRectROI">Region of interest in the destination images (may overlap destination image size width and height).</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, or NPPI_INTER_SUPER. </param>
        /// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiWarpAffineBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void WarpAffineBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiWarpAffineBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
        {
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiWarpAffineBatch_32f_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBatch_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }




        /// <summary>
        /// 1 channel 32-bit floating point morphological close with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphCloseBorder(NPPImage_32fC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphCloseBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphCloseBorder_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 1 channel 32-bit floating point morphological open with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphOpenBorder(NPPImage_32fC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphOpenBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphOpenBorder_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 32-bit floating point morphological top hat with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphTopHatBorder(NPPImage_32fC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphTopHatBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphTopHatBorder_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 32-bit floating point morphological black hat with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphBlackHatBorder(NPPImage_32fC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphBlackHatBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphBlackHatBorder_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// 1 channel 32-bit floating point morphological gradient with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="pMask">Pointer to the start address of the mask array</param>
        /// <param name="oMaskSize">Width and Height mask array.</param>
        /// <param name="oAnchor">X and Y offsets of the mask origin frame of reference w.r.t the source pixel.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public void MorphGradientBorder(NPPImage_32fC3 dest, CudaDeviceVariable<byte> pMask, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ComplexImageMorphology.nppiMorphGradientBorder_32f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, pMask.DevicePointer, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMorphGradientBorder_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region new in Cuda 9.2

        /// <summary>
        /// floating point image warp perspective batch.
        /// </summary>
        /// <param name="oSmallestSrcSize">Size in pixels of the entire smallest source image width and height, may be from different images.</param>
        /// <param name="oSrcRectROI">Region of interest in the source images (may overlap source image size width and height).</param>
        /// <param name="oDstRectROI">Region of interest in the destination images (may overlap destination image size width and height).</param>
        /// <param name="eInterpolation">The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, or NPPI_INTER_CUBIC. </param>
        /// <param name="pBatchList">Device memory pointer to nBatchSize list of NppiWarpPerspectiveBatchCXR structures.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
        public static void WarpPerspectiveBatch(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, InterpolationMode eInterpolation, CudaDeviceVariable<NppiWarpPerspectiveBatchCXR> pBatchList, NppStreamContext nppStreamCtx)
        {
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiWarpPerspectiveBatch_32f_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBatch_32f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }
        #endregion

        #region new in Cuda 10.0


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
            NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResizeBatch_32f_C3R_Advanced_Ctx(nMaxWidth, nMaxHeight, pBatchSrc.DevicePointer, pBatchDst.DevicePointer,
                pBatchROI.DevicePointer, pBatchDst.Size, eInterpolation, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeBatch_32f_C3R_Advanced_Ctx", status));
            NPPException.CheckNppStatus(status, null);
        }
        #endregion
    }
}
