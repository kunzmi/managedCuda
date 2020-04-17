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
	public partial class NPPImage_16fC3 : NPPImageBase
	{
		#region ColorTwist
		/// <summary>
		/// An input color twist matrix with floating-point pixel values is applied
		/// within ROI.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void ColorTwist(NPPImage_16fC3 dest, float[,] twistMatrix, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist32f_16f_C3R_Ctx(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16f_C3R_Ctx", status));
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
			status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist32f_16f_C3IR_Ctx(_devPtr, _pitch, _sizeRoi, aTwist, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16f_C3IR_Ctx", status));
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
			NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwistBatch.nppiColorTwistBatch32f_16f_C3R_Ctx(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch32f_16f_C3R_Ctx", status));
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
			NppStatus status = NPPNativeMethods_Ctx.NPPi.ColorTwistBatch.nppiColorTwistBatch32f_16f_C3IR_Ctx(nMin, nMax, oSizeROI, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwistBatch32f_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, pBatchList);
		}
		#endregion

		#region Abs
		/// <summary>
		/// Image absolute value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Abs(NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Abs.nppiAbs_16f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image absolute value. In place.
		/// </summary>
		public void Abs(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Abs.nppiAbs_16f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16f_C3IR_Ctx", status));
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
		public void Add(NPPImage_16fC3 src2, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_16f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(NPPImage_16fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_16f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(float[] nConstant, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_16f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_16f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16f_C3IR_Ctx", status));
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
		public void Sub(NPPImage_16fC3 src2, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_16f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(NPPImage_16fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_16f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(float[] nConstant, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_16f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_16f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16f_C3IR_Ctx", status));
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
		public void Mul(NPPImage_16fC3 src2, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_16f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_16fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_16f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(float[] nConstant, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_16f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_16f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16f_C3IR_Ctx", status));
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
		public void Div(NPPImage_16fC3 src2, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_16f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_16fC3 src2, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_16f_C3IR_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(float[] nConstant, NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_16f_C3R_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(float[] nConstant, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_16f_C3IR_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Ln
		/// <summary>
		/// Natural logarithm.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Ln(NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Ln.nppiLn_16f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Natural logarithm.
		/// </summary>
		public void Ln(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Ln.nppiLn_16f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqr
		/// <summary>
		/// Image squared.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqr(NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqr.nppiSqr_16f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image squared.
		/// </summary>
		public void Sqr(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqr.nppiSqr_16f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqrt
		/// <summary>
		/// Image square root.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sqrt(NPPImage_16fC3 dest, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqrt.nppiSqrt_16f_C3R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image square root.
		/// </summary>
		public void Sqrt(NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sqrt.nppiSqrt_16f_C3IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16f_C3IR_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Copy

		/// <summary>
		/// image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_16fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_16f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
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
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_16f_C3R_Ctx(nValue, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Convert
		/// <summary>
		/// 16-bit floating point to 32-bit conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Convert(NPPImage_32fC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.BitDepthConversion.nppiConvert_16f32f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16f32f_C3R_Ctx", status));
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
		public void Filter(NPPImage_16fC3 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Convolution.nppiFilter32f_16f_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit float convolution filter with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order</param>
		/// <param name="nKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void FilterBorder(NPPImage_16fC3 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.FilterBorder32f.nppiFilterBorder32f_16f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Resize
		/// <summary>
		/// Resizes images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eInterpolation">Interpolation mode</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Resize(NPPImage_16fC3 dest, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResize_16f_C3R_Ctx(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointer, dest.Pitch, dest.Size, new NppiRect(dest.PointRoi, dest.SizeRoi), eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
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
			NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiResizeBatch_16f_C3R_Advanced_Ctx(nMaxWidth, nMaxHeight, pBatchSrc.DevicePointer, pBatchDst.DevicePointer,
				pBatchROI.DevicePointer, pBatchDst.Size, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeBatch_16f_C3R_Advanced_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region WarpAffine

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
		public void WarpAffine(NPPImage_16fC3 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.AffinTransforms.nppiWarpAffine_16f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffine_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

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
			NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiWarpAffineBatch_16f_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBatch_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		#endregion

		#region WarpPerspective

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
		public void WarpPerspective(NPPImage_16fC3 dest, double[,] coeffs, InterpolationMode eInterpolation, NppStreamContext nppStreamCtx)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods_Ctx.NPPi.PerspectiveTransforms.nppiWarpPerspective_16f_C3R_Ctx(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspective_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
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
			NppStatus status = NPPNativeMethods_Ctx.NPPi.GeometricTransforms.nppiWarpPerspectiveBatch_16f_C3R_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList.DevicePointer, pBatchList.Size, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBatch_16f_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, null);
		}

		#endregion
	}
}
