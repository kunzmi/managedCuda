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
	public class NPPImage_32fC1 : NPPImageBase
	{
		#region Constructors
		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="nWidthPixels">Image width in pixels</param>
		/// <param name="nHeightPixels">Image height in pixels</param>
		public NPPImage_32fC1(int nWidthPixels, int nHeightPixels)
		{
			_sizeOriginal.width = nWidthPixels;
			_sizeOriginal.height = nHeightPixels;
			_sizeRoi.width = nWidthPixels;
			_sizeRoi.height = nHeightPixels;
			_channels = 1;
			_isOwner = true;
			_typeSize = sizeof(float);

			_devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_32f_C1(nWidthPixels, nHeightPixels, ref _pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_32f_C1", res, _pitch, _channels));
			
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
		public NPPImage_32fC1(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
			_typeSize = sizeof(float);
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="width">Image width in pixels</param>
		/// <param name="height">Image height in pixels</param>
		/// <param name="pitch">Pitch / Line step</param>
		public NPPImage_32fC1(CUdeviceptr devPtr, int width, int height, int pitch)
			: this(devPtr, width, height, pitch, false)
		{

		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
		/// </summary>
		/// <param name="image">NPP image</param>
		public NPPImage_32fC1(NPPImageBase image)
			: this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
		{

		}

		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="size">Image size</param>
		public NPPImage_32fC1(NppiSize size)
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
		public NPPImage_32fC1(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
			: this(devPtr, size.width, size.height, pitch, isOwner)
		{ 
			
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="size">Image size</param>
		/// <param name="pitch">Pitch / Line step</param>
		public NPPImage_32fC1(CUdeviceptr devPtr, NppiSize size, int pitch)
			: this(devPtr, size.width, size.height, pitch)
		{

		}

		/// <summary>
		/// For dispose
		/// </summary>
		~NPPImage_32fC1()
		{
			Dispose (false);
		}
		#endregion

		#region Converter operators

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		public CudaPitchedDeviceVariable<float> ToCudaPitchedDeviceVariable()
		{
			return new CudaPitchedDeviceVariable<float>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
		}

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		/// <param name="img">NPPImage</param>
		/// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
		public static implicit operator CudaPitchedDeviceVariable<float>(NPPImage_32fC1 img)
		{
			return img.ToCudaPitchedDeviceVariable();
		}

		/// <summary>
		/// Converts a CudaPitchedDeviceVariable to a NPPImage 
		/// </summary>
		/// <param name="img">CudaPitchedDeviceVariable</param>
		/// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
		public static implicit operator NPPImage_32fC1(CudaPitchedDeviceVariable<float> img)
		{
			return img.ToNPPImage();
		}
		#endregion

		#region Copy
		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Copy(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="mask">Mask image</param>
		public void Copy(NPPImage_32fC1 dst, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32f_C1MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		public void Copy(NPPImage_32fC3 dst, int channel)
		{
			if (channel < 0 | channel >= dst.Channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32f_C1C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi + channel * _typeSize, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C1C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		public void Copy(NPPImage_32fC4 dst, int channel)
		{
			if (channel < 0 | channel >= dst.Channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..3].");
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32f_C1C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi + channel * _typeSize, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C1C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set</param>
		public void Set(float nValue)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_32f_C1R(nValue, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32f_C1R", status));
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
		public void Set(float nValue, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_32f_C1MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Convert
		/// <summary>
		/// 32-bit floating point to 16-bit conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		public void Convert(NPPImage_16uC1 dst, NppRoundMode roundMode)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f16u_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f16u_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 16-bit conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		public void Convert(NPPImage_16sC1 dst, NppRoundMode roundMode)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f16s_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f16s_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 8-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		public void Convert(NPPImage_8uC1 dst, NppRoundMode roundMode)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f8u_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f8u_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 8-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		public void Convert(NPPImage_8sC1 dst, NppRoundMode roundMode)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f8s_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f8s_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 8-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="scaleFactor">Integer Result Scaling.</param>
		public void Convert(NPPImage_8uC1 dst, NppRoundMode roundMode, int scaleFactor)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f8u_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f8u_C1RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 8-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="scaleFactor">Integer Result Scaling.</param>
		public void Convert(NPPImage_8sC1 dst, NppRoundMode roundMode, int scaleFactor)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f8s_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f8s_C1RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 16-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="scaleFactor">Integer Result Scaling.</param>
		public void Convert(NPPImage_16uC1 dst, NppRoundMode roundMode, int scaleFactor)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f16u_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f16u_C1RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 16-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="scaleFactor">Integer Result Scaling.</param>
		public void Convert(NPPImage_16sC1 dst, NppRoundMode roundMode, int scaleFactor)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f16s_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f16s_C1RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 32-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="scaleFactor">Integer Result Scaling.</param>
		public void Convert(NPPImage_32uC1 dst, NppRoundMode roundMode, int scaleFactor)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f32u_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f32u_C1RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 32-bit floating point to 32-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="roundMode">Flag specifying how fractional float values are rounded to integer values.</param>
		/// <param name="scaleFactor">Integer Result Scaling.</param>
		public void Convert(NPPImage_32sC1 dst, NppRoundMode roundMode, int scaleFactor)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32f32s_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32f32s_C1RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Add
		/// <summary>
		/// Image addition.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void Add(NPPImage_32fC1 src2, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void Add(NPPImage_32fC1 src2)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_32f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image.
		/// </summary>
		/// <param name="nConstant">Value to add</param>
		/// <param name="dest">Destination image</param>
		public void Add(float nConstant, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to add</param>
		public void Add(float nConstant)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Geometric Transforms

		/// <summary>
		/// Mirror image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		public void Mirror(NPPImage_32fC1 dest, NppiAxis flip)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

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
		public void Rotate(NPPImage_32fC1 dest, double nAngle, double nShiftX, double nShiftY, InterpolationMode eInterpolation)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiRotate_32f_C1R(_devPtr, _sizeRoi, _pitch, new NppiRect(_pointRoi, _sizeRoi),
				dest.DevicePointer, dest.Pitch, new NppiRect(dest.PointRoi, dest.SizeRoi), nAngle, nShiftX, nShiftY, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRotate_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sub
		/// <summary>
		/// Image subtraction.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void Sub(NPPImage_32fC1 src2, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void Sub(NPPImage_32fC1 src2)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_32f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		public void Sub(float nConstant, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		public void Sub(float nConstant)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Mul
		/// <summary>
		/// Image multiplication.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void Mul(NPPImage_32fC1 src2, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void Mul(NPPImage_32fC1 src2)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_32f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		public void Mul(float nConstant, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		public void Mul(float nConstant)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Div
		/// <summary>
		/// Image division.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void Div(NPPImage_32fC1 src2, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		public void Div(NPPImage_32fC1 src2)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_32f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		public void Div(float nConstant, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		public void Div(float nConstant)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Exp
		/// <summary>
		/// Exponential.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Exp(NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Exp.nppiExp_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace exponential.
		/// </summary>
		public void Exp()
		{
			status = NPPNativeMethods.NPPi.Exp.nppiExp_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Ln
		/// <summary>
		/// Natural logarithm.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Ln(NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Ln.nppiLn_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Natural logarithm.
		/// </summary>
		public void Ln()
		{
			status = NPPNativeMethods.NPPi.Ln.nppiLn_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqr
		/// <summary>
		/// Image squared.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Sqr(NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image squared.
		/// </summary>
		public void Sqr()
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqrt
		/// <summary>
		/// Image square root.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Sqrt(NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image square root.
		/// </summary>
		public void Sqrt()
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_32f_C1IR", status));
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
		public void AlphaComp(float alpha1, NPPImage_32fC1 src2, float alpha2, NPPImage_32fC1 dest, NppiAlphaOp nppAlphaOp)
		{
			status = NPPNativeMethods.NPPi.AlphaCompConst.nppiAlphaCompC_32f_C1R(_devPtrRoi, _pitch, alpha1, src2.DevicePointerRoi, src2.Pitch, alpha2, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaCompC_32f_C1R", status));
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
		public void WarpAffine(NPPImage_32fC1 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffine_32f_C1R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffine_32f_C1R", status));
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
		public void WarpAffineBack(NPPImage_32fC1 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffineBack_32f_C1R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineBack_32f_C1R", status));
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
		public void WarpAffineQuad(double[,] srcQuad, NPPImage_32fC1 dest, double[,] dstQuad, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.AffinTransforms.nppiWarpAffineQuad_32f_C1R(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, dstQuad, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpAffineQuad_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
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
		public void WarpPerspective(NPPImage_32fC1 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspective_32f_C1R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspective_32f_C1R", status));
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
		public void WarpPerspectiveBack(NPPImage_32fC1 dest, double[,] coeffs, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspectiveBack_32f_C1R(_devPtr, _sizeOriginal, _pitch, rectIn, dest.DevicePointer, dest.Pitch, rectOut, coeffs, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveBack_32f_C1R", status));
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
		public void WarpPerspectiveQuad(double[,] srcQuad, NPPImage_32fC1 dest, double[,] destQuad, InterpolationMode eInterpolation)
		{
			NppiRect rectIn = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect rectOut = new NppiRect(dest.PointRoi, dest.SizeRoi);
			status = NPPNativeMethods.NPPi.PerspectiveTransforms.nppiWarpPerspectiveQuad_32f_C1R(_devPtr, _sizeOriginal, _pitch, rectIn, srcQuad, dest.DevicePointer, dest.Pitch, rectOut, destQuad, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWarpPerspectiveQuad_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion
		
		#region AbsDiff
		/// <summary>
		/// Absolute difference of this minus src2.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void AbsDiff(NPPImage_32fC1 src2, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.AbsDiff.nppiAbsDiff_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbsDiff_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Absolute difference with constant.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		public void AbsDiff(float nConstant, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.AbsDiffConst.nppiAbsDiffC_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nConstant);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbsDiffC_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Abs
		/// <summary>
		/// Image absolute value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Abs(NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Abs.nppiAbs_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image absolute value. In place.
		/// </summary>
		public void Abs()
		{
			status = NPPNativeMethods.NPPi.Abs.nppiAbs_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region AddProduct
		/// <summary>
		/// Image product added to in place floating point destination image.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		public void AddProduct(NPPImage_32fC1 src2, NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.AddProduct.nppiAddProduct_32f_C1IR(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddProduct_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="mask">Mask image</param>
		public void AddProduct(NPPImage_32fC1 src2, NPPImage_32fC1 dest, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.AddProduct.nppiAddProduct_32f_C1IMR(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, mask.DevicePointerRoi, mask.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddProduct_32f_C1IMR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region AddSquare
		/// <summary>
		/// Image squared then added to in place floating point destination image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void AddProduct(NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.AddSquare.nppiAddSquare_32f_C1IR(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddSquare_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="mask">Mask image</param>
		public void AddProduct(NPPImage_32fC1 dest, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.AddSquare.nppiAddSquare_32f_C1IMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddSquare_32f_C1IMR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region AddWeighted
		/// <summary>
		/// Channel alpha weighted image added to in place floating point destination image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
		public void AddWeighted(NPPImage_32fC1 dest, float nAlpha)
		{
			status = NPPNativeMethods.NPPi.AddWeighted.nppiAddWeighted_32f_C1IR(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAlpha);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddWeighted_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="mask">Mask image</param>
		/// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
		public void AddWeighted(NPPImage_32fC1 dest, NPPImage_8uC1 mask, float nAlpha)
		{
			status = NPPNativeMethods.NPPi.AddWeighted.nppiAddWeighted_32f_C1IMR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nAlpha);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddWeighted_32f_C1IMR", status));
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
		/// Scratch-buffer size for HistogramRange.
		/// </summary>
		/// <param name="nLevels"></param>
		/// <returns></returns>
		public int HistogramRangeGetBufferSize(int nLevels)
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRangeGetBufferSize_32f_C1R(_sizeRoi, nLevels, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRangeGetBufferSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The array must be of size nLevels-1.</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The array must be of size nLevels</param>
		public void HistogramRange(CudaDeviceVariable<int> histogram, CudaDeviceVariable<int> pLevels)
		{
			int bufferSize = HistogramRangeGetBufferSize(histogram.Size);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, histogram.DevicePointer, pLevels.DevicePointer, pLevels.Size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The array must be of size nLevels-1.</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The array must be of size nLevels</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramRangeGetBufferSize"/></param>
		public void HistogramRange(CudaDeviceVariable<int> histogram, CudaDeviceVariable<int> pLevels, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = HistogramRangeGetBufferSize(histogram.Size);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, histogram.DevicePointer, pLevels.DevicePointer, pLevels.Size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sum
		/// <summary>
		/// Scratch-buffer size for nppiSum_32f_C1R.
		/// </summary>
		/// <returns></returns>
		public int SumGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Sum.nppiSumGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image sum with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 1 * sizeof(double)</param>
		public void Sum(CudaDeviceVariable<double> result)
		{
			int bufferSize = SumGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Sum.nppiSum_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SumGetBufferHostSize()"/></param>
		public void Sum(CudaDeviceVariable<double> result, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = SumGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Sum.nppiSum_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.Min.nppiMinGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		public void Min(CudaDeviceVariable<float> min)
		{
			int bufferSize = MinGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Min.nppiMin_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinGetBufferHostSize()"/></param>
		public void Min(CudaDeviceVariable<float> min, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Min.nppiMin_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndxGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndxGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		public void MinIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MinIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinIndexGetBufferHostSize()"/></param>
		public void MinIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.Max.nppiMaxGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		public void Max(CudaDeviceVariable<float> max)
		{
			int bufferSize = MaxGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Max.nppiMax_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxGetBufferHostSize()"/></param>
		public void Max(CudaDeviceVariable<float> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Max.nppiMax_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndxGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndxGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		public void MaxIndex(CudaDeviceVariable<float> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MaxIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="indexX">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 1 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxIndexGetBufferHostSize()"/></param>
		public void MaxIndex(CudaDeviceVariable<float> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMaxGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		public void MinMax(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max)
		{
			int bufferSize = MinMaxGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxGetBufferHostSize()"/></param>
		public void MinMax(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndxGetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndxGetBufferHostSize_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		public void MinMaxIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexGetBufferHostSize()"/></param>
		public void MinMaxIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void MinMaxIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C1MR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum values with their indices. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="max">Allocated device memory with size of at least 1 * sizeof(float)</param>
		/// <param name="minIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="maxIndex">Allocated device memory with size of at least 1 * sizeof(NppiPoint)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxIndexMaskedGetBufferHostSize()"/></param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void MinMaxIndex(CudaDeviceVariable<float> min, CudaDeviceVariable<float> max, CudaDeviceVariable<NppiPoint> minIndex, CudaDeviceVariable<NppiPoint> maxIndex, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxIndxNew.nppiMinMaxIndx_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, minIndex.DevicePointer, maxIndex.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxIndx_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.MeanNew.nppiMeanGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.MeanNew.nppiMeanGetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		public void Mean(CudaDeviceVariable<double> mean)
		{
			int bufferSize = MeanGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanGetBufferHostSize()"/></param>
		public void Mean(CudaDeviceVariable<double> mean, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void Mean(CudaDeviceVariable<double> mean, NPPImage_8uC1 mask)
		{
			int bufferSize = MeanMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C1MR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanMaskedGetBufferHostSize()"/></param>
		/// <param name="mask">If the mask is filled with zeros, then all the returned values are zeros, i.e., pMinIndex = {0, 0}, pMaxIndex = {0, 0}, pMinValue = 0, pMaxValue = 0.</param>
		public void Mean(CudaDeviceVariable<double> mean, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMeanStdDevGetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanStdDevGetBufferHostSize_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev)
		{
			int bufferSize = MeanStdDevGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanStdDevGetBufferHostSize()"/></param>
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanStdDevGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean and standard deviation. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="stdDev">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C1MR", status));
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
		public void MeanStdDev(CudaDeviceVariable<double> mean, CudaDeviceVariable<double> stdDev, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanStdDevMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanStdDevNew.nppiMean_StdDev_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer, stdDev.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_StdDev_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormInf.nppiNormInfGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormInf.nppiNormInfGetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		public void NormInf(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfGetBufferHostSize()"/></param>
		public void NormInf(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void NormInf(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C1MR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfMaskedGetBufferHostSize()"/></param>
		public void NormInf(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormInfMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormL1.nppiNormL1GetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormL1.nppiNormL1GetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		public void NormL1(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1GetBufferHostSize()"/></param>
		public void NormL1(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void NormL1(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C1MR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1MaskedGetBufferHostSize()"/></param>
		public void NormL1(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL1MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormL2.nppiNormL2GetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormL2.nppiNormL2GetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		public void NormL2(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2GetBufferHostSize()"/></param>
		public void NormL2(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		public void NormL2(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C1MR", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 1 * sizeof(double)</param>
		/// <param name="mask">mask</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2MaskedGetBufferHostSize()"/></param>
		public void NormL2(CudaDeviceVariable<double> norm, NPPImage_8uC1 mask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL2MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_32f_C1MR(_devPtrRoi, _pitch, mask.DevicePointerRoi, mask.Pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_32f_C1MR", status));
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
		public void Threshold(NPPImage_32fC1 dest, float nThreshold, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		public void Threshold(float nThreshold, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_32f_C1IR", status));
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
		public void ThresholdGT(NPPImage_32fC1 dest, float nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdGT(float nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_32f_C1IR", status));
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
		public void ThresholdLT(NPPImage_32fC1 dest, float nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdLT(float nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_32f_C1IR", status));
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
		public void Threshold(NPPImage_32fC1 dest, float nThreshold, float nValue, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_32f_C1R", status));
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
		public void Threshold(float nThreshold, float nValue, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_32f_C1IR", status));
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
		public void ThresholdGT(NPPImage_32fC1 dest, float nThreshold, float nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdGT(float nThreshold, float nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_32f_C1IR", status));
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
		public void ThresholdLT(NPPImage_32fC1 dest, float nThreshold, float nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdLT(float nThreshold, float nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_32f_C1IR", status));
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
		public void ThresholdLTGT(NPPImage_32fC1 dest, float nThresholdLT, float nValueLT, float nThresholdGT, float nValueGT)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_32f_C1R", status));
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
		public void ThresholdLTGT(float nThresholdLT, float nValueLT, float nThresholdGT, float nValueGT)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_32f_C1IR", status));
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
		public void Compare(NPPImage_32fC1 src2, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompare_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompare_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc's pixels with constant value.
		/// </summary>
		/// <param name="nConstant">constant value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		public void Compare(float nConstant, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompareC_32f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareC_32f_C1R", status));
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
		public void CompareEqualEps(NPPImage_32fC1 src2, NPPImage_8uC1 dest, float epsilon)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompareEqualEps_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, epsilon);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareEqualEps_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc's pixels with constant value.
		/// </summary>
		/// <param name="nConstant">constant value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="epsilon">epsilon tolerance value to compare to pixel absolute differences.</param>
		public void CompareEqualEps(float nConstant, NPPImage_8uC1 dest, float epsilon)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompareEqualEpsC_32f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, epsilon);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareEqualEpsC_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//new in Cuda 5.5
		#region DotProduct
		/// <summary>
		/// Device scratch buffer size (in bytes) for nppiDotProd_32f64f_C1R.
		/// </summary>
		/// <returns></returns>
		public int DotProdGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.DotProd.nppiDotProdGetBufferHostSize_32f64f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProdGetBufferHostSize_32f64f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// One-channel 32-bit floating point image DotProd.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="DotProdGetBufferHostSize()"/></param>
		public void DotProduct(NPPImage_32fC1 src2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = DotProdGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_32f64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_32f64f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// One-channel 32-bit floating point DotProd. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (1 * sizeof(double))</param>
		public void DotProduct(NPPImage_32fC1 src2, CudaDeviceVariable<double> pDp)
		{
			int bufferSize = DotProdGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_32f64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_32f64f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region CopyNew

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
		public void Copy(NPPImage_32fC1 dst, int nTopBorderHeight, int nLeftBorderWidth, byte nValue)
		{
			status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_32f_C1R", status));
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
		public void CopyReplicateBorder(NPPImage_32fC1 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_32f_C1R", status));
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
		public void CopyWrapBorder(NPPImage_32fC1 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// linearly interpolated source image subpixel coordinate color copy.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nDx">Fractional part of source image X coordinate.</param>
		/// <param name="nDy">Fractional part of source image Y coordinate.</param>
		public void CopySubpix(NPPImage_32fC1 dst, float nDx, float nDy)
		{
			status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_8u_C1R", status));
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
		public void LUT(NPPImage_32fC1 dst, CudaDeviceVariable<float> pValues, CudaDeviceVariable<float> pLevels)
		{
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubic(NPPImage_32fC1 dst, CudaDeviceVariable<float> pValues, CudaDeviceVariable<float> pLevels)
		{
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// Inplace look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUT(CudaDeviceVariable<float> pValues, CudaDeviceVariable<float> pLevels)
		{
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubic(CudaDeviceVariable<float> pValues, CudaDeviceVariable<float> pLevels)
		{
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace linear interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Pointer to an array of user defined OUTPUT values</param>
		/// <param name="pLevels">Pointer to an array of user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTLinear(CudaDeviceVariable<float> pValues, CudaDeviceVariable<float> pLevels)
		{
			status = NPPNativeMethods.NPPi.ColorLUTLinear.nppiLUT_Linear_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, pValues.DevicePointer, pLevels.DevicePointer, pLevels.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		//nppiLUT_Linear_32f_C1R

		/// <summary>
		/// look-up-table color conversion.<para/>
		/// The LUT is derived from a set of user defined mapping points through linear interpolation.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="values">array of user defined OUTPUT values</param>
		/// <param name="levels">array of user defined INPUT values</param>
		public void LUTLinear(NPPImage_32fC1 dest, CudaDeviceVariable<float> values, CudaDeviceVariable<float> levels)
		{
			if (values.Size != levels.Size) throw new ArgumentException("values and levels must have same size.");

			status = NPPNativeMethods.NPPi.ColorLUTLinear.nppiLUT_Linear_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, values.DevicePointer, levels.DevicePointer, (int)levels.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_32f_C1R", status));
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
		public void Dilate(NPPImage_32fC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiDilate_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate_32f_C1R", status));
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
		public void Erode(NPPImage_32fC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiErode_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 3x3 dilation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Dilate3x3(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiDilate3x3_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Erode3x3(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.MorphologyFilter2D.nppiErode3x3_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3_32f_C1R", status));
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
		public void DilateBorder(NPPImage_32fC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.DilationWithBorderControl.nppiDilateBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilateBorder_32f_C1R", status));
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
		public void ErodeBorder(NPPImage_32fC1 dest, CudaDeviceVariable<byte> Mask, NppiSize aMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.ErosionWithBorderControl.nppiErodeBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Mask.DevicePointer, aMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErodeBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 dilation with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void Dilate3x3Border(NPPImage_32fC1 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.Dilate3x3Border.nppiDilate3x3Border_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDilate3x3Border_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 3x3 erosion with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void Erode3x3Border(NPPImage_32fC1 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.Erode3x3Border.nppiErode3x3Border_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiErode3x3Border_32f_C1R", status));
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
		public void FilterColumn(NPPImage_32fC1 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumn_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 1D row convolution.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterRow(NPPImage_32fC1 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRow_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 1D row convolution with border control. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRowBorder(NPPImage_32fC1 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRowBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder_32f_C1R", status));
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
		public void Filter(NPPImage_32fC1 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittHoriz(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHoriz_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHoriz_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittVert(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVert_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVert_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void SobelHoriz(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelHoriz_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHoriz_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSobelVert(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelVert_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVert_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// second derivative, horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterSobelHorizSecond(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelHorizSecond_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizSecond_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// second derivative, vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterSobelVertSecond(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelVertSecond_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertSecond_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// second cross derivative Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterSobelCross(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelCross_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelCross_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsDown(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDown_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDown_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter..
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsUp(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUp_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUp_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLaplace(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplace_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplace_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Gauss filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterGauss(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGauss_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGauss_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterHighPass(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPass_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPass_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLowPass(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPass_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPass_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSharpen(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpen_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpen_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Scharr filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterScharrHoriz(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterScharrHoriz_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrHoriz_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Scharr filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterScharrVert(NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterScharrVert_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrVert_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterBox(NPPImage_32fC1 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBox_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBox_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMin(NPPImage_32fC1 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMin_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMin_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMax(NPPImage_32fC1 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMax_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMax_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterSobelHoriz(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelHorizMask_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizMask_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterSobelVert(NPPImage_32fC1 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelVertMask_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertMask_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Scharr filter kernel with border control.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterScharrHoriz(NPPImage_32fC1 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterScharrHorizBorder.nppiFilterScharrHorizBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrHoriz_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_32f_C1MR", status));
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
		public void NormDiff_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		public void NormDiff_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormDiffInfMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_32f_C1MR", status));
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
		public void NormDiff_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		public void NormDiff_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormDiffL1MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_32f_C1MR", status));
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
		public void NormDiff_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		public void NormDiff_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormDiffL2MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_32f_C1MR", status));
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
		public void NormRel_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		public void NormRel_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormRelInfMaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_32f_C1MR", status));
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
		public void NormRel_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		public void NormRel_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormRelL1MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C1MR", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_32f_C1MR(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_32f_C1MR", status));
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
		public void NormRel_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C1MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="pMask">Mask image.</param>
		public void NormRel_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, NPPImage_8uC1 pMask)
		{
			int bufferSize = NormRelL2MaskedGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_32f_C1MR(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, pMask.DevicePointerRoi, pMask.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C1MR", status));
			buffer.Dispose();
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfGetBufferHostSize()"/></param>
		public void NormDiff_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (1 * sizeof(double))</param>
		public void NormDiff_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1GetBufferHostSize()"/></param>
		public void NormDiff_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (1 * sizeof(double))</param>
		public void NormDiff_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2GetBufferHostSize()"/></param>
		public void NormDiff_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (1 * sizeof(double))</param>
		public void NormDiff_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfGetBufferHostSize()"/></param>
		public void NormRel_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		public void NormRel_Inf(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1GetBufferHostSize()"/></param>
		public void NormRel_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		public void NormRel_L1(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2GetBufferHostSize()"/></param>
		public void NormRel_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (1 * sizeof(double))</param>
		public void NormRel_L2(NPPImage_32fC1 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_32f_C1R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrFull_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="FullNormLevelGetBufferHostSize()"/></param>
		public void CrossCorrFull_NormLevel(NPPImage_32fC1 tpl, NPPImage_32fC1 dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = FullNormLevelGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		public void CrossCorrFull_NormLevel(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			int bufferSize = FullNormLevelGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrSame_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SameNormLevelGetBufferHostSize()"/></param>
		public void CrossCorrSame_NormLevel(NPPImage_32fC1 tpl, NPPImage_32fC1 dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = SameNormLevelGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		public void CrossCorrSame_NormLevel(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			int bufferSize = SameNormLevelGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// CrossCorrValid_NormLevel.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="ValidNormLevelGetBufferHostSize()"/></param>
		public void CrossCorrValid_NormLevel(NPPImage_32fC1 tpl, NPPImage_32fC1 dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination image</param>
		public void CrossCorrValid_NormLevel(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			int bufferSize = ValidNormLevelGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}








		/// <summary>
		/// image SqrDistanceFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void SqrDistanceFull_Norm(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceFull_Norm_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceFull_Norm_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void SqrDistanceSame_Norm(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceSame_Norm_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceSame_Norm_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image SqrDistanceValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void SqrDistanceValid_Norm(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiSqrDistanceValid_Norm_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrDistanceValid_Norm_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// image CrossCorrFull_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void CrossCorrFull_Norm(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_Norm_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrSame_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void CrossCorrSame_Norm(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_Norm_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid_Norm.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void CrossCorrValid_Norm(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_Norm_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image CrossCorrValid.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="dst">Destination-Image</param>
		public void CrossCorrValid(NPPImage_32fC1 tpl, NPPImage_32fC1 dst)
		{
			status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.CountInRange.nppiCountInRangeGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRangeGetBufferHostSize_32f_C1R", status));
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
		public void CountInRange(CudaDeviceVariable<int> pCounts, float nLowerBound, float nUpperBound, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = CountInRangeGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.CountInRange.nppiCountInRange_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image CountInRange.
		/// </summary>
		/// <param name="pCounts">Pointer to the number of pixels that fall into the specified range. (1 * sizeof(int))</param>
		/// <param name="nLowerBound">Lower bound of the specified range.</param>
		/// <param name="nUpperBound">Upper bound of the specified range.</param>
		public void CountInRange(CudaDeviceVariable<int> pCounts, float nLowerBound, float nUpperBound)
		{
			int bufferSize = CountInRangeGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.CountInRange.nppiCountInRange_32f_C1R(_devPtrRoi, _pitch, _sizeRoi, pCounts.DevicePointer, nLowerBound, nUpperBound, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCountInRange_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.QualityIndex.nppiQualityIndexGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndexGetBufferHostSize_32f_C3R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (1 * sizeof(float))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="QualityIndexGetBufferHostSize()"/></param>
		public void QualityIndex(NPPImage_32fC1 src2, CudaDeviceVariable<float> dst, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = QualityIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.QualityIndex.nppiQualityIndex_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image QualityIndex.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dst">Pointer to the quality index. (1 * sizeof(float))</param>
		public void QualityIndex(NPPImage_32fC1 src2, CudaDeviceVariable<float> dst)
		{
			int bufferSize = QualityIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.QualityIndex.nppiQualityIndex_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, dst.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiQualityIndex_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMaxEveryNew
		/// <summary>
		/// image MinEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MinEvery(NPPImage_32fC1 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMinEvery_32f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinEvery_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image MaxEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MaxEvery(NPPImage_32fC1 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMaxEvery_32f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxEvery_32f_C1IR", status));
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
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_32f_C1IR(_devPtrRoi, _pitch, _sizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_32f_C1IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region DupNew
		/// <summary>
		/// source image duplicated in all 3 channels of destination image.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Dup(NPPImage_32fC3 dst)
		{
			status = NPPNativeMethods.NPPi.Dup.nppiDup_32f_C1C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_32f_C1C3R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// source image duplicated in all 4 channels of destination image.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void Dup(NPPImage_32fC4 dst)
		{
			status = NPPNativeMethods.NPPi.Dup.nppiDup_32f_C1C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_32f_C1C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void DupA(NPPImage_32fC4 dst)
		{
			status = NPPNativeMethods.NPPi.Dup.nppiDup_32f_C1AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_32f_C1AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Transpose
		/// <summary>
		/// image transpose
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Transpose(NPPImage_32fC1 dest)
		{
			status = NPPNativeMethods.NPPi.Transpose.nppiTranspose_32f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_32f_C1R", status));
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
		public void ResizeSqrPixel(NPPImage_32fC1 dst, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect dstRect = new NppiRect(dst.PointRoi, dst.SizeRoi);
			status = NPPNativeMethods.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_32f_C1R(_devPtr, _sizeRoi, _pitch, srcRect, dst.DevicePointer, dst.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image remap.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. </param>
		/// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. </param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public void Remap(NPPImage_32fC1 dst, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Remap.nppiRemap_32f_C1R(_devPtr, _sizeRoi, _pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
		/// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
		public void Scale(NPPImage_32fC1 dst, float nMin, float nMax)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Scale.nppiScale_32f8u_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nMin, nMax);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_32f8u_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Resizes images.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="xFactor">X scaling factor</param>
		/// <param name="yFactor">Y scaling factor</param>
		/// <param name="eInterpolation">Interpolation mode</param>
		public void Resize(NPPImage_32fC1 dest, double xFactor, double yFactor, InterpolationMode eInterpolation)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_32f_C1R(_devPtr, _sizeOriginal, _pitch, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, xFactor, yFactor, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region RectStdDev


		/// <summary>
		///  RectStdDev.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="sqr">Destination-Image</param>
		public void RectStdDev(NPPImage_32fC1 dst, NPPImage_32fC1 sqr)
		{
			status = NPPNativeMethods.NPPi.Integral.nppiRectStdDev_32f_C1R(_devPtr, _pitch, sqr.DevicePointer, sqr.Pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, new NppiRect(_pointRoi, _sizeRoi));
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRectStdDev_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//New in Cuda 6.0

		#region Filter Median
		/// <summary>
		/// Result pixel value is the median of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterMedian(NPPImage_32fC1 dst, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_32f_C1R", status));
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
		public void FilterMedian(NPPImage_32fC1 dst, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for FilterMedian.
		/// </summary>
		/// <returns></returns>
		public int FilterMedianGetBufferHostSize(NppiSize oMaskSize)
		{
			uint bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianGetBufferSize_32f_C1R(_sizeRoi, oMaskSize, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianGetBufferSize_32f_C1R", status));
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
		public void MaxError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
		public void MaxError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_32f_C1R", status));
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
		public void AverageError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
		public void AverageError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_32f_C1R", status));
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
		public void MaximumRelativeError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
		public void MaximumRelativeError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_32f_C1R", status));
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
		public void AverageRelativeError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32f_C1R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
		public void AverageRelativeError(NPPImage_32fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_32f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_32f_C1R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region FilterBorder
		/// <summary>
		/// One channel 32-bit float convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_32fC1 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder.nppiFilterBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterScharrBorder
		/// <summary>
		/// Filters the image using a horizontal Scharr filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterScharrHorizBorder(NPPImage_32fC1 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterScharrHorizBorder.nppiFilterScharrHorizBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrHorizBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Scharr filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterScharrVertBorder(NPPImage_32fC1 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterScharrVertBorder.nppiFilterScharrVertBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterScharrVertBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterSobelBorder
		/// <summary>
		/// Filters the image using a horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelHorizBorder(NPPImage_32fC1 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelVertBorder(NPPImage_32fC1 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelVertBorder.nppiFilterSobelVertBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelHorizBorder(NPPImage_32fC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizMaskBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizMaskBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelVertBorder(NPPImage_32fC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelVertBorder.nppiFilterSobelVertMaskBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertMaskBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a second derivative, horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelHorizSecondBorder(NPPImage_32fC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelHorizSecondBorder.nppiFilterSobelHorizSecondBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizSecondBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a second derivative, vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelVertSecondBorder(NPPImage_32fC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelVertSecondBorder.nppiFilterSobelVertSecondBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertSecondBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a second cross derivative Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelCrossBorder(NPPImage_32fC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelCrossBorder.nppiFilterSobelCrossBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelCrossBorder_32f_C1R", status));
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
		public void ColorTwist(NPPImage_32fC1 dest, float[,] twistMatrix)
		{
			status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist_32f_C1R(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist_32f_C1IR(_devPtr, _pitch, _sizeRoi, aTwist);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist_32f_C1IR", status));
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
		public void FilterGaussBorder(NPPImage_32fC1 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussBorder_32f_C1R", status));
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
		public void FilterColumnBorder(NPPImage_32fC1 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumnBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder_32f_C1R", status));
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
		public void FilterBoxBorder(NPPImage_32fC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorder_32f_C1R", status));
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
		public void FilterMinBorder(NPPImage_32fC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMinBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMinBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterMaxBorder(NPPImage_32fC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMaxBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMaxBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterOthers


		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittHorizBorder(NPPImage_32fC1 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHorizBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHorizBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittVertBorder(NPPImage_32fC1 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVertBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVertBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsDownBorder(NPPImage_32fC1 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDownBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDownBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsUpBorder(NPPImage_32fC1 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUpBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUpBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLaplaceBorder(NPPImage_32fC1 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplaceBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplaceBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterHighPassBorder(NPPImage_32fC1 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPassBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPassBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLowPassBorder(NPPImage_32fC1 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPassBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPassBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSharpenBorder(NPPImage_32fC1 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpenBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpenBorder_32f_C1R", status));
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
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpGetBufferSize_32f_C1R(nRadius, nSigma, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpGetBufferSize_32f_C1R", status));
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
		public void FilterUnsharpBorder(NPPImage_32fC1 dst, float nRadius, float nSigma, float nWeight, float nThreshold, NppiBorderType eBorderType, CudaDeviceVariable<byte> buffer)
		{
			if (buffer.Size < FilterUnsharpGetBufferSize(nRadius, nSigma))
				throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpBorder_32f_C1R(_devPtr, _pitch, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nRadius, nSigma, nWeight, nThreshold, eBorderType, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Filter Gauss Advanced

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		public void FilterGauss(NPPImage_32fC1 dst, CudaDeviceVariable<float> Kernel)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGaussAdvanced_32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvanced_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterGaussBorder(NPPImage_32fC1 dst, CudaDeviceVariable<float> Kernel, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussAdvancedBorder_32f_C1R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvancedBorder_32f_C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion
	}
}
