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
	public partial class NPPImage_16sC4 : NPPImageBase
	{
		#region Constructors
		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="nWidthPixels">Image width in pixels</param>
		/// <param name="nHeightPixels">Image height in pixels</param>
		public NPPImage_16sC4(int nWidthPixels, int nHeightPixels)
		{
			_sizeOriginal.width = nWidthPixels;
			_sizeOriginal.height = nHeightPixels;
			_sizeRoi.width = nWidthPixels;
			_sizeRoi.height = nHeightPixels;
			_channels = 4;
			_isOwner = true;
			_typeSize = sizeof(short);

			_devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_16s_C4(nWidthPixels, nHeightPixels, ref _pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_16s_C4", res, _pitch, _channels));
			
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
		public NPPImage_16sC4(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
		{
			_devPtr = devPtr;
			_devPtrRoi = _devPtr;
			_sizeOriginal.width = width;
			_sizeOriginal.height = height;
			_sizeRoi.width = width;
			_sizeRoi.height = height;
			_pitch = pitch;
			_channels = 4;
			_isOwner = isOwner;
			_typeSize = sizeof(short);
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="width">Image width in pixels</param>
		/// <param name="height">Image height in pixels</param>
		/// <param name="pitch">Pitch / Line step</param>
		public NPPImage_16sC4(CUdeviceptr devPtr, int width, int height, int pitch)
			: this(devPtr, width, height, pitch, false)
		{

		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
		/// </summary>
		/// <param name="image">NPP image</param>
		public NPPImage_16sC4(NPPImageBase image)
			: this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
		{

		}

		/// <summary>
		/// Allocates new memory on device using NPP-Api.
		/// </summary>
		/// <param name="size">Image size</param>
		public NPPImage_16sC4(NppiSize size)
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
		public NPPImage_16sC4(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
			: this(devPtr, size.width, size.height, pitch, isOwner)
		{ 
			
		}

		/// <summary>
		/// Creates a new NPPImage from allocated device ptr.
		/// </summary>
		/// <param name="devPtr">Already allocated device ptr.</param>
		/// <param name="size">Image size</param>
		/// <param name="pitch">Pitch / Line step</param>
		public NPPImage_16sC4(CUdeviceptr devPtr, NppiSize size, int pitch)
			: this(devPtr, size.width, size.height, pitch)
		{

		}

		/// <summary>
		/// For dispose
		/// </summary>
		~NPPImage_16sC4()
		{
			Dispose (false);
		}
		#endregion

		#region Converter operators

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		public CudaPitchedDeviceVariable<VectorTypes.short4> ToCudaPitchedDeviceVariable()
		{
			return new CudaPitchedDeviceVariable<VectorTypes.short4>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
		}

		/// <summary>
		/// Converts a NPPImage to a CudaPitchedDeviceVariable
		/// </summary>
		/// <param name="img">NPPImage</param>
		/// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
		public static implicit operator CudaPitchedDeviceVariable<VectorTypes.short4>(NPPImage_16sC4 img)
		{
			return img.ToCudaPitchedDeviceVariable();
		}

		/// <summary>
		/// Converts a CudaPitchedDeviceVariable to a NPPImage 
		/// </summary>
		/// <param name="img">CudaPitchedDeviceVariable</param>
		/// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
		public static implicit operator NPPImage_16sC4(CudaPitchedDeviceVariable<VectorTypes.short4> img)
		{
			return img.ToNPPImage();
		}
		#endregion

		#region Copy
		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channel">Channel number. This number is added to the dst pointer</param>
		public void Copy(NPPImage_16sC1 dst, int channel)
		{
			if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..3].");
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C4C1R(_devPtrRoi + channel * _typeSize, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C4C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="channelSrc">Channel number. This number is added to the src pointer</param>
		/// <param name="channelDst">Channel number. This number is added to the dst pointer</param>
		public void Copy(NPPImage_16sC4 dst, int channelSrc, int channelDst)
		{
			if (channelSrc < 0 | channelSrc >= _channels) throw new ArgumentOutOfRangeException("channelSrc", "channelSrc must be in range [0..2].");
			if (channelDst < 0 | channelDst >= dst.Channels) throw new ArgumentOutOfRangeException("channelDst", "channelDst must be in range [0..2].");
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C4CR(_devPtrRoi + channelSrc * _typeSize, _pitch, dst.DevicePointerRoi + channelDst * _typeSize, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C4CR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="mask">Mask image</param>
		public void Copy(NPPImage_16sC4 dst, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C4MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C4MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy. Not affecting Alpha channel.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="mask">Mask image</param>
		public void CopyA(NPPImage_16sC4 dst, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_AC4MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_AC4MR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Copy(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Masked Operation 8-bit unsigned image copy. Not affecting Alpha channel.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void CopyA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 8-bit unsigned packed to planar image copy.
		/// </summary>
		/// <param name="dst0">Destination image channel 0</param>
		/// <param name="dst1">Destination image channel 1</param>
		/// <param name="dst2">Destination image channel 2</param>
		/// <param name="dst3">Destination image channel 3</param>
		public void Copy(NPPImage_16sC1 dst0, NPPImage_16sC1 dst1, NPPImage_16sC1 dst2, NPPImage_16sC1 dst3)
		{
			CUdeviceptr[] array = new CUdeviceptr[] { dst0.DevicePointerRoi, dst1.DevicePointerRoi, dst2.DevicePointerRoi, dst3.DevicePointerRoi };
			status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C4P4R(_devPtrRoi, _pitch, array, dst0.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C4P4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Three-channel 8-bit unsigned planar to packed image copy.
		/// </summary>
		/// <param name="src0">Source image channel 0</param>
		/// <param name="src1">Source image channel 1</param>
		/// <param name="src2">Source image channel 2</param>
		/// <param name="src3">Source image channel 2</param>
		/// <param name="dest">Destination image</param>
		public static void Copy(NPPImage_16sC1 src0, NPPImage_16sC1 src1, NPPImage_16sC1 src2, NPPImage_16sC1 src3, NPPImage_16sC4 dest)
		{
			CUdeviceptr[] array = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi, src3.DevicePointerRoi };
			NppStatus status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_P4C4R(array, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_P4C4R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set (Array size = 4)</param>
		public void Set(short[] nValue)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_C4R(nValue, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Set pixel values to nValue. <para/>
		/// The 8-bit mask image affects setting of the respective pixels in the destination image. <para/>
		/// If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
		/// destination pixel is set to specified value.
		/// </summary>
		/// <param name="nValue">Value to be set (Array size = 4)</param>
		/// <param name="mask">Mask image</param>
		public void Set(short[] nValue, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_C4MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_C4MR", status));
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
		public void Set(short nValue, int channel)
		{
			if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..3].");
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_C4CR(nValue, _devPtrRoi + channel * _typeSize, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_C4CR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Set pixel values to nValue. <para/>
		/// The 8-bit mask image affects setting of the respective pixels in the destination image. <para/>
		/// If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
		/// destination pixel is set to specified value. Not affecting alpha channel.
		/// </summary>
		/// <param name="nValue">Value to be set (Array size = 3)</param>
		public void SetA(short[] nValue)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_AC4R(nValue, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Set pixel values to nValue. <para/>
		/// The 8-bit mask image affects setting of the respective pixels in the destination image. <para/>
		/// If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
		/// destination pixel is set to specified value. Not affecting alpha channel.
		/// </summary>
		/// <param name="nValue">Value to be set (Array size = 3)</param>
		/// <param name="mask">Mask image</param>
		public void SetA(short[] nValue, NPPImage_8uC1 mask)
		{
			status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_AC4MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_AC4MR", status));
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
		public void Add(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_16s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Add(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_16s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Add(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Add(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image addition, scale by 2^(-nScaleFactor), then clamp to saturated value. Unmodified Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void AddA(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_16s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value. Unmodified Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void AddA(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Add.nppiAdd_16s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unmodified Alpha.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void AddA(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unmodified Alpha.
		/// </summary>
		/// <param name="nConstant">Values to add</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void AddA(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Logical

		/// <summary>
		/// image bit shift by constant (right).
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 4)</param>
		/// <param name="dest">Destination image</param>
		public void RShiftC(uint[] nConstant, NPPImage_16sC4 dest)
		{
			status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_16s_C4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (right), inplace.
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 4)</param>
		public void RShiftC(uint[] nConstant)
		{
			status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_16s_C4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_16s_C4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (right). Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 4)</param>
		/// <param name="dest">Destination image</param>
		public void RShiftCA(uint[] nConstant, NPPImage_16sC4 dest)
		{
			status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_16s_AC4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image bit shift by constant (right), inplace. Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Constant (Array length = 4)</param>
		public void RShiftCA(uint[] nConstant)
		{
			status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_16s_AC4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_16s_AC4IR", status));
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
		public void Sub(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_16s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sub(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_16s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sub(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sub(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SubA(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_16s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SubA(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sub.nppiSub_16s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SubA(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SubA(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16s_AC4IRSfs", status));
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
		public void Mul(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_16s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Mul(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_16s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Mul(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Mul(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void MulA(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_16s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void MulA(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Mul.nppiMul_16s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void MulA(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void MulA(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16s_AC4IRSfs", status));
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
		public void Div(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_16s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_16s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(NPPImage_16sC4 src2, NPPImage_16sC4 dest, NppRoundMode rndMode, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_16s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, rndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Div(NPPImage_16sC4 src2, NppRoundMode rndMode, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_16s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, rndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void DivA(NPPImage_16sC4 src2, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_16s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void DivA(NPPImage_16sC4 src2, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Div.nppiDiv_16s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void DivA(short[] nConstant, NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unchanged Alpha.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void DivA(short[] nConstant, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void DivA(NPPImage_16sC4 src2, NPPImage_16sC4 dest, NppRoundMode rndMode, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_16s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, rndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="rndMode">Result Rounding mode to be used</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void DivA(NPPImage_16sC4 src2, NppRoundMode rndMode, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_16s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, rndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqr
		/// <summary>
		/// Image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqr(NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16s_C4RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqr(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16s_C4IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image squared, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SqrA(NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16s_AC4RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image squared, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SqrA(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16s_AC4IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sqrt
		/// <summary>
		/// Image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqrt(NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16s_C4RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16s_C4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void Sqrt(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16s_C4IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16s_C4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image square root, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SqrtA(NPPImage_16sC4 dest, int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16s_AC4RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16s_AC4RSfs", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace image square root, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
		/// </summary>
		/// <param name="nScaleFactor">scaling factor</param>
		public void SqrtA(int nScaleFactor)
		{
			status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16s_AC4IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16s_AC4IRSfs", status));
			NPPException.CheckNppStatus(status, this);
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
			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEvenGetBufferSize_16s_C4R(_sizeRoi, nLevels, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEvenGetBufferSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for HistogramEven. Not affecting Alpha channel. 
		/// </summary>
		/// <param name="nLevels"></param>
		/// <returns></returns>
		public int HistogramEvenGetBufferSizeA(int[] nLevels)
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEvenGetBufferSize_16s_AC4R(_sizeRoi, nLevels, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEvenGetBufferSize_16s_AC4R", status));
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
		/// <param name="histogram">Allocated device memory of size nLevels (4 Variables)</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]. Size = 4</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]. Size = 4</param>
		public void HistogramEven(CudaDeviceVariable<int>[] histogram, int[] nLowerLevel, int[] nUpperLevel)
		{
			int[] size = new int[] { histogram[0].Size + 1, histogram[1].Size + 1, histogram[2].Size + 1, histogram[3].Size + 1 };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer, histogram[3].DevicePointer };


			int bufferSize = HistogramEvenGetBufferSize(size);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEven_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, size, nLowerLevel, nUpperLevel, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with evenly distributed bins. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">Allocated device memory of size nLevels (4 Variables)</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]. Size = 4</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]. Size = 4</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramEvenGetBufferSize(int[])"/></param>
		public void HistogramEven(CudaDeviceVariable<int>[] histogram, int[] nLowerLevel, int[] nUpperLevel, CudaDeviceVariable<byte> buffer)
		{
			int[] size = new int[] { histogram[0].Size + 1, histogram[1].Size + 1, histogram[2].Size + 1, histogram[3].Size + 1 };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer, histogram[3].DevicePointer };

			int bufferSize = HistogramEvenGetBufferSize(size);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEven_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, size, nLowerLevel, nUpperLevel, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with evenly distributed bins. Buffer is internally allocated and freed. Alpha channel is ignored during the histograms computations.
		/// </summary>
		/// <param name="histogram">Allocated device memory of size nLevels (3 Variables)</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]. Size = 3</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]. Size = 3</param>
		public void HistogramEvenA(CudaDeviceVariable<int>[] histogram, int[] nLowerLevel, int[] nUpperLevel)
		{
			int[] size = new int[] { histogram[0].Size + 1, histogram[1].Size + 1, histogram[2].Size + 1 };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };


			int bufferSize = HistogramEvenGetBufferSizeA(size);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEven_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, size, nLowerLevel, nUpperLevel, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with evenly distributed bins. No additional buffer is allocated. Alpha channel is ignored during the histograms computations.
		/// </summary>
		/// <param name="histogram">Allocated device memory of size nLevels (3 Variables)</param>
		/// <param name="nLowerLevel">Lower boundary of lowest level bin. E.g. 0 for [0..255]. Size = 3</param>
		/// <param name="nUpperLevel">Upper boundary of highest level bin. E.g. 256 for [0..255]. Size = 3</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramEvenGetBufferSize(int[])"/></param>
		public void HistogramEvenA(CudaDeviceVariable<int>[] histogram, int[] nLowerLevel, int[] nUpperLevel, CudaDeviceVariable<byte> buffer)
		{
			int[] size = new int[] { histogram[0].Size + 1, histogram[1].Size + 1, histogram[2].Size + 1 };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };

			int bufferSize = HistogramEvenGetBufferSizeA(size);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramEven_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, size, nLowerLevel, nUpperLevel, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramEven_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRangeGetBufferSize_16s_C4R(_sizeRoi, nLevels, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRangeGetBufferSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for HistogramRange. Not affecting Alpha channel.
		/// </summary>
		/// <param name="nLevels"></param>
		/// <returns></returns>
		public int HistogramRangeGetBufferSizeA(int[] nLevels)
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRangeGetBufferSize_16s_AC4R(_sizeRoi, nLevels, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRangeGetBufferSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 4</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 4</param>
		public void HistogramRange(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels)
		{
			int[] size = new int[] { histogram[0].Size, histogram[1].Size, histogram[2].Size, histogram[3].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer, histogram[3].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer, pLevels[3].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSize(size);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. No additional buffer is allocated.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 4</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 4</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramRangeGetBufferSize(int[])"/></param>
		public void HistogramRange(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels, CudaDeviceVariable<byte> buffer)
		{
			int[] size = new int[] { histogram[0].Size, histogram[1].Size, histogram[2].Size, histogram[3].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer, histogram[3].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer, pLevels[3].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSize(size);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. Buffer is internally allocated and freed. Alpha channel is ignored during the histograms computations.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 3</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 3</param>
		public void HistogramRangeA(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels)
		{
			int[] size = new int[] { histogram[0].Size, histogram[1].Size, histogram[2].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSizeA(size);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Histogram with bins determined by pLevels array. No additional buffer is allocated. Alpha channel is ignored during the histograms computations.
		/// </summary>
		/// <param name="histogram">array that receives the computed histogram. The CudaDeviceVariable must be of size nLevels-1. Array size = 3</param>
		/// <param name="pLevels">Array in device memory containing the level sizes of the bins. The CudaDeviceVariable must be of size nLevels. Array size = 3</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="HistogramRangeGetBufferSize(int[])"/></param>
		public void HistogramRangeA(CudaDeviceVariable<int>[] histogram, CudaDeviceVariable<int>[] pLevels, CudaDeviceVariable<byte> buffer)
		{
			int[] size = new int[] { histogram[0].Size, histogram[1].Size, histogram[2].Size };
			CUdeviceptr[] devPtrs = new CUdeviceptr[] { histogram[0].DevicePointer, histogram[1].DevicePointer, histogram[2].DevicePointer };
			CUdeviceptr[] devLevels = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };

			int bufferSize = HistogramRangeGetBufferSizeA(size);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Histogram.nppiHistogramRange_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, devPtrs, devLevels, size, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiHistogramRange_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Convert
		/// <summary>
		/// 16-bit unsigned to 8-bit unsigned conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Convert(NPPImage_8uC4 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s8u_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s8u_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 16-bit unsigned to 8-bit unsigned conversion. Not Affecting alpha channel
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void ConvertA(NPPImage_8uC4 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s8u_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s8u_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 16-bit unsigned to 32-bit signed conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Convert(NPPImage_32sC4 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s32s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s32s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 16-bit unsigned to 32-bit float conversion.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void Convert(NPPImage_32fC4 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s32f_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s32f_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 16-bit unsigned to 32-bit signed conversion. Not Affecting alpha channel.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void ConvertA(NPPImage_32sC4 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s32s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s32s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 16-bit unsigned to 32-bit float conversion. Not Affecting alpha channel.
		/// </summary>
		/// <param name="dst">Destination image</param>
		public void ConvertA(NPPImage_32fC4 dst)
		{
			status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s32f_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s32f_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Abs
		/// <summary>
		/// Image absolute value.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Abs(NPPImage_16sC4 dest)
		{
			status = NPPNativeMethods.NPPi.Abs.nppiAbs_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image absolute value. In place.
		/// </summary>
		public void Abs()
		{
			status = NPPNativeMethods.NPPi.Abs.nppiAbs_16s_C4IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16s_C4IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image absolute value. Not affecting Alpha channel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void AbsA(NPPImage_16sC4 dest)
		{
			status = NPPNativeMethods.NPPi.Abs.nppiAbs_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image absolute value. In place. Not affecting Alpha channel.
		/// </summary>
		public void AbsA()
		{
			status = NPPNativeMethods.NPPi.Abs.nppiAbs_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sum
		/// <summary>
		/// Scratch-buffer size for nppiSum_16s_C4R.
		/// </summary>
		/// <returns></returns>
		public int SumGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Sum.nppiSumGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image sum with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 4 * sizeof(double)</param>
		public void Sum(CudaDeviceVariable<double> result)
		{
			int bufferSize = SumGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Sum.nppiSum_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 4 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SumGetBufferHostSize()"/></param>
		public void Sum(CudaDeviceVariable<double> result, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = SumGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Sum.nppiSum_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for nppiSum_16s_C4R. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int SumGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Sum.nppiSumGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image sum with 64-bit double precision result. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void SumA(CudaDeviceVariable<double> result)
		{
			int bufferSize = SumGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Sum.nppiSum_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image sum with 64-bit double precision result. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="result">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="SumGetBufferHostSizeA()"/></param>
		public void SumA(CudaDeviceVariable<double> result, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = SumGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Sum.nppiSum_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSum_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.Min.nppiMinGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 4 * sizeof(short)</param>
		public void Min(CudaDeviceVariable<short> min)
		{
			int bufferSize = MinGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Min.nppiMin_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinGetBufferHostSize()"/></param>
		public void Min(CudaDeviceVariable<short> min, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Min.nppiMin_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for Min. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int MinGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Min.nppiMinGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(short)</param>
		public void MinA(CudaDeviceVariable<short> min)
		{
			int bufferSize = MinGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Min.nppiMin_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinGetBufferHostSizeA()"/></param>
		public void MinA(CudaDeviceVariable<short> min, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Min.nppiMin_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMin_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndxGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndxGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 4 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 4 * sizeof(int)</param>
		public void MinIndex(CudaDeviceVariable<short> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MinIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 4 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 4 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinIndexGetBufferHostSize()"/></param>
		public void MinIndex(CudaDeviceVariable<short> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for MinIndex. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int MinIndexGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndxGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndxGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		public void MinIndexA(CudaDeviceVariable<short> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MinIndexGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinIndexGetBufferHostSizeA()"/></param>
		public void MinIndexA(CudaDeviceVariable<short> min, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinIndexGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinIdx.nppiMinIndx_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, min.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinIndx_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.Max.nppiMaxGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 4 * sizeof(short)</param>
		public void Max(CudaDeviceVariable<short> max)
		{
			int bufferSize = MaxGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Max.nppiMax_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxGetBufferHostSize()"/></param>
		public void Max(CudaDeviceVariable<short> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Max.nppiMax_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for Max. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int MaxGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.Max.nppiMaxGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(short)</param>
		public void MaxA(CudaDeviceVariable<short> max)
		{
			int bufferSize = MaxGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.Max.nppiMax_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel maximum. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxGetBufferHostSizeA()"/></param>
		public void MaxA(CudaDeviceVariable<short> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.Max.nppiMax_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMax_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndxGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndxGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 4 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 4 * sizeof(int)</param>
		public void MaxIndex(CudaDeviceVariable<short> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MaxIndexGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 4 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 4 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxIndexGetBufferHostSize()"/></param>
		public void MaxIndex(CudaDeviceVariable<short> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxIndexGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for MaxIndex. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int MaxIndexGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndxGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndxGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel maximum. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		public void MaxIndexA(CudaDeviceVariable<short> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY)
		{
			int bufferSize = MaxIndexGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="indexX">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="indexY">Allocated device memory with size of at least 3 * sizeof(int)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MaxIndexGetBufferHostSizeA()"/></param>
		public void MaxIndexA(CudaDeviceVariable<short> max, CudaDeviceVariable<int> indexX, CudaDeviceVariable<int> indexY, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxIndexGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaxIdx.nppiMaxIndx_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, max.DevicePointer, indexX.DevicePointer, indexY.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxIndx_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMaxGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="max">Allocated device memory with size of at least 4 * sizeof(short)</param>
		public void MinMax(CudaDeviceVariable<short> min, CudaDeviceVariable<short> max)
		{
			int bufferSize = MinMaxGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum. No additional buffer is allocated.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="max">Allocated device memory with size of at least 4 * sizeof(short)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxGetBufferHostSize()"/></param>
		public void MinMax(CudaDeviceVariable<short> min, CudaDeviceVariable<short> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for MinMax. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int MinMaxGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMaxGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMaxGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Image pixel minimum and maximum. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(short)</param>
		public void MinMaxA(CudaDeviceVariable<short> min, CudaDeviceVariable<short> max)
		{
			int bufferSize = MinMaxGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Image pixel minimum and maximum. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="min">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="max">Allocated device memory with size of at least 3 * sizeof(short)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MinMaxGetBufferHostSizeA()"/></param>
		public void MinMaxA(CudaDeviceVariable<short> min, CudaDeviceVariable<short> max, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MinMaxGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MinMaxNew.nppiMinMax_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, min.DevicePointer, max.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinMax_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.MeanNew.nppiMeanGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 4 * sizeof(double)</param>
		public void Mean(CudaDeviceVariable<double> mean)
		{
			int bufferSize = MeanGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 4 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanGetBufferHostSize()"/></param>
		public void Mean(CudaDeviceVariable<double> mean, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for Mean. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int MeanGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MeanNew.nppiMeanGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMeanGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image mean with 64-bit double precision result. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void MeanA(CudaDeviceVariable<double> mean)
		{
			int bufferSize = MeanGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image mean with 64-bit double precision result. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="mean">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="MeanGetBufferHostSize()"/></param>
		public void MeanA(CudaDeviceVariable<double> mean, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MeanGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MeanNew.nppiMean_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, buffer.DevicePointer, mean.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMean_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.NormInf.nppiNormInfGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 4 * sizeof(double)</param>
		public void NormInf(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 4 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfGetBufferHostSize()"/></param>
		public void NormInf(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for Norm inf. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormInfGetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormInf.nppiNormInfGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormInfGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image infinity norm. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void NormInfA(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormInfGetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image infinity norm. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormInfGetBufferHostSize()"/></param>
		public void NormInfA(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormInfGetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormInf.nppiNorm_Inf_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_Inf_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.NormL1.nppiNormL1GetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 4 * sizeof(double)</param>
		public void NormL1(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 4 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1GetBufferHostSize()"/></param>
		public void NormL1(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Scratch-buffer size for Norm L1. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormL1GetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormL1.nppiNormL1GetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL1GetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L1 norm. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void NormL1A(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL1GetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L1 norm. No additional buffer is allocated. Not affecting Alpha.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL1GetBufferHostSize()"/></param>
		public void NormL1A(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL1GetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL1.nppiNorm_L1_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L1_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.NormL2.nppiNormL2GetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 4 * sizeof(double)</param>
		public void NormL2(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 4 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2GetBufferHostSize()"/></param>
		public void NormL2(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Scratch-buffer size for Norm L2. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormL2GetBufferHostSizeA()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormL2.nppiNormL2GetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormL2GetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image L2 norm. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		public void NormL2A(CudaDeviceVariable<double> norm)
		{
			int bufferSize = NormL2GetBufferHostSizeA();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image L2 norm. No additional buffer is allocated.
		/// </summary>
		/// <param name="norm">Allocated device memory with size of at least 3 * sizeof(double)</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormL2GetBufferHostSize()"/></param>
		public void NormL2A(CudaDeviceVariable<double> norm, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormL2GetBufferHostSizeA();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormL2.nppiNorm_L2_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, norm.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNorm_L2_16s_AC4R", status));
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
		public void Compare(NPPImage_16sC4 src2, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompare_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompare_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc's pixels with constant value.
		/// </summary>
		/// <param name="nConstant">constant value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		public void Compare(short[] nConstant, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompareC_16s_C4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareC_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc1's pixels with corresponding pixels in pSrc2. Not affecting Alpha.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		public void CompareA(NPPImage_16sC4 src2, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompare_16s_AC4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompare_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Compare pSrc's pixels with constant value. Not affecting Alpha.
		/// </summary>
		/// <param name="nConstant">constant value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
		public void CompareA(short[] nConstant, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Compare.nppiCompareC_16s_AC4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareC_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Threshold
		/// <summary>
		/// Image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		public void ThresholdA(NPPImage_16sC4 dest, short[] nThreshold, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		public void ThresholdA(short[] nThreshold, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdGT
		/// <summary>
		/// Image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdGTA(NPPImage_16sC4 dest, short[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdGTA(short[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdLT
		/// <summary>
		/// Image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdLTA(NPPImage_16sC4 dest, short[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nThreshold, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		public void ThresholdLTA(short[] nThreshold)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdVal
		/// <summary>
		/// Image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		public void ThresholdA(NPPImage_16sC4 dest, short[] nThreshold, short[] nValue, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		/// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
		public void ThresholdA(short[] nThreshold, short[] nValue, NppCmpOp eComparisonOperation)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdGTVal
		/// <summary>
		/// Image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdGTA(NPPImage_16sC4 dest, short[] nThreshold, short[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdGTA(short[] nThreshold, short[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdLTVal
		/// <summary>
		/// Image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdLTA(NPPImage_16sC4 dest, short[] nThreshold, short[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
		/// to nValue, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThreshold">The threshold value.</param>
		/// <param name="nValue">The threshold replacement value.</param>
		public void ThresholdLTA(short[] nThreshold, short[] nValue)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region ThresholdLTValGTVal
		/// <summary>
		/// Image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
		/// to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nThresholdLT">The thresholdLT value.</param>
		/// <param name="nValueLT">The thresholdLT replacement value.</param>
		/// <param name="nThresholdGT">The thresholdGT value.</param>
		/// <param name="nValueGT">The thresholdGT replacement value.</param>
		public void ThresholdLTGTA(NPPImage_16sC4 dest, short[] nThresholdLT, short[] nValueLT, short[] nThresholdGT, short[] nValueGT)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image threshold. Not affecting Alpha.<para/>
		/// If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
		/// to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
		/// </summary>
		/// <param name="nThresholdLT">The thresholdLT value.</param>
		/// <param name="nValueLT">The thresholdLT replacement value.</param>
		/// <param name="nThresholdGT">The thresholdGT value.</param>
		/// <param name="nValueGT">The thresholdGT replacement value.</param>
		public void ThresholdLTGTA(short[] nThresholdLT, short[] nValueLT, short[] nThresholdGT, short[] nValueGT)
		{
			status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		//new in Cuda 5.5
		#region DotProduct
		/// <summary>
		/// Device scratch buffer size (in bytes) for nppiDotProd_16s64f_C4R.
		/// </summary>
		/// <returns></returns>
		public int DotProdGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.DotProd.nppiDotProdGetBufferHostSize_16s64f_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProdGetBufferHostSize_16s64f_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Four-channel 16-bit signed image DotProd.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (4 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="DotProdGetBufferHostSize()"/></param>
		public void DotProduct(NPPImage_16sC4 src2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = DotProdGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_16s64f_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_16s64f_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Four-channel 16-bit signed image DotProd. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (4 * sizeof(double))</param>
		public void DotProduct(NPPImage_16sC4 src2, CudaDeviceVariable<double> pDp)
		{
			int bufferSize = DotProdGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_16s64f_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_16s64f_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for nppiDotProd_16s64f_C4R. Ignoring alpha channel.
		/// </summary>
		/// <returns></returns>
		public int ADotProdGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.DotProd.nppiDotProdGetBufferHostSize_16s64f_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProdGetBufferHostSize_16s64f_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Four-channel 16-bit signed image DotProd. Ignoring alpha channel.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="ADotProdGetBufferHostSize()"/></param>
		public void ADotProduct(NPPImage_16sC4 src2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = DotProdGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_16s64f_AC4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_16s64f_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Four-channel 16-bit signed image DotProd. Buffer is internally allocated and freed. Ignoring alpha channel.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pDp">Pointer to the computed dot product of the two images. (3 * sizeof(double))</param>
		public void ADotProduct(NPPImage_16sC4 src2, CudaDeviceVariable<double> pDp)
		{
			int bufferSize = DotProdGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_16s64f_AC4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_16s64f_AC4R", status));
			buffer.Dispose();
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
		/// <param name="values3">array of user defined OUTPUT values, channel 3</param>
		/// <param name="levels3">array of user defined INPUT values, channel 3</param>
		public void Lut(NPPImage_16sC4 dest, CudaDeviceVariable<int> values0, CudaDeviceVariable<int> levels0, CudaDeviceVariable<int> values1, CudaDeviceVariable<int> levels1,
			CudaDeviceVariable<int> values2, CudaDeviceVariable<int> levels2, CudaDeviceVariable<int> values3, CudaDeviceVariable<int> levels3)
		{
			CUdeviceptr[] values = new CUdeviceptr[4];
			CUdeviceptr[] levels = new CUdeviceptr[4];
			int[] levelLengths = new int[4];

			values[0] = values0.DevicePointer;
			values[1] = values1.DevicePointer;
			values[2] = values2.DevicePointer;
			values[3] = values3.DevicePointer;

			levels[0] = levels0.DevicePointer;
			levels[1] = levels1.DevicePointer;
			levels[2] = levels2.DevicePointer;
			levels[3] = levels3.DevicePointer;

			levelLengths[0] = levels0.Size;
			levelLengths[1] = levels1.Size;
			levelLengths[2] = levels2.Size;
			levelLengths[3] = levels3.Size;

			status = NPPNativeMethods.NPPi.ColorLUTLinear.nppiLUT_Linear_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, values, levels, levelLengths);

			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUT(NPPImage_16sC4 dst, CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer, pValues[3].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer, pLevels[3].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size, pLevels[3].Size };
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubic(NPPImage_16sC4 dst, CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer, pValues[3].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer, pLevels[3].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size, pLevels[3].Size };
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation.
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUT(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer, pValues[3].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer, pLevels[3].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size, pLevels[3].Size };
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_16s_C4IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_16s_C4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubic(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer, pValues[3].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer, pLevels[3].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size, pLevels[3].Size };
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_16s_C4IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_16s_C4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace linear interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation. 
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTLinear(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer, pValues[3].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer, pLevels[3].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size, pLevels[3].Size };
			status = NPPNativeMethods.NPPi.ColorLUTLinear.nppiLUT_Linear_16s_C4IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_16s_C4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Transpose
		/// <summary>
		/// image transpose
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void Transpose(NPPImage_16sC4 dest)
		{
			status = NPPNativeMethods.NPPi.Transpose.nppiTranspose_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Copy

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
		public void Copy(NPPImage_16sC4 dst, int nTopBorderHeight, int nLeftBorderWidth, short[] nValue)
		{
			status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_16s_C4R", status));
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
		public void CopyReplicateBorder(NPPImage_16sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_16s_C4R", status));
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
		public void CopyWrapBorder(NPPImage_16sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_16s_C4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// linearly interpolated source image subpixel coordinate color copy.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nDx">Fractional part of source image X coordinate.</param>
		/// <param name="nDy">Fractional part of source image Y coordinate.</param>
		public void CopySubpix(NPPImage_16sC4 dst, float nDx, float nDy)
		{
			status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMaxEveryNew
		/// <summary>
		/// image MinEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MinEvery(NPPImage_16sC4 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMinEvery_16s_C4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinEvery_16s_C4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image MaxEvery
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MaxEvery(NPPImage_16sC4 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMaxEvery_16s_C4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxEvery_16s_C4IR", status));
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
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_16s_C4IR(_devPtrRoi, _pitch, _sizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_16s_C4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Mirror image.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		public void Mirror(NPPImage_16sC4 dest, NppiAxis flip)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Filter
		/// <summary>
		/// 1D column convolution.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterColumn(NPPImage_16sC4 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumn32f_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn32f_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 1D row convolution.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterRow(NPPImage_16sC4 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRow32f_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow32f_16s_C4R", status));
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
		public void Filter(NPPImage_16sC4 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter32f_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Gauss filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterGauss(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGauss_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGauss_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterHighPass(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPass_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPass_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLowPass(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPass_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPass_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSharpen(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpen_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpen_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Pixels under the mask are multiplied by the respective weights in the mask and the results are summed.<para/>
		/// Before writing the result pixel the sum is scaled back via division by nDivisor.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="aKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		public void Filter(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, NppiSize aKernelSize, NppiPoint oAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, aKernelSize, oAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter_16s_C4R", status));
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
		public void FilterColumn(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumn_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn_16s_C4R", status));
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
		public void FilterRow(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRow_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow_16s_C4R", status));
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
		public void FilterRowBorder(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRowBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterBox(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBox_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBox_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMin(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMin_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMin_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMax(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMax_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMax_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittHoriz(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHoriz_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHoriz_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittVert(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVert_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVert_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void SobelHoriz(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelHoriz_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHoriz_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Sobel filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSobelVert(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelVert_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVert_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsDown(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDown_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDown_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter..
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsUp(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUp_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUp_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLaplace(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplace_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplace_16s_C4R", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfGetBufferHostSize()"/></param>
		public void NormDiff_Inf(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_Inf(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_16s_C4R", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1GetBufferHostSize()"/></param>
		public void NormDiff_L1(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_L1(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_16s_C4R", status));
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
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2GetBufferHostSize()"/></param>
		public void NormDiff_L2(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_L2(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_16s_C4R", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfGetBufferHostSize()"/></param>
		public void NormRel_Inf(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelInfGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_Inf(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelInfGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_16s_C4R", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1GetBufferHostSize()"/></param>
		public void NormRel_L1(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL1GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_L1(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL1GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_16s_C4R", status));
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
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2GetBufferHostSize()"/></param>
		public void NormRel_L2(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL2GetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_L2(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL2GetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_16s_C4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_16s_C4R", status));
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
		public void ResizeSqrPixel(NPPImage_16sC4 dst, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect dstRect = new NppiRect(dst.PointRoi, dst.SizeRoi);
			status = NPPNativeMethods.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_16s_C4R(_devPtr, _sizeRoi, _pitch, srcRect, dst.DevicePointer, dst.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image remap.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. </param>
		/// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. </param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public void Remap(NPPImage_16sC4 dst, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Remap.nppiRemap_16s_C4R(_devPtr, _sizeRoi, _pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// image conversion.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
		public void Scale(NPPImage_8uC4 dst, NppHintAlgorithm hint)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Scale.nppiScale_16s8u_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, hint);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_16s8u_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region SwapChannelNew


		/// <summary>
		/// Swap channels.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. The n-th entry of the array
		/// contains the number of the channel that is stored in the n-th channel of the output image. E.g.
		/// Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR channel order.</param>
		public void SwapChannels(NPPImage_16sC4 dest, int[] aDstOrder)
		{
			status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_16s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Swap channels.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry
		/// of the array contains the number of the channel that is stored in the n-th channel of
		/// the output image. <para/>E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
		/// channel order.</param>
		public void SwapChannels(NPPImage_16sC3 dest, int[] aDstOrder)
		{
			status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_16s_C4C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_16s_C4C3R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Swap channels, in-place.
		/// </summary>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. The n-th entry of the array
		/// contains the number of the channel that is stored in the n-th channel of the output image. E.g.
		/// Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR channel order.</param>
		public void SwapChannels(int[] aDstOrder)
		{
			status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_16s_C4IR(_devPtrRoi, _pitch, _sizeRoi, aDstOrder);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_16s_C4IR", status));
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
		public static void ColorTwist(NPPImage_16sC1 src0, NPPImage_16sC1 src1, NPPImage_16sC1 src2, NPPImage_16sC1 dest0, NPPImage_16sC1 dest1, NPPImage_16sC1 dest2, float[,] twistMatrix)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi };

			NppStatus status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_16s_P3R(src, src0.Pitch, dst, dest0.Pitch, src0.SizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16s_P3R", status));
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
		public static void ColorTwist(NPPImage_16sC1 srcDest0, NPPImage_16sC1 srcDest1, NPPImage_16sC1 srcDest2, float[,] twistMatrix)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { srcDest0.DevicePointerRoi, srcDest1.DevicePointerRoi, srcDest2.DevicePointerRoi };

			NppStatus status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_16s_IP3R(src, srcDest0.Pitch, srcDest0.SizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16s_IP3R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion



		//Alpha
		#region Color...New
		/// <summary>
		/// Swap channels. Not affecting Alpha
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aDstOrder">Integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of
		/// the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
		/// channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
		/// </param>
		public void SwapChannelsA(NPPImage_16sC4 dest, int[] aDstOrder)
		{
			status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// An input color twist matrix with floating-point pixel values is applied
		/// within ROI. Alpha channel is the last channel and is not processed.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
		public void ColorTwistA(NPPImage_16sC4 dest, float[,] twistMatrix)
		{
			status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_16s_AC4R(_devPtr, _pitch, dest.DevicePointer, dest.Pitch, _sizeRoi, twistMatrix);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// RGB to Gray conversion, not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		public void RGBToGrayA(NPPImage_16sC1 dest)
		{
			status = NPPNativeMethods.NPPi.RGBToGray.nppiRGBToGray_16s_AC4C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRGBToGray_16s_AC4C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Color to Gray conversion, not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
		public void ColorToGrayA(NPPImage_16sC1 dest, float[] aCoeffs)
		{
			status = NPPNativeMethods.NPPi.ColorToGray.nppiColorToGray_16s_AC4C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aCoeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorToGray_16s_AC4C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Color to Gray conversion.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
		public void ColorToGray(NPPImage_16sC1 dest, float[] aCoeffs)
		{
			status = NPPNativeMethods.NPPi.ColorToGray.nppiColorToGray_16s_C4C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aCoeffs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorToGray_16s_C4C1R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// in place color twist, not affecting Alpha.
		/// 
		/// An input color twist matrix with floating-point coefficient values is applied
		/// within ROI.
		/// </summary>
		/// <param name="aTwist">The color twist matrix with floating-point coefficient values. [3,4]</param>
		public void ColorTwistA(float[,] aTwist)
		{
			status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_16s_AC4IR(_devPtr, _pitch, _sizeRoi, aTwist);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Filter
		/// <summary>
		/// Pixels under the mask are multiplied by the respective weights in the mask and the results are summed.<para/>
		/// Before writing the result pixel the sum is scaled back via division by nDivisor. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="aKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		public void FilterA(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, NppiSize aKernelSize, NppiPoint oAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, aKernelSize, oAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Apply convolution filter with user specified 1D column of weights. Result pixel is equal to the sum of
		/// the products between the kernel coefficients (pKernel array) and corresponding neighboring column pixel
		/// values in the source image defined by nKernelDim and nAnchorY, divided by nDivisor. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nKernelSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		public void FilterColumnA(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumn_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Apply general linear Row convolution filter, with rescaling, in a 1D mask region around each source pixel. 
		/// Result pixel is equal to the sum of the products between the kernel
		/// coefficients (pKernel array) and corresponding neighboring row pixel values in the source image defined
		/// by iKernelDim and iAnchorX, divided by iDivisor. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nKernelSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		public void FilterRowA(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRow_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Apply general linear Row convolution filter, with rescaling, in a 1D mask region around each source pixel with border control. 
		/// Result pixel is equal to the sum of the products between the kernel
		/// coefficients (pKernel array) and corresponding neighboring row pixel values in the source image defined
		/// by iKernelDim and iAnchorX, divided by iDivisor. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="Kernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
		/// <param name="nKernelSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRowBorderA(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nKernelSize, int nAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRowBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, Kernel.DevicePointer, nKernelSize, nAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterBoxA(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBox_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBox_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMinA(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMin_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMin_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void FilterMaxA(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMax_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMax_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// 1D column convolution. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterColumnA(NPPImage_16sC4 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumn32f_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumn32f_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 1D row convolution. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array. pKernel.Sizes gives kernel size<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterRowA(NPPImage_16sC4 dst, CudaDeviceVariable<float> pKernel, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRow32f_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, pKernel.Size, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRow32f_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// convolution filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pKernel">Pointer to the start address of the kernel coefficient array.<para/>
		/// Coefficients are expected to be stored in reverse order.</param>
		/// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference</param>
		public void FilterA(NPPImage_16sC4 dst, CudaDeviceVariable<float> pKernel, NppiSize oKernelSize, NppiPoint oAnchor)
		{
			status = NPPNativeMethods.NPPi.Convolution.nppiFilter32f_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, pKernel.DevicePointer, oKernelSize, oAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilter32f_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Gauss filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterGaussA(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGauss_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGauss_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// High pass filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterHighPassA(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPass_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPass_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLowPassA(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPass_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPass_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSharpenA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpen_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpen_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Prewitt filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittHorizA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHoriz_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHoriz_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterPrewittVertA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVert_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVert_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Sobel filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void SobelHorizA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelHoriz_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHoriz_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Sobel filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterSobelVertA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSobelVert_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVert_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// horizontal Roberts filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsDownA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDown_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDown_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// vertical Roberts filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		public void FilterRobertsUpA(NPPImage_16sC4 dst)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUp_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUp_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Laplace filter. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		public void FilterLaplaceA(NPPImage_16sC4 dst, MaskSize eMaskSize)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplace_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplace_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region GeometryNew

		/// <summary>
		/// image resize. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nXFactor">Factor by which x dimension is changed. </param>
		/// <param name="nYFactor">Factor by which y dimension is changed. </param>
		/// <param name="nXShift">Source pixel shift in x-direction.</param>
		/// <param name="nYShift">Source pixel shift in y-direction.</param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public void ResizeSqrPixelA(NPPImage_16sC4 dst, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			NppiRect dstRect = new NppiRect(dst.PointRoi, dst.SizeRoi);
			status = NPPNativeMethods.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_16s_AC4R(_devPtr, _sizeRoi, _pitch, srcRect, dst.DevicePointer, dst.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image remap. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. </param>
		/// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. </param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public void RemapA(NPPImage_16sC4 dst, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Remap.nppiRemap_16s_AC4R(_devPtr, _sizeRoi, _pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}




		/// <summary>
		/// image conversion. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
		public void ScaleA(NPPImage_8uC4 dst, NppHintAlgorithm hint)
		{
			NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
			status = NPPNativeMethods.NPPi.Scale.nppiScale_16s8u_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, hint);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_16s8u_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region LUT

		/// <summary>
		/// look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTA(NPPImage_16sC4 dst, CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation.  Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubicA(NPPImage_16sC4 dst, CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Inplace look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points with no interpolation. Not affecting Alpha.
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTA(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUT.nppiLUT_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace cubic interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation.  Not affecting Alpha.
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTCubicA(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUTCubic.nppiLUT_Cubic_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Cubic_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Inplace linear interpolated look-up-table color conversion.
		/// The LUT is derived from a set of user defined mapping points through cubic interpolation.  Not affecting Alpha.
		/// </summary>
		/// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
		/// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values. pLevels.Size gives nLevels.</param>
		public void LUTLinearA(CudaDeviceVariable<int>[] pValues, CudaDeviceVariable<int>[] pLevels)
		{
			CUdeviceptr[] ptrsV = new CUdeviceptr[] { pValues[0].DevicePointer, pValues[1].DevicePointer, pValues[2].DevicePointer };
			CUdeviceptr[] ptrsL = new CUdeviceptr[] { pLevels[0].DevicePointer, pLevels[1].DevicePointer, pLevels[2].DevicePointer };
			int[] size = new int[] { pLevels[0].Size, pLevels[1].Size, pLevels[2].Size };
			status = NPPNativeMethods.NPPi.ColorLUTLinear.nppiLUT_Linear_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, ptrsV, ptrsL, size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// look-up-table color conversion.<para/>
		/// The LUT is derived from a set of user defined mapping points through linear interpolation. Not affecting alpha channel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="values0">array of user defined OUTPUT values, channel 0</param>
		/// <param name="levels0">array of user defined INPUT values, channel 0</param>
		/// <param name="values1">array of user defined OUTPUT values, channel 1</param>
		/// <param name="levels1">array of user defined INPUT values, channel 1</param>
		/// <param name="values2">array of user defined OUTPUT values, channel 2</param>
		/// <param name="levels2">array of user defined INPUT values, channel 2</param>
		public void LutA(NPPImage_16sC4 dest, CudaDeviceVariable<int> values0, CudaDeviceVariable<int> levels0, CudaDeviceVariable<int> values1,
			CudaDeviceVariable<int> levels1, CudaDeviceVariable<int> values2, CudaDeviceVariable<int> levels2)
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

			levelLengths[0] = levels0.Size;
			levelLengths[1] = levels1.Size;
			levelLengths[2] = levels2.Size;

			status = NPPNativeMethods.NPPi.ColorLUTLinear.nppiLUT_Linear_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, values, levels, levelLengths);

			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLUT_Linear_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MinMaxEveryNew
		/// <summary>
		/// image MinEvery Not affecting Alpha.
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MinEveryA(NPPImage_16sC4 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMinEvery_16s_AC4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMinEvery_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image MaxEvery Not affecting Alpha.
		/// </summary>
		/// <param name="src2">Source-Image</param>
		public void MaxEveryA(NPPImage_16sC4 src2)
		{
			status = NPPNativeMethods.NPPi.MinMaxEvery.nppiMaxEvery_16s_AC4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaxEvery_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MirrorNew


		/// <summary>
		/// Mirror image inplace. Not affecting Alpha.
		/// </summary>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		public void MirrorA(NppiAxis flip)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_16s_AC4IR(_devPtrRoi, _pitch, _sizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_16s_AC4IR", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Mirror image. Not affecting Alpha.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="flip">Specifies the axis about which the image is to be mirrored.</param>
		public void MirrorA(NPPImage_16sC4 dest, NppiAxis flip)
		{
			status = NPPNativeMethods.NPPi.GeometricTransforms.nppiMirror_16s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, flip);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMirror_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Copy
		/// <summary>
		/// Copy image and pad borders with a constant, user-specifiable color. Not affecting Alpha channel.
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
		public void CopyConstBorderA(NPPImage_16sC4 dst, int nTopBorderHeight, int nLeftBorderWidth, short[] nValue)
		{
			status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image copy with nearest source image pixel color. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nTopBorderHeight">Height (in pixels) of the top border. The height of the border at the bottom of
		/// the destination ROI is implicitly defined by the size of the source ROI: nBottomBorderHeight =
		/// oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
		/// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		public void CopyReplicateBorderA(NPPImage_16sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// image copy with the borders wrapped by replication of source image pixel colors. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nTopBorderHeight">Height (in pixels) of the top border. The height of the border at the bottom of
		/// the destination ROI is implicitly defined by the size of the source ROI: nBottomBorderHeight =
		/// oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
		/// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of
		/// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
		/// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
		public void CopyWrapBorderA(NPPImage_16sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
		{
			status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_16s_AC4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// linearly interpolated source image subpixel coordinate color copy. Not affecting Alpha.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="nDx">Fractional part of source image X coordinate.</param>
		/// <param name="nDy">Fractional part of source image Y coordinate.</param>
		public void CopySubpixA(NPPImage_16sC4 dst, float nDx, float nDy)
		{
			status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// planar image resize.
		/// </summary>
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="src3">Source image (Channel 3)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="dest3">Destination image (Channel 3)</param>
		/// <param name="nXFactor">Factor by which x dimension is changed. </param>
		/// <param name="nYFactor">Factor by which y dimension is changed. </param>
		/// <param name="nXShift">Source pixel shift in x-direction.</param>
		/// <param name="nYShift">Source pixel shift in y-direction.</param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public static void ResizeSqrPixel(NPPImage_16sC1 src0, NPPImage_16sC1 src1, NPPImage_16sC1 src2, NPPImage_16sC1 src3, NPPImage_16sC1 dest0, NPPImage_16sC1 dest1, NPPImage_16sC1 dest2, NPPImage_16sC1 dest3, double nXFactor, double nYFactor, double nXShift, double nYShift, InterpolationMode eInterpolation)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer, src3.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer, dest3.DevicePointer };
			NppiRect srcRect = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppiRect dstRect = new NppiRect(dest0.PointRoi, dest0.SizeRoi);
			NppStatus status = NPPNativeMethods.NPPi.ResizeSqrPixel.nppiResizeSqrPixel_16s_P4R(src, src0.SizeRoi, src0.Pitch, srcRect, dst, dest0.Pitch, dstRect, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResizeSqrPixel_16s_P4R", status));
			NPPException.CheckNppStatus(status, null);
		}

		/// <summary>
		/// planar image remap.
		/// </summary>
		/// <param name="src0">Source image (Channel 0)</param>
		/// <param name="src1">Source image (Channel 1)</param>
		/// <param name="src2">Source image (Channel 2)</param>
		/// <param name="src3">Source image (Channel 3)</param>
		/// <param name="dest0">Destination image (Channel 0)</param>
		/// <param name="dest1">Destination image (Channel 1)</param>
		/// <param name="dest2">Destination image (Channel 2)</param>
		/// <param name="dest3">Destination image (Channel 3)</param>
		/// <param name="pXMap">Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. </param>
		/// <param name="pYMap">Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. </param>
		/// <param name="eInterpolation">The type of eInterpolation to perform resampling.</param>
		public static void Remap(NPPImage_16sC1 src0, NPPImage_16sC1 src1, NPPImage_16sC1 src2, NPPImage_16sC1 src3, NPPImage_16sC1 dest0, NPPImage_16sC1 dest1, NPPImage_16sC1 dest2, NPPImage_16sC1 dest3, NPPImage_32fC1 pXMap, NPPImage_32fC1 pYMap, InterpolationMode eInterpolation)
		{
			CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer, src3.DevicePointer };
			CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointerRoi, dest1.DevicePointerRoi, dest2.DevicePointerRoi, dest3.DevicePointerRoi };
			NppiRect srcRect = new NppiRect(src0.PointRoi, src0.SizeRoi);
			NppStatus status = NPPNativeMethods.NPPi.Remap.nppiRemap_16s_P4R(src, src0.SizeRoi, src0.Pitch, srcRect, pXMap.DevicePointerRoi, pXMap.Pitch, pYMap.DevicePointerRoi, pYMap.Pitch, dst, dest0.Pitch, dest0.SizeRoi, eInterpolation);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRemap_16s_P4R", status));
			NPPException.CheckNppStatus(status, null);
		}
		#endregion

		#region NormNew
		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_Inf. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormDiffInfAGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffInfGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffInfGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_Inf. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffInfAGetBufferHostSize()"/></param>
		public void NormDiff_InfA(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffInfAGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_Inf. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed Inf-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_InfA(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffInfAGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_Inf_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_Inf_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L1. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL1AGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL1GetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL1GetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L1. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL1AGetBufferHostSize()"/></param>
		public void NormDiff_L1A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL1AGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L1. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L1-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_L1A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL1AGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L1_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L1_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormDiff_L2. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormDiffL2AGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiffL2GetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiffL2GetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormDiff_L2. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormDiffL2AGetBufferHostSize()"/></param>
		public void NormDiff_L2A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormDiffL2AGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormDiff_L2. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormDiff">Pointer to the computed L2-norm of differences. (3 * sizeof(double))</param>
		public void NormDiff_L2A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormDiff)
		{
			int bufferSize = NormDiffL2AGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormDiff.nppiNormDiff_L2_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormDiff.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormDiff_L2_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}



		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_Inf. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormRelInfAGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelInfGetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelInfGetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_Inf. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelInfAGetBufferHostSize()"/></param>
		public void NormRel_InfA(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelInfAGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_Inf. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_InfA(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelInfAGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_Inf_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_Inf_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L1. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormRelL1AGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL1GetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL1GetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L1. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL1AGetBufferHostSize()"/></param>
		public void NormRel_L1A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL1AGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L1. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_L1A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL1AGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L1_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L1_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Device scratch buffer size (in bytes) for NormRel_L2. Not affecting Alpha.
		/// </summary>
		/// <returns></returns>
		public int NormRelL2AGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.NormRel.nppiNormRelL2GetBufferHostSize_16s_AC4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRelL2GetBufferHostSize_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// image NormRel_L2. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		/// <param name="buffer">Allocated device memory with size of at <see cref="NormRelL2AGetBufferHostSize()"/></param>
		public void NormRel_L2A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = NormRelL2AGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image NormRel_L2. Buffer is internally allocated and freed. Not affecting Alpha.
		/// </summary>
		/// <param name="tpl">template image.</param>
		/// <param name="pNormRel">Pointer to the computed relative error for the infinity norm of two images. (3 * sizeof(double))</param>
		public void NormRel_L2A(NPPImage_16sC4 tpl, CudaDeviceVariable<double> pNormRel)
		{
			int bufferSize = NormRelL2AGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

			status = NPPNativeMethods.NPPi.NormRel.nppiNormRel_L2_16s_AC4R(_devPtrRoi, _pitch, tpl.DevicePointerRoi, tpl.Pitch, _sizeRoi, pNormRel.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiNormRel_L2_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		//New in Cuda 6.0

		#region SumWindow
		/// <summary>
		/// 16-bit signed 1D (column) sum to 32f.
		/// Apply Column Window Summation filter over a 1D mask region around each
		/// source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point
		/// output.  <para/>
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring column pixel values in a mask region of the source image defined by
		/// nMaskSize and nAnchor. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">Y offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void SumWindowColumn(NPPImage_32fC4 dest, int nMaskSize, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowColumn_16s32f_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowColumn_16s32f_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// 16-bit signed 1D (row) sum to 32f.<para/>
		/// Apply Row Window Summation filter over a 1D mask region around each source
		/// pixel for 4-channel 16-bit pixel input images with 32-bit floating point output.  
		/// Result 32-bit floating point pixel is equal to the sum of the corresponding and
		/// neighboring row pixel values in a mask region of the source image defined
		/// by nKernelDim and nAnchorX. 
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="nMaskSize">Length of the linear kernel array.</param>
		/// <param name="nAnchor">X offset of the kernel origin frame of reference w.r.t the source pixel.</param>
		public void SumWindowRow(NPPImage_32fC4 dest, int nMaskSize, int nAnchor)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowRow_16s32f_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowRow_16s32f_C4R", status));
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
		public void FilterMedian(NPPImage_16sC4 dst, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_16s_C4R", status));
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
		public void FilterMedian(NPPImage_16sC4 dst, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = FilterMedianGetBufferHostSize(oMaskSize);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for FilterMedian.
		/// </summary>
		/// <returns></returns>
		public int FilterMedianGetBufferHostSize(NppiSize oMaskSize)
		{
			uint bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianGetBufferSize_16s_C4R(_sizeRoi, oMaskSize, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianGetBufferSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return (int)bufferSize; //We stay consistent with other GetBufferHostSize functions and convert to int.
		}
		/// <summary>
		/// Result pixel value is the median of pixel values under the rectangular mask region, ignoring alpha channel.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		public void FilterMedianA(NPPImage_16sC4 dst, NppiSize oMaskSize, NppiPoint oAnchor)
		{
			int bufferSize = FilterMedianGetBufferHostSizeA(oMaskSize);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_16s_AC4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the median of pixel values under the rectangular mask region, ignoring alpha channel.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
		public void FilterMedianA(NPPImage_16sC4 dst, NppiSize oMaskSize, NppiPoint oAnchor, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = FilterMedianGetBufferHostSizeA(oMaskSize);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedian_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, oMaskSize, oAnchor, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedian_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for FilterMedian, ignoring alpha channel.
		/// </summary>
		/// <returns></returns>
		public int FilterMedianGetBufferHostSizeA(NppiSize oMaskSize)
		{
			uint bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianGetBufferSize_16s_AC4R(_sizeRoi, oMaskSize, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianGetBufferSize_16s_AC4R", status));
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
		public void MaxError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
		public void MaxError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaxErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_16s_C4R", status));
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
		public void AverageError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
		public void AverageError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_16s_C4R", status));
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
		public void MaximumRelativeError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
		public void MaximumRelativeError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_16s_C4R", status));
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
		public void AverageRelativeError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_16s_C4R", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
		public void AverageRelativeError(NPPImage_16sC4 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize();
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_16s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize()
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_16s_C4R(_sizeRoi, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region FilterBorder
		/// <summary>
		/// Four channel 16-bit signed convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_16sC4 dest, CudaDeviceVariable<int> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder.nppiFilterBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Four channel 16-bit signed convolution filter with border control.<para/>
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
		public void FilterBorder(NPPImage_16sC4 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder32f.nppiFilterBorder32f_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Four channel 16-bit signed convolution filter with border control, ignoring alpha channel.<para/>
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
		public void FilterBorderA(NPPImage_16sC4 dest, CudaDeviceVariable<int> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder.nppiFilterBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Four channel 16-bit signed convolution filter with border control, ignoring alpha channel.<para/>
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
		public void FilterBorderA(NPPImage_16sC4 dest, CudaDeviceVariable<float> pKernel, NppiSize nKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterBorder32f.nppiFilterBorder32f_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, pKernel.DevicePointer, nKernelSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBorder32f_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterSobelBorder
		/// <summary>
		/// Filters the image using a horizontal Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelHorizBorder(NPPImage_16sC4 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelVertBorder(NPPImage_16sC4 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelVertBorder.nppiFilterSobelVertBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a horizontal Sobel filter kernel with border control, ignoring alpha channel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelHorizBorderA(NPPImage_16sC4 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelHorizBorder.nppiFilterSobelHorizBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelHorizBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Filters the image using a vertical Sobel filter kernel with border control, ignoring alpha channel.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSobelVertBorderA(NPPImage_16sC4 dest, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterSobelVertBorder.nppiFilterSobelVertBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSobelVertBorder_16s_AC4R", status));
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
		public void FilterGaussBorder(NPPImage_16sC4 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>Filters the image using a Gaussian filter kernel with border control, ignoring alpha channel:<para/>
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
		public void FilterGaussBorderA(NPPImage_16sC4 dest, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussBorder_16s_AC4R", status));
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
		public void FilterColumnBorder(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumnBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder_16s_C4R", status));
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
		public void FilterColumnBorder(NPPImage_16sC4 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumnBorder32f_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder32f_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
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
		public void FilterColumnBorderA(NPPImage_16sC4 dest, CudaDeviceVariable<int> Kernel, int nAnchor, int nDivisor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumnBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, nDivisor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder_16s_AC4R", status));
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
		public void FilterColumnBorderA(NPPImage_16sC4 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterColumnBorder32f_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterColumnBorder32f_16s_AC4R", status));
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
		public void FilterRowBorder(NPPImage_16sC4 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRowBorder32f_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder32f_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

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
		public void FilterRowBorderA(NPPImage_16sC4 dest, CudaDeviceVariable<float> Kernel, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFilter1D.nppiFilterRowBorder32f_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi, Kernel.DevicePointer, Kernel.Size, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRowBorder32f_16s_AC4R", status));
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
		public void SumWindowColumnBorder(NPPImage_32fC4 dest, int nMaskSize, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowColumnBorder_16s32f_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowColumnBorder_16s32f_C4R", status));
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
		public void SumWindowRowBorder(NPPImage_32fC4 dest, int nMaskSize, int nAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.WindowSum1D.nppiSumWindowRowBorder_16s32f_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nMaskSize, nAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSumWindowRowBorder_16s32f_C4R", status));
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
		public void FilterBoxBorder(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Computes the average pixel values of the pixels under a rectangular mask.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterBoxBorderA(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorder_16s_AC4R", status));
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
		public void FilterMinBorder(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMinBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMinBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterMaxBorder(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMaxBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMaxBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// Result pixel value is the minimum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterMinBorderA(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMinBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMinBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Result pixel value is the maximum of pixel values under the rectangular mask region.
		/// </summary>
		/// <param name="dest">Destination image</param>
		/// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
		/// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterMaxBorderA(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.RankFilters.nppiFilterMaxBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMaxBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region FilterOthers


		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittHorizBorder(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHorizBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHorizBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittHorizBorderA(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittHorizBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittHorizBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittVertBorder(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVertBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVertBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// vertical Prewitt filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterPrewittVertBorderA(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterPrewittVertBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterPrewittVertBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsDownBorder(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDownBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDownBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// horizontal Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsDownBorderA(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsDownBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsDownBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsUpBorder(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUpBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUpBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// vertical Roberts filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterRobertsUpBorderA(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterRobertsUpBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterRobertsUpBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLaplaceBorder(NPPImage_16sC4 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplaceBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplaceBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Laplace filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLaplaceBorderA(NPPImage_16sC4 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLaplaceBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLaplaceBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}


		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterHighPassBorder(NPPImage_16sC4 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPassBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPassBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// High pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterHighPassBorderA(NPPImage_16sC4 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterHighPassBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterHighPassBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLowPassBorder(NPPImage_16sC4 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPassBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPassBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Low pass filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterLowPassBorderA(NPPImage_16sC4 dst, MaskSize eMaskSize, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterLowPassBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eMaskSize, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterLowPassBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSharpenBorder(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpenBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpenBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Sharpen filter.
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterSharpenBorderA(NPPImage_16sC4 dst, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterSharpenBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterSharpenBorder_16s_AC4R", status));
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
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpGetBufferSize_16s_C4R(nRadius, nSigma, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpGetBufferSize_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}

		/// <summary>
		/// Scratch-buffer size for unsharp filter.
		/// </summary>
		/// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
		/// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
		/// <returns></returns>
		public int FilterUnsharpGetBufferSizeA(float nRadius, float nSigma)
		{
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpGetBufferSize_16s_AC4R(nRadius, nSigma, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpGetBufferSize_16s_AC4R", status));
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
		public void FilterUnsharpBorder(NPPImage_16sC4 dst, float nRadius, float nSigma, float nWeight, float nThreshold, NppiBorderType eBorderType, CudaDeviceVariable<byte> buffer)
		{
			if (buffer.Size < FilterUnsharpGetBufferSize(nRadius, nSigma))
				throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpBorder_16s_C4R(_devPtr, _pitch, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nRadius, nSigma, nWeight, nThreshold, eBorderType, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
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
		public void FilterUnsharpBorderA(NPPImage_16sC4 dst, float nRadius, float nSigma, float nWeight, float nThreshold, NppiBorderType eBorderType, CudaDeviceVariable<byte> buffer)
		{
			if (buffer.Size < FilterUnsharpGetBufferSizeA(nRadius, nSigma))
				throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpBorder_16s_AC4R(_devPtr, _pitch, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nRadius, nSigma, nWeight, nThreshold, eBorderType, buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		#endregion

		#region Filter Gauss Advanced

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		public void FilterGauss(NPPImage_16sC4 dst, CudaDeviceVariable<float> Kernel)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGaussAdvanced_16s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvanced_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterGaussBorder(NPPImage_16sC4 dst, CudaDeviceVariable<float> Kernel, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussAdvancedBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvancedBorder_16s_C4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		public void FilterGaussA(NPPImage_16sC4 dst, CudaDeviceVariable<float> Kernel)
		{
			status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterGaussAdvanced_16s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvanced_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Filters the image using a separable Gaussian filter kernel with user supplied floating point coefficients
		/// </summary>
		/// <param name="dst">Destination-Image</param>
		/// <param name="Kernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F, where nFilterTaps =  2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
		/// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
		public void FilterGaussBorderA(NPPImage_16sC4 dst, CudaDeviceVariable<float> Kernel, NppiBorderType eBorderType)
		{
			status = NPPNativeMethods.NPPi.FilterGaussBorder.nppiFilterGaussAdvancedBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, Kernel.Size, Kernel.DevicePointer, eBorderType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterGaussAdvancedBorder_16s_AC4R", status));
			NPPException.CheckNppStatus(status, this);
		}

        #endregion


        //New in Cuda 9.0
        #region New Cuda9
        /// <summary>
        /// Wiener filter with border control.
        /// </summary>
        /// <param name="dest">destination_image_pointer</param>
        /// <param name="oMaskSize">Pixel Width and Height of the rectangular region of interest surrounding the source pixel.</param>
        /// <param name="oAnchor">Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.</param>
        /// <param name="aNoise">Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void FilterWienerBorder(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, float[] aNoise, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.FilterWienerBorder.nppiFilterWienerBorder_16s_C4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, aNoise, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterWienerBorder_16s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Wiener filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="dest">destination_image_pointer</param>
        /// <param name="oMaskSize">Pixel Width and Height of the rectangular region of interest surrounding the source pixel.</param>
        /// <param name="oAnchor">Positive X and Y relative offsets of primary pixel in region of interest surrounding the source pixel relative to bottom right of oMaskSize.</param>
        /// <param name="aNoise">Fixed size array of per-channel noise variance level value in range of 0.0F to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public void FilterWienerBorderA(NPPImage_16sC4 dest, NppiSize oMaskSize, NppiPoint oAnchor, float[] aNoise, NppiBorderType eBorderType)
        {
            status = NPPNativeMethods.NPPi.FilterWienerBorder.nppiFilterWienerBorder_16s_AC4R(_devPtr, _pitch, _sizeOriginal, _pointRoi, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, oMaskSize, oAnchor, aNoise, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterWienerBorder_16s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Resizes images.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="eInterpolation">Interpolation mode</param>
        public void Resize(NPPImage_16sC4 dest, InterpolationMode eInterpolation)
        {
            status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_16s_C4R(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointer, dest.Pitch, dest.Size, new NppiRect(dest.PointRoi, dest.SizeRoi), eInterpolation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_16s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Resizes images. Not affecting Alpha.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="eInterpolation">Interpolation mode</param>
        public void ResizeA(NPPImage_16sC4 dest, InterpolationMode eInterpolation)
        {
            status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_16s_AC4R(_devPtr, _pitch, _sizeOriginal, new NppiRect(_pointRoi, _sizeRoi), dest.DevicePointer, dest.Pitch, dest.Size, new NppiRect(dest.PointRoi, dest.SizeRoi), eInterpolation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_16s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// resizes planar images.
        /// </summary>
        /// <param name="src0">Source image (Channel 0)</param>
        /// <param name="src1">Source image (Channel 1)</param>
        /// <param name="src2">Source image (Channel 2)</param>
        /// <param name="src3">Source image (Channel 3)</param>
        /// <param name="dest0">Destination image (Channel 0)</param>
        /// <param name="dest1">Destination image (Channel 1)</param>
        /// <param name="dest2">Destination image (Channel 2)</param>
        /// <param name="dest3">Destination image (Channel 3)</param>
        /// <param name="eInterpolation">Interpolation mode</param>
        public static void Resize(NPPImage_16sC1 src0, NPPImage_16sC1 src1, NPPImage_16sC1 src2, NPPImage_16sC1 src3, NPPImage_16sC1 dest0, NPPImage_16sC1 dest1, NPPImage_16sC1 dest2, NPPImage_16sC1 dest3, InterpolationMode eInterpolation)
        {
            CUdeviceptr[] src = new CUdeviceptr[] { src0.DevicePointer, src1.DevicePointer, src2.DevicePointer, src3.DevicePointer };
            CUdeviceptr[] dst = new CUdeviceptr[] { dest0.DevicePointer, dest1.DevicePointer, dest2.DevicePointer, dest3.DevicePointer };
            NppStatus status = NPPNativeMethods.NPPi.GeometricTransforms.nppiResize_16s_P4R(src, src0.Pitch, src0.Size, new NppiRect(src0.PointRoi, src0.SizeRoi), dst, dest0.Pitch, dest0.Size, new NppiRect(dest0.PointRoi, dest0.SizeRoi), eInterpolation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiResize_16s_P4R", status));
            NPPException.CheckNppStatus(status, null);
        }
        #endregion
    }
}
