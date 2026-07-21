// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
// http://kunzmi.github.io/managedCuda
//
// This file is part of ManagedCuda.
//
// Commercial License Usage
//  Licensees holding valid commercial ManagedCuda licenses may use this
//  file in accordance with the commercial license agreement provided with
//  the Software or, alternatively, in accordance with the terms contained
//  in a written agreement between you and Artic Imaging SARL. For further
//  information contact us at managedcuda@articimaging.eu.
//  
// GNU General Public License Usage
//  Alternatively, this file may be used under the terms of the GNU General
//  Public License as published by the Free Software Foundation, either 
//  version 3 of the License, or (at your option) any later version.
//  
//  ManagedCuda is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <http://www.gnu.org/licenses/>.


using ManagedCuda.BasicTypes;
using System;
using System.Diagnostics;

namespace ManagedCuda.NPP
{
    /// <summary>
    /// 
    /// </summary>
    public partial class NPPImage_32sC4 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_32sC4(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 4;
            _isOwner = true;
            _typeSize = sizeof(int);
            _dataType = NppDataType.NPP_32S;
            _nppChannels = NppiChannels.NPP_CH_4;

            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_32s_C4(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_32s_C4", res, _pitch, _channels));

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
        public NPPImage_32sC4(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = sizeof(int);
            _dataType = NppDataType.NPP_32S;
            _nppChannels = NppiChannels.NPP_CH_4;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32sC4(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_32sC4(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_32sC4(NppiSize size)
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
        public NPPImage_32sC4(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32sC4(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_32sC4()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<VectorTypes.int4> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<VectorTypes.int4>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<VectorTypes.int4>(NPPImage_32sC4 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_32sC4(CudaPitchedDeviceVariable<VectorTypes.int4> img)
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
        public void Copy(NPPImage_32sC1 dst, int channel)
        {
            if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..3].");
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C4C1R(_devPtrRoi + channel * _typeSize, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C4C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="channelSrc">Channel number. This number is added to the src pointer</param>
        /// <param name="channelDst">Channel number. This number is added to the dst pointer</param>
        public void Copy(NPPImage_32sC4 dst, int channelSrc, int channelDst)
        {
            if (channelSrc < 0 | channelSrc >= _channels) throw new ArgumentOutOfRangeException("channelSrc", "channelSrc must be in range [0..2].");
            if (channelDst < 0 | channelDst >= dst.Channels) throw new ArgumentOutOfRangeException("channelDst", "channelDst must be in range [0..2].");
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C4CR(_devPtrRoi + channelSrc * _typeSize, _pitch, dst.DevicePointerRoi + channelDst * _typeSize, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C4CR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Masked Operation 8-bit unsigned image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="mask">Mask image</param>
        public void Copy(NPPImage_32sC4 dst, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C4MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C4MR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Masked Operation 8-bit unsigned image copy. Not affecting Alpha channel.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="mask">Mask image</param>
        public void CopyA(NPPImage_32sC4 dst, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_AC4MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_AC4MR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Copy(NPPImage_32sC4 dst)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image copy. Not affecting Alpha channel.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void CopyA(NPPImage_32sC4 dst)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Three-channel 8-bit unsigned packed to planar image copy.
        /// </summary>
        /// <param name="dst0">Destination image channel 0</param>
        /// <param name="dst1">Destination image channel 1</param>
        /// <param name="dst2">Destination image channel 2</param>
        /// <param name="dst3">Destination image channel 3</param>
        public void Copy(NPPImage_32sC1 dst0, NPPImage_32sC1 dst1, NPPImage_32sC1 dst2, NPPImage_32sC1 dst3)
        {
            CUdeviceptr[] array = new CUdeviceptr[] { dst0.DevicePointerRoi, dst1.DevicePointerRoi, dst2.DevicePointerRoi, dst3.DevicePointerRoi };
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C4P4R(_devPtrRoi, _pitch, array, dst0.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C4P4R", status));
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
        public static void Copy(NPPImage_32sC1 src0, NPPImage_32sC1 src1, NPPImage_32sC1 src2, NPPImage_32sC1 src3, NPPImage_32sC4 dest)
        {
            CUdeviceptr[] array = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi, src3.DevicePointerRoi };
            NppStatus status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_P4C4R(array, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_P4C4R", status));
            NPPException.CheckNppStatus(status, null);
        }
        #endregion

        #region Convert
        /// <summary>
        /// 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_8uC4 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s8u_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s8u_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 8-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_8sC4 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s8s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s8s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 8-bit unsigned conversion. Not affecting Alpha
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void ConvertA(NPPImage_8uC4 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s8u_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s8u_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 8-bit signed conversion. Not affecting Alpha
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void ConvertA(NPPImage_8sC4 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s8s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s8s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set (Array size = 4)</param>
        public void Set(int[] nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32s_C4R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32s_C4R", status));
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
        public void Set(int[] nValue, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32s_C4MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32s_C4MR", status));
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
        public void Set(int nValue, int channel)
        {
            if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..3].");
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32s_C4CR(nValue, _devPtrRoi + channel * _typeSize, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32s_C4CR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Set pixel values to nValue. <para/>
        /// The 8-bit mask image affects setting of the respective pixels in the destination image. <para/>
        /// If the mask value is zero (0) the pixel is not set, if the mask is non-zero, the corresponding
        /// destination pixel is set to specified value. Not affecting alpha channel.
        /// </summary>
        /// <param name="nValue">Value to be set (Array size = 3)</param>
        public void SetA(int[] nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32s_AC4R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32s_AC4R", status));
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
        public void SetA(int[] nValue, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32s_AC4MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32s_AC4MR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Add
        ///// <summary>
        ///// Image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Add(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Add.nppiAdd_32s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32s_C4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Add(NPPImage_32sC4 src2, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Add.nppiAdd_32s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32s_C4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="nConstant">Values to add</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Add(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32s_C4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        ///// </summary>
        ///// <param name="nConstant">Values to add</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Add(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32s_C4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Image addition, scale by 2^(-nScaleFactor), then clamp to saturated value. Unmodified Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void AddA(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Add.nppiAdd_32s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value. Unmodified Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void AddA(NPPImage_32sC4 src2, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Add.nppiAdd_32s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unmodified Alpha.
        ///// </summary>
        ///// <param name="nConstant">Values to add</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void AddA(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unmodified Alpha.
        ///// </summary>
        ///// <param name="nConstant">Values to add</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void AddA(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        #endregion

        #region Logical
        /// <summary>
        /// image bit shift by constant (left).
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void LShiftC(uint[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_32s_C4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (left), inplace.
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        public void LShiftC(uint[] nConstant)
        {
            status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_32s_C4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (left). Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void LShiftCA(uint[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_32s_AC4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (left), inplace. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        public void LShiftCA(uint[] nConstant)
        {
            status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_32s_AC4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_32s_AC4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image bit shift by constant (right).
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void RShiftC(uint[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_32s_C4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (right), inplace.
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        public void RShiftC(uint[] nConstant)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_32s_C4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (right). Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void RShiftCA(uint[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_32s_AC4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (right), inplace. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 4)</param>
        public void RShiftCA(uint[] nConstant)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_32s_AC4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_32s_AC4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical and.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void And(NPPImage_32sC4 src2, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.And.nppiAnd_32s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical and.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void And(NPPImage_32sC4 src2)
        {
            status = NPPNativeMethods.NPPi.And.nppiAnd_32s_C4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical and with constant.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void And(int[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.AndConst.nppiAndC_32s_C4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical and with constant.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        public void And(int[] nConstant)
        {
            status = NPPNativeMethods.NPPi.AndConst.nppiAndC_32s_C4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical and. Unchanged Alpha.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void AndA(NPPImage_32sC4 src2, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.And.nppiAnd_32s_AC4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical and. Unchanged Alpha.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void AndA(NPPImage_32sC4 src2)
        {
            status = NPPNativeMethods.NPPi.And.nppiAnd_32s_AC4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_32s_AC4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical and with constant. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void AndA(int[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.AndConst.nppiAndC_32s_AC4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical and with constant. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        public void AndA(int[] nConstant)
        {
            status = NPPNativeMethods.NPPi.AndConst.nppiAndC_32s_AC4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_32s_AC4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Or.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Or(NPPImage_32sC4 src2, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.Or.nppiOr_32s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Or.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Or(NPPImage_32sC4 src2)
        {
            status = NPPNativeMethods.NPPi.Or.nppiOr_32s_C4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Or with constant.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void Or(int[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.OrConst.nppiOrC_32s_C4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Or with constant.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        public void Or(int[] nConstant)
        {
            status = NPPNativeMethods.NPPi.OrConst.nppiOrC_32s_C4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Or. Unchanged Alpha.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void OrA(NPPImage_32sC4 src2, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.Or.nppiOr_32s_AC4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Or. Unchanged Alpha.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void OrA(NPPImage_32sC4 src2)
        {
            status = NPPNativeMethods.NPPi.Or.nppiOr_32s_AC4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_32s_AC4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Or with constant. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void OrA(int[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.OrConst.nppiOrC_32s_AC4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Or with constant. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        public void OrA(int[] nConstant)
        {
            status = NPPNativeMethods.NPPi.OrConst.nppiOrC_32s_AC4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_32s_AC4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Xor.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Xor(NPPImage_32sC4 src2, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.Xor.nppiXor_32s_C4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Xor.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Xor(NPPImage_32sC4 src2)
        {
            status = NPPNativeMethods.NPPi.Xor.nppiXor_32s_C4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Xor with constant.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void Xor(int[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.XorConst.nppiXorC_32s_C4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Xor with constant.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        public void Xor(int[] nConstant)
        {
            status = NPPNativeMethods.NPPi.XorConst.nppiXorC_32s_C4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Xor. Unchanged Alpha.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void XorA(NPPImage_32sC4 src2, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.Xor.nppiXor_32s_AC4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Xor. Unchanged Alpha.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void XorA(NPPImage_32sC4 src2)
        {
            status = NPPNativeMethods.NPPi.Xor.nppiXor_32s_AC4IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_32s_AC4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Xor with constant. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        /// <param name="dest">Destination image</param>
        public void XorA(int[] nConstant, NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.XorConst.nppiXorC_32s_AC4R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Xor with constant. Unchanged Alpha.
        /// </summary>
        /// <param name="nConstant">Value (Array length = 4)</param>
        public void XorA(int[] nConstant)
        {
            status = NPPNativeMethods.NPPi.XorConst.nppiXorC_32s_AC4IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_32s_AC4IR", status));
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
        public void Sub(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_32s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32s_C4RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(NPPImage_32sC4 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_32s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32s_C4IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        ///// <summary>
        ///// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="nConstant">Value to subtract</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Sub(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32s_C4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        ///// </summary>
        ///// <param name="nConstant">Value to subtract</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Sub(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32s_C4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void SubA(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Sub.nppiSub_32s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void SubA(NPPImage_32sC4 src2, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Sub.nppiSub_32s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="nConstant">Value to subtract</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void SubA(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unchanged Alpha.
        ///// </summary>
        ///// <param name="nConstant">Value to subtract</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void SubA(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        #endregion

        #region Mul
        ///// <summary>
        ///// Image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Mul(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Mul.nppiMul_32s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32s_C4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Mul(NPPImage_32sC4 src2, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Mul.nppiMul_32s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32s_C4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Mul(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32s_C4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Mul(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32s_C4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}


        ///// <summary>
        ///// Image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void MulA(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Mul.nppiMul_32s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void MulA(NPPImage_32sC4 src2, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Mul.nppiMul_32s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void MulA(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unchanged Alpha.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void MulA(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        #endregion

        #region Div
        ///// <summary>
        ///// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Div(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Div.nppiDiv_32s_C4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32s_C4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Div(NPPImage_32sC4 src2, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Div.nppiDiv_32s_C4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32s_C4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Div(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32s_C4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32s_C4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void Div(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32s_C4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32s_C4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void DivA(NPPImage_32sC4 src2, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Div.nppiDiv_32s_AC4RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void DivA(NPPImage_32sC4 src2, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.Div.nppiDiv_32s_AC4IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}

        ///// <summary>
        ///// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Unchanged Alpha.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void DivA(int[] nConstant, NPPImage_32sC4 dest, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32s_AC4RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32s_AC4RSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
        ///// <summary>
        ///// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace. Unchanged Alpha.
        ///// </summary>
        ///// <param name="nConstant">Value</param>
        ///// <param name="nScaleFactor">scaling factor</param>
        //public void DivA(int[] nConstant, int nScaleFactor)
        //{
        //	status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32s_AC4IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32s_AC4IRSfs", status));
        //	NPPException.CheckNppStatus(status, this);
        //}
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
        #endregion

        #region Alpha Composition

        /// <summary>
        /// Four 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nppAlphaOp">alpha compositing operation</param>
        public void AlphaComp(NPPImage_32sC4 src2, NPPImage_32sC4 dest, NppiAlphaOp nppAlphaOp)
        {
            status = NPPNativeMethods.NPPi.AlphaComp.nppiAlphaComp_32s_AC4R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaComp_32s_AC4R", status));
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
        public void CopyConstBorderA(NPPImage_32sC4 dst, int nTopBorderHeight, int nLeftBorderWidth, int[] nValue)
        {
            status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_32s_AC4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_32s_AC4R", status));
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
        public void CopyReplicateBorderA(NPPImage_32sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_32s_AC4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_32s_AC4R", status));
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
        public void CopyWrapBorderA(NPPImage_32sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_32s_AC4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_32s_AC4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// linearly interpolated source image subpixel coordinate color copy. Not affecting Alpha.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        public void CopySubpixA(NPPImage_32sC4 dst, float nDx, float nDy)
        {
            status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_32s_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_32s_AC4R", status));
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
        public void Copy(NPPImage_32sC4 dst, int nTopBorderHeight, int nLeftBorderWidth, int[] nValue)
        {
            status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_32s_C4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_32s_C4R", status));
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
        public void CopyReplicateBorder(NPPImage_32sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_32s_C4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_32s_C4R", status));
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
        public void CopyWrapBorder(NPPImage_32sC4 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_32s_C4R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        public void CopySubpix(NPPImage_32sC4 dst, float nDx, float nDy)
        {
            status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_32s_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region SwapChannel
        /// <summary>
        /// Swap channels. Not affecting Alpha
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="aDstOrder">Integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of
        /// the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
        /// channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
        /// </param>
        public void SwapChannelsA(NPPImage_32sC4 dest, int[] aDstOrder)
        {
            status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_32s_AC4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_32s_AC4R", status));
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
        public void SwapChannels(NPPImage_32sC3 dest, int[] aDstOrder)
        {
            status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_32s_C4C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_32s_C4C3R", status));
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
            status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_32s_C4IR(_devPtrRoi, _pitch, _sizeRoi, aDstOrder);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_32s_C4IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Swap channels.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="aDstOrder">Integer array describing how channel values are permutated. The n-th entry of the array
        /// contains the number of the channel that is stored in the n-th channel of the output image. E.g.
        /// Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR channel order.</param>
        public void SwapChannels(NPPImage_32sC4 dest, int[] aDstOrder)
        {
            status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_32s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Scale


        /// <summary>
        /// image conversion. Not affecting Alpha.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        public void ScaleA(NPPImage_8uC4 dst, NppHintAlgorithm hint)
        {
            NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
            status = NPPNativeMethods.NPPi.Scale.nppiScale_32s8u_AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, hint);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_32s8u_AC4R", status));
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
            status = NPPNativeMethods.NPPi.Scale.nppiScale_32s8u_C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, hint);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_32s8u_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Transpose
        /// <summary>
        /// image transpose
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Transpose(NPPImage_32sC4 dest)
        {
            status = NPPNativeMethods.NPPi.Transpose.nppiTranspose_32s_C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_32s_C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

    }
}
