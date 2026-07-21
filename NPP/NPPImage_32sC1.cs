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
    public partial class NPPImage_32sC1 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_32sC1(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 1;
            _isOwner = true;
            _typeSize = sizeof(int);
            _dataType = NppDataType.NPP_32S;
            _nppChannels = NppiChannels.NPP_CH_1;

            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_32s_C1(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_32s_C1", res, _pitch, _channels));

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
        public NPPImage_32sC1(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = sizeof(int);
            _dataType = NppDataType.NPP_32S;
            _nppChannels = NppiChannels.NPP_CH_1;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32sC1(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_32sC1(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_32sC1(NppiSize size)
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
        public NPPImage_32sC1(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32sC1(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_32sC1()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<int> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<int>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<int>(NPPImage_32sC1 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_32sC1(CudaPitchedDeviceVariable<int> img)
        {
            return img.ToNPPImage();
        }
        #endregion

        #region Copy
        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Copy(NPPImage_32sC1 dst)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Masked Operation 8-bit unsigned image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="mask">Mask image</param>
        public void Copy(NPPImage_32sC1 dst, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C1MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C1MR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="channel">Channel number. This number is added to the dst pointer</param>
        public void Copy(NPPImage_32sC3 dst, int channel)
        {
            if (channel < 0 | channel >= dst.Channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C1C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi + channel * _typeSize, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C1C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="channel">Channel number. This number is added to the dst pointer</param>
        public void Copy(NPPImage_32sC4 dst, int channel)
        {
            if (channel < 0 | channel >= dst.Channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..3].");
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32s_C1C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi + channel * _typeSize, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32s_C1C4R", status));
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
        public void Copy(NPPImage_32sC1 dst, int nTopBorderHeight, int nLeftBorderWidth, int nValue)
        {
            status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_32s_C1R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set</param>
        public void Set(int nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32s_C1R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32s_C1R", status));
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
        public void Set(int nValue, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32s_C1MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32s_C1MR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Convert
        /// <summary>
        /// 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_8uC1 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s8u_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s8u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 8-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_8sC1 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s8s_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s8s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="roundMode">Round mode</param>
        /// <param name="scaleFactor">scaling factor</param>
        public void Convert(NPPImage_16uC1 dst, NppRoundMode roundMode, int scaleFactor)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s16u_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s16u_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 16-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="roundMode">Round mode</param>
        /// <param name="scaleFactor">scaling factor</param>
        public void Convert(NPPImage_16sC1 dst, NppRoundMode roundMode, int scaleFactor)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s16s_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s16s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 32-bit unsigned conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32uC1 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s32u_C1Rs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s32u_C1Rs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit signed to 32-bit float conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32fC1 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32s32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32s32f_C1R", status));
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
        public void Add(NPPImage_32sC1 src2, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_32s_C1RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(NPPImage_32sC1 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_32s_C1IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(int nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(int nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_32s_C1IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Logical
        /// <summary>
        /// image bit shift by constant (left).
        /// </summary>
        /// <param name="nConstant">Constant</param>
        /// <param name="dest">Destination image</param>
        public void LShiftC(uint nConstant, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_32s_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (left), inplace.
        /// </summary>
        /// <param name="nConstant">Constant</param>
        public void LShiftC(uint nConstant)
        {
            status = NPPNativeMethods.NPPi.LeftShiftConst.nppiLShiftC_32s_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLShiftC_32s_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (right).
        /// </summary>
        /// <param name="nConstant">Constant</param>
        /// <param name="dest">Destination image</param>
        public void RShiftC(uint nConstant, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_32s_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (right), inplace.
        /// </summary>
        /// <param name="nConstant">Constant</param>
        public void RShiftC(uint nConstant)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_32s_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_32s_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical and.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void And(NPPImage_32sC1 src2, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.And.nppiAnd_32s_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical and.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void And(NPPImage_32sC1 src2)
        {
            status = NPPNativeMethods.NPPi.And.nppiAnd_32s_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAnd_32s_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical and with constant.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void And(int nConstant, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.AndConst.nppiAndC_32s_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical and with constant.
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void And(int nConstant)
        {
            status = NPPNativeMethods.NPPi.AndConst.nppiAndC_32s_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAndC_32s_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Or.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Or(NPPImage_32sC1 src2, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.Or.nppiOr_32s_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Or.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Or(NPPImage_32sC1 src2)
        {
            status = NPPNativeMethods.NPPi.Or.nppiOr_32s_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOr_32s_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Or with constant.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void Or(int nConstant, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.OrConst.nppiOrC_32s_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Or with constant.
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void Or(int nConstant)
        {
            status = NPPNativeMethods.NPPi.OrConst.nppiOrC_32s_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiOrC_32s_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Xor.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Xor(NPPImage_32sC1 src2, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.Xor.nppiXor_32s_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Xor.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Xor(NPPImage_32sC1 src2)
        {
            status = NPPNativeMethods.NPPi.Xor.nppiXor_32s_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXor_32s_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image logical Xor with constant.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void Xor(int nConstant, NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.XorConst.nppiXorC_32s_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image logical Xor with constant.
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void Xor(int nConstant)
        {
            status = NPPNativeMethods.NPPi.XorConst.nppiXorC_32s_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiXorC_32s_C1IR", status));
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
        public void Sub(NPPImage_32sC1 src2, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_32s_C1RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(NPPImage_32sC1 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_32s_C1IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(int nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(int nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_32s_C1IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32s_C1IRSfs", status));
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
        public void Mul(NPPImage_32sC1 src2, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_32s_C1RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(NPPImage_32sC1 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_32s_C1IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(int nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(int nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_32s_C1IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32s_C1IRSfs", status));
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
        public void Div(NPPImage_32sC1 src2, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_32s_C1RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(NPPImage_32sC1 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_32s_C1IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(int nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(int nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_32s_C1IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Alpha Composition
        ///// <summary>
        ///// Image composition using image alpha values (0 - max channel pixel value).
        ///// Also the function is called *AC1R, it is a two channel image with second channel as alpha.
        ///// </summary>
        ///// <param name="src2">2nd source image</param>
        ///// <param name="dest">Destination image</param>
        ///// <param name="nppAlphaOp">alpha compositing operation</param>
        //public void AlphaComp(NPPImage_32sC1 src2, NPPImage_32sC1 dest, NppiAlphaOp nppAlphaOp)
        //{
        //    status = NPPNativeMethods.NPPi.AlphaComp.nppiAlphaComp_32s_AC1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaComp_32s_AC1R", status));
        //    NPPException.CheckNppStatus(status, this);
        //}

        /// <summary>
        /// Image composition using constant alpha.
        /// </summary>
        /// <param name="alpha1">constant alpha for this image</param>
        /// <param name="src2">2nd source image</param>
        /// <param name="alpha2">constant alpha for src2</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nppAlphaOp">alpha compositing operation</param>
        public void AlphaComp(int alpha1, NPPImage_32sC1 src2, int alpha2, NPPImage_32sC1 dest, NppiAlphaOp nppAlphaOp)
        {
            status = NPPNativeMethods.NPPi.AlphaCompConst.nppiAlphaCompC_32s_C1R(_devPtrRoi, _pitch, alpha1, src2.DevicePointerRoi, src2.Pitch, alpha2, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaCompC_32s_C1R", status));
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

        //new in Cuda 5.5
        #region DupNew
        /// <summary>
        /// source image duplicated in all 3 channels of destination image.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        public void Dup(NPPImage_32sC3 dst)
        {
            status = NPPNativeMethods.NPPi.Dup.nppiDup_32s_C1C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_32s_C1C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// source image duplicated in all 4 channels of destination image.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        public void Dup(NPPImage_32sC4 dst)
        {
            status = NPPNativeMethods.NPPi.Dup.nppiDup_32s_C1C4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_32s_C1C4R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        public void DupA(NPPImage_32sC4 dst)
        {
            status = NPPNativeMethods.NPPi.Dup.nppiDup_32s_C1AC4R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDup_32s_C1AC4R", status));
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
        /// the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth =
        /// oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        public void CopyReplicateBorder(NPPImage_32sC1 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_32s_C1R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_32s_C1R", status));
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
        public void CopyWrapBorder(NPPImage_32sC1 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_32s_C1R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        public void CopySubpix(NPPImage_32sC1 dst, float nDx, float nDy)
        {
            status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_32s_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Transpose
        /// <summary>
        /// image transpose
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Transpose(NPPImage_32sC1 dest)
        {
            status = NPPNativeMethods.NPPi.Transpose.nppiTranspose_32s_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_32s_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Scale
        /// <summary>
        /// image conversion.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        public void Scale(NPPImage_8uC1 dst, NppHintAlgorithm hint)
        {
            NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
            status = NPPNativeMethods.NPPi.Scale.nppiScale_32s8u_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, hint);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_32s8u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region New in Cuda 12.0
#if ADD_MISSING_CTX
        #region Add
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(CudaDeviceVariable<int> nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(CudaDeviceVariable<int> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_32s_C1IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion
        #region Sub

        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(CudaDeviceVariable<int> nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(CudaDeviceVariable<int> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_32s_C1IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Mul

        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(CudaDeviceVariable<int> nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(CudaDeviceVariable<int> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_32s_C1IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Div

        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(CudaDeviceVariable<int> nConstant, NPPImage_32sC1 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_32s_C1RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(CudaDeviceVariable<int> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_32s_C1IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_32s_C1IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion
#endif
        #endregion
    }
}
