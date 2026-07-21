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
    public partial class NPPImage_16sC3 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_16sC3(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 3;
            _isOwner = true;
            _typeSize = sizeof(short);
            _dataType = NppDataType.NPP_16S;
            _nppChannels = NppiChannels.NPP_CH_3;

            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_16u_C3(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_16u_C3", res, _pitch, _channels));

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
        public NPPImage_16sC3(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = sizeof(short);
            _dataType = NppDataType.NPP_16S;
            _nppChannels = NppiChannels.NPP_CH_3;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16sC3(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_16sC3(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_16sC3(NppiSize size)
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
        public NPPImage_16sC3(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16sC3(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_16sC3()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<VectorTypes.short3> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<VectorTypes.short3>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<VectorTypes.short3>(NPPImage_16sC3 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_16sC3(CudaPitchedDeviceVariable<VectorTypes.short3> img)
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
            if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C3C1R(_devPtrRoi + channel * _typeSize, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C3C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Three-channel 8-bit unsigned packed to planar image copy.
        /// </summary>
        /// <param name="dst0">Destination image channel 0</param>
        /// <param name="dst1">Destination image channel 1</param>
        /// <param name="dst2">Destination image channel 2</param>
        public void Copy(NPPImage_16sC1 dst0, NPPImage_16sC1 dst1, NPPImage_16sC1 dst2)
        {
            CUdeviceptr[] array = new CUdeviceptr[] { dst0.DevicePointerRoi, dst1.DevicePointerRoi, dst2.DevicePointerRoi };
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C3P3R(_devPtrRoi, _pitch, array, dst0.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C3P3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Three-channel 8-bit unsigned planar to packed image copy.
        /// </summary>
        /// <param name="src0">Source image channel 0</param>
        /// <param name="src1">Source image channel 1</param>
        /// <param name="src2">Source image channel 2</param>
        /// <param name="dest">Destination image</param>
        public static void Copy(NPPImage_16sC1 src0, NPPImage_16sC1 src1, NPPImage_16sC1 src2, NPPImage_16sC3 dest)
        {
            CUdeviceptr[] array = new CUdeviceptr[] { src0.DevicePointerRoi, src1.DevicePointerRoi, src2.DevicePointerRoi };
            NppStatus status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_P3C3R(array, src0.Pitch, dest.DevicePointerRoi, dest.Pitch, dest.SizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_P3C3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="channelSrc">Channel number. This number is added to the src pointer</param>
        /// <param name="channelDst">Channel number. This number is added to the dst pointer</param>
        public void Copy(NPPImage_16sC3 dst, int channelSrc, int channelDst)
        {
            if (channelSrc < 0 | channelSrc >= _channels) throw new ArgumentOutOfRangeException("channelSrc", "channelSrc must be in range [0..2].");
            if (channelDst < 0 | channelDst >= dst.Channels) throw new ArgumentOutOfRangeException("channelDst", "channelDst must be in range [0..2].");
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C3CR(_devPtrRoi + channelSrc * _typeSize, _pitch, dst.DevicePointerRoi + channelDst * _typeSize, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C3CR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Masked Operation 8-bit unsigned image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="mask">Mask image</param>
        public void Copy(NPPImage_16sC3 dst, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C3MR(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C3MR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Logical
        /// <summary>
        /// image bit shift by constant (right).
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 3)</param>
        /// <param name="dest">Destination image</param>
        public void RShiftC(uint[] nConstant, NPPImage_16sC3 dest)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_16s_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (right), inplace.
        /// </summary>
        /// <param name="nConstant">Constant (Array length = 3)</param>
        public void RShiftC(uint[] nConstant)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_16s_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_16s_C3IR", status));
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
        public void Add(NPPImage_16sC3 src2, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_16s_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(NPPImage_16sC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_16s_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Values to add</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(short[] nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Values to add</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(short[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16s_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16s_C3IRSfs", status));
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
        public void Sub(NPPImage_16sC3 src2, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_16s_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(NPPImage_16sC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_16s_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(short[] nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(short[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16s_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16s_C3IRSfs", status));
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
        public void Mul(NPPImage_16sC3 src2, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_16s_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(NPPImage_16sC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_16s_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(short[] nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(short[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16s_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16s_C3IRSfs", status));
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
        public void Div(NPPImage_16sC3 src2, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_16s_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(NPPImage_16sC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_16s_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(short[] nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(short[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16s_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        /// <param name="rndMode">Result Rounding mode to be used</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(NPPImage_16sC3 src2, NPPImage_16sC3 dest, NppRoundMode rndMode, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_16s_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, rndMode, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="rndMode">Result Rounding mode to be used</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(NPPImage_16sC3 src2, NppRoundMode rndMode, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivRound.nppiDiv_Round_16s_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, rndMode, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_Round_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Exp
        /// <summary>
        /// Exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Exp(NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Exp.nppiExp_16s_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Inplace exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Exp(int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Exp.nppiExp_16s_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiExp_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Ln
        /// <summary>
        /// Natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Ln(NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Ln.nppiLn_16s_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Ln(int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Ln.nppiLn_16s_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Sqr
        /// <summary>
        /// Image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sqr(NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16s_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Inplace image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sqr(int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16s_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Sqrt
        /// <summary>
        /// Image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sqrt(NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16s_C3RSfs(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Inplace image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sqrt(int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16s_C3IRSfs(_devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Abs
        /// <summary>
        /// Image absolute value.
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Abs(NPPImage_16sC3 dest)
        {
            status = NPPNativeMethods.NPPi.Abs.nppiAbs_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image absolute value. In place.
        /// </summary>
        public void Abs()
        {
            status = NPPNativeMethods.NPPi.Abs.nppiAbs_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16s_C3IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Convert
        /// <summary>
        /// 16-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32sC3 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s32s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s32s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 16-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_8uC3 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 16-bit unsigned to 32-bit floating point conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32fC3 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16s32f_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16s32f_C3R", status));
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
        public void Threshold(NPPImage_16sC3 dest, short[] nThreshold, NppCmpOp eComparisonOperation)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, eComparisonOperation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image threshold.<para/>
        /// If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
        /// to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">eComparisonOperation. Only allowed values are <see cref="NppCmpOp.Less"/> and <see cref="NppCmpOp.Greater"/></param>
        public void Threshold(short[] nThreshold, NppCmpOp eComparisonOperation)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, eComparisonOperation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_16s_C3IR", status));
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
        public void ThresholdGT(NPPImage_16sC3 dest, short[] nThreshold)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image threshold.<para/>
        /// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
        /// to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="nThreshold">The threshold value.</param>
        public void ThresholdGT(short[] nThreshold)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GT_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GT_16s_C3IR", status));
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
        public void ThresholdLT(NPPImage_16sC3 dest, short[] nThreshold)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image threshold.<para/>
        /// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
        /// to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="nThreshold">The threshold value.</param>
        public void ThresholdLT(short[] nThreshold)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LT_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LT_16s_C3IR", status));
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
        public void Threshold(NPPImage_16sC3 dest, short[] nThreshold, short[] nValue, NppCmpOp eComparisonOperation)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_16s_C3R", status));
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
        public void Threshold(short[] nThreshold, short[] nValue, NppCmpOp eComparisonOperation)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_Val_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue, eComparisonOperation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_Val_16s_C3IR", status));
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
        public void ThresholdGT(NPPImage_16sC3 dest, short[] nThreshold, short[] nValue)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image threshold.<para/>
        /// If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
        /// to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        public void ThresholdGT(short[] nThreshold, short[] nValue)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_GTVal_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_GTVal_16s_C3IR", status));
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
        public void ThresholdLT(NPPImage_16sC3 dest, short[] nThreshold, short[] nValue)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThreshold, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image threshold.<para/>
        /// If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
        /// to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        public void ThresholdLT(short[] nThreshold, short[] nValue)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTVal_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThreshold, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTVal_16s_C3IR", status));
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
        public void ThresholdLTGT(NPPImage_16sC3 dest, short[] nThresholdLT, short[] nValueLT, short[] nThresholdGT, short[] nValueGT)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_16s_C3R", status));
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
        public void ThresholdLTGT(short[] nThresholdLT, short[] nValueLT, short[] nThresholdGT, short[] nValueGT)
        {
            status = NPPNativeMethods.NPPi.Threshold.nppiThreshold_LTValGTVal_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, nThresholdLT, nValueLT, nThresholdGT, nValueGT);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiThreshold_LTValGTVal_16s_C3IR", status));
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
        public void Compare(NPPImage_16sC3 src2, NPPImage_8uC1 dest, NppCmpOp eComparisonOperation)
        {
            status = NPPNativeMethods.NPPi.Compare.nppiCompare_16s_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompare_16s_C3R", status));
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
            status = NPPNativeMethods.NPPi.Compare.nppiCompareC_16s_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eComparisonOperation);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompareC_16s_C3R", status));
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

        #endregion

        //new in Cuda 5.5
        #region Transpose
        /// <summary>
        /// image transpose
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Transpose(NPPImage_16sC3 dest)
        {
            status = NPPNativeMethods.NPPi.Transpose.nppiTranspose_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_16s_C3R", status));
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
        public void SwapChannels(NPPImage_16sC3 dest, int[] aDstOrder)
        {
            status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_16s_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_16s_C3R", status));
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
        public void SwapChannels(NPPImage_16sC4 dest, int[] aDstOrder, short nValue)
        {
            status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_16s_C3C4R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, aDstOrder, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_16s_C3C4R", status));
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
            status = NPPNativeMethods.NPPi.SwapChannel.nppiSwapChannels_16s_C3IR(_devPtrRoi, _pitch, _sizeRoi, aDstOrder);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSwapChannels_16s_C3IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set (Array size = 3)</param>
        public void Set(short[] nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_C3R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_C3R", status));
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
        public void Set(short[] nValue, NPPImage_8uC1 mask)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_C3MR(nValue, _devPtrRoi, _pitch, _sizeRoi, mask.DevicePointerRoi, mask.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_C3MR", status));
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
            if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..2].");
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_16s_C3CR(nValue, _devPtrRoi + channel * _typeSize, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16s_C3CR", status));
            NPPException.CheckNppStatus(status, this);
        }

        #endregion

        #region Copy

        /// <summary>
        /// image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Copy(NPPImage_16sC3 dst)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16s_C3R", status));
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
        public void Copy(NPPImage_16sC3 dst, int nTopBorderHeight, int nLeftBorderWidth, short[] nValue)
        {
            status = NPPNativeMethods.NPPi.CopyConstBorder.nppiCopyConstBorder_16s_C3R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth, nValue);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyConstBorder_16s_C3R", status));
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
        public void CopyReplicateBorder(NPPImage_16sC3 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyReplicateBorder.nppiCopyReplicateBorder_16s_C3R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyReplicateBorder_16s_C3R", status));
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
        public void CopyWrapBorder(NPPImage_16sC3 dst, int nTopBorderHeight, int nLeftBorderWidth)
        {
            status = NPPNativeMethods.NPPi.CopyWrapBorder.nppiCopyWrapBorder_16s_C3R(_devPtrRoi, _pitch, _sizeRoi, dst.DevicePointerRoi, dst.Pitch, dst.SizeRoi, nTopBorderHeight, nLeftBorderWidth);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopyWrapBorder_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        public void CopySubpix(NPPImage_16sC3 dst, float nDx, float nDy)
        {
            status = NPPNativeMethods.NPPi.CopySubpix.nppiCopySubpix_16s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nDx, nDy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopySubpix_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion


        #region GeometryNew

        /// <summary>
        /// image conversion.
        /// </summary>
        /// <param name="dst">Destination-Image</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        public void Scale(NPPImage_8uC3 dst, NppHintAlgorithm hint)
        {
            NppiRect srcRect = new NppiRect(_pointRoi, _sizeRoi);
            status = NPPNativeMethods.NPPi.Scale.nppiScale_16s8u_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, hint);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiScale_16s8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }


        #endregion

        #region SwapChannelNew


        #endregion

        //New in Cuda 6.0


        //New in Cuda 7.0
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
            status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpGetBufferSize_16s_C3R(nRadius, nSigma, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpGetBufferSize_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #endregion


        #region New in Cuda 12.0

        /// <summary>
        /// Returns the required size of host memory buffer needed by most nppiFilterBoxBorderAdvanced functions.
        /// </summary>
        public int FilterBoxBorderAdvancedGetDeviceBufferSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorderAdvancedGetDeviceBufferSize(_sizeRoi, _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorderAdvancedGetDeviceBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

#if ADD_MISSING_CTX

        /// <summary>
        /// median filter scratch memory size.
        /// </summary>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        public uint FilterMedianBorderGetBufferSize(NppiSize oMaskSize, NppiBorderType eBorderType)
        {
            uint bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianBorderGetBufferSize_16s_C3R(_sizeRoi, oMaskSize, ref bufferSize, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianBorderGetBufferSize_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #region Add
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(CudaDeviceVariable<short> nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(CudaDeviceVariable<short> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_16s_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_16s_C3IRSfs", status));
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
        public void Sub(CudaDeviceVariable<short> nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(CudaDeviceVariable<short> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_16s_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_16s_C3IRSfs", status));
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
        public void Mul(CudaDeviceVariable<short> nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(CudaDeviceVariable<short> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_16s_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_16s_C3IRSfs", status));
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
        public void Div(CudaDeviceVariable<short> nConstant, NPPImage_16sC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_16s_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_16s_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(CudaDeviceVariable<short> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_16s_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_16s_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        /// <summary>
        /// median filter with border control.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <param name="filterArea">The area where the filter is allowed to read pixels. The point is relative to the ROI set to source image, the size is the total size starting from the filterArea point. Default value is the set ROI.</param>
        public void FilterMedianBorder(NPPImage_16sC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, CudaDeviceVariable<byte> pBuffer, NppiRect filterArea = new NppiRect())
        {
            if (filterArea.Size == new NppiSize())
            {
                filterArea.Size = _sizeRoi;
            }
            status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianBorder_16s_C3R(_devPtrRoi, _pitch, filterArea.Size, filterArea.Location, dest.DevicePointerRoi,
                                                    dest.Pitch, dest.SizeRoi, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianBorder_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }




        /// <summary>
        /// Box filter with border control. 
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference w.r.t the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <param name="filterArea">The area where the filter is allowed to read pixels. The point is relative to the ROI set to source image, the size is the total size starting from the filterArea point. Default value is the set ROI.</param>
        public void FilterBoxBorderAdvanced(NPPImage_16sC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, CudaDeviceVariable<byte> pBuffer, NppiRect filterArea = new NppiRect())
        {
            if (filterArea.Size == new NppiSize())
            {
                filterArea.Size = _sizeRoi;
            }
            status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorderAdvanced_16s_C3R(_devPtrRoi, _pitch, filterArea.Size, filterArea.Location, dest.DevicePointerRoi,
                                                    dest.Pitch, dest.SizeRoi, oMaskSize, oAnchor, eBorderType, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorderAdvanced_16s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
#endif
        #endregion
    }
}
