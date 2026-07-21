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
            _dataType = NppDataType.NPP_8U;
            _nppChannels = NppiChannels.NPP_CH_3;

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
            _dataType = NppDataType.NPP_8U;
            _nppChannels = NppiChannels.NPP_CH_3;
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
            Dispose(false);
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
        public void Transpose(NPPImage_8uC3 dest)
        {
            status = NPPNativeMethods.NPPi.Transpose.nppiTranspose_8u_C3R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiTranspose_8u_C3R", status));
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

        #region GeometryNew

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

        #endregion


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
            status = NPPNativeMethods.NPPi.FixedFilters.nppiFilterUnsharpGetBufferSize_8u_C3R(nRadius, nSigma, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterUnsharpGetBufferSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        #endregion

        //New in Cuda 8.0

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


        #endregion



        //New in Cuda 9.0

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


        #endregion

        #region New in Cuda 11.2

        /// <summary>
        /// Three-channel 8-bit unsigned image MSE.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pMSE">Pointer to the computed MSE of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiMSEGetBufferHostSize_8u_C3R to compute the required size (in bytes).</param>
        public void MSE(NPPImage_8uC3 src2, CudaDeviceVariable<float> pMSE, CudaDeviceVariable<byte> pDeviceBuffer)
        {
            status = NPPNativeMethods.NPPi.IQA.nppiMSE_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pMSE.DevicePointer, pDeviceBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSE_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Three-channel 8-bit unsigned image PSNR.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pPSNR">Pointer to the computed PSNR of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiPSNRGetBufferHostSize_8u_C3R to compute the required size (in bytes).</param>
        public void PSNR(NPPImage_8uC3 src2, CudaDeviceVariable<float> pPSNR, CudaDeviceVariable<byte> pDeviceBuffer)
        {
            status = NPPNativeMethods.NPPi.IQA.nppiPSNR_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pPSNR.DevicePointer, pDeviceBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiPSNR_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Three-channel 8-bit unsigned image SSIM.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pSSIM">Pointer to the computed SSIM of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiSSIMGetBufferHostSize_8u_C3R to compute the required size (in bytes).</param>
        public void SSIM(NPPImage_8uC3 src2, CudaDeviceVariable<float> pSSIM, CudaDeviceVariable<byte> pDeviceBuffer)
        {
            status = NPPNativeMethods.NPPi.IQA.nppiSSIM_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pSSIM.DevicePointer, pDeviceBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSSIM_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Three-channel 8-bit unsigned image MS-SSIM*.
        /// This function uses the algorithm described in the paper by Wang et.al.
        /// Wang, Z., Simoncelli, E.P., Bovik, A.C.Multiscale Structural Similarity for Image Quality Assessment.In: The
        /// Thirty-Seventh Asilomar Conference on Signals, Systems and Computers, 2003, 13981402. Pacific Grove, CA, USA: IEEE,
        /// 2003. https://doi.org/10.1109/ACSSC.2003.1292216. <para/>
        /// NOTE: this API call can only process oSizeROI dimensions 16px by 16px and above.Any oSizeROI dimensions less than 16px by 16px
        /// will result in undefined behaviour.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pMSSSIM">Pointer to the computed SSIM of two images. </param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation. Use \ref nppiWMSSSIMGetBufferHostSize_8u_C3R to compute the required size (in bytes).</param>
        public void WMSSSIM(NPPImage_8uC3 src2, CudaDeviceVariable<float> pMSSSIM, CudaDeviceVariable<byte> pDeviceBuffer)
        {
            status = NPPNativeMethods.NPPi.IQA.nppiWMSSSIM_8u_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pMSSSIM.DevicePointer, pDeviceBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWMSSSIM_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Buffer size for \ref nppiMSE_8u_C3R.
        /// </summary>
        public SizeT MSEGetBufferHostSize()
        {
            SizeT ret = 0;
            status = NPPNativeMethods.NPPi.IQA.nppiMSEGetBufferHostSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSEGetBufferHostSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// Buffer size for \ref nppiPSNR_8u_C3R.
        /// </summary>
        public SizeT PSNRGetBufferHostSize()
        {
            SizeT ret = 0;
            status = NPPNativeMethods.NPPi.IQA.nppiPSNRGetBufferHostSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiPSNRGetBufferHostSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// Buffer size for \ref nppiSSIM_8u_C3R.
        /// </summary>
        public SizeT SSIMGetBufferHostSize()
        {
            SizeT ret = 0;
            status = NPPNativeMethods.NPPi.IQA.nppiSSIMGetBufferHostSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSSIMGetBufferHostSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// Buffer size for \ref nppiWMSSSIM_8u_C3R.
        /// </summary>
        public SizeT WMSSSIMGetBufferHostSize()
        {
            SizeT ret = 0;
            status = NPPNativeMethods.NPPi.IQA.nppiWMSSSIMGetBufferHostSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWMSSSIMGetBufferHostSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }



        /// <summary>
        /// 3 channel 8-bit unsigned MSE for a batch of image pairs for a single ROI.
        /// Provided oSizeROI will be used for all images passed in pSrc1BatchList and pSrc2BatchList
        /// arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
        /// </summary>
        /// <param name="oSizeROI"></param>
        /// <param name="pBatchSrc1">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pBatchSrc2">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pMSE">Device memory pointer to output array of the computed MSE for nBatchSize* sizeof(Npp32f)* 3 image pairs.</param>
        /// <param name="pDeviceBufferList">Device memory pointer to the list of NppiBufferDescriptor buffer descriptors specifying per image device
        /// memory buffer pointers and size as returned by at least one MSEBatchGetBufferHostSize call.</param>
        public static void MSEBatch(NppiSize oSizeROI, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc1, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc2, CudaDeviceVariable<float> pMSE,
                                        CudaDeviceVariable<NppiBufferDescriptor> pDeviceBufferList)
        {
            NppStatus status = NPPNativeMethods.NPPi.IQA.nppiMSEBatch_8u_C3R(pBatchSrc1.DevicePointer, pBatchSrc2.DevicePointer, pBatchSrc1.Size, oSizeROI, pMSE.DevicePointer,
                pDeviceBufferList.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSEBatch_8u_C3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 3 channel 8-bit unsigned MSE for a batch of image pairs with per-image ROI
        /// </summary>
        /// <param name="oMaxSizeROI">maximum ROI width and height of ALL images in the batch.</param>
        /// <param name="pBatchSrc1">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pBatchSrc2">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pMSE">Device memory pointer to output array of the computed MSE for nBatchSize* sizeof(Npp32f)* 3 image pairs.</param>
        /// <param name="pDeviceBufferList">Device memory pointer to the list of NppiBufferDescriptor buffer descriptors specifying per image device
        /// memory buffer pointers and size as returned by at least one MSEBatchGetBufferHostSize call.</param>
        public static void MSEBatchAdvanced(NppiSize oMaxSizeROI, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc1, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc2, CudaDeviceVariable<float> pMSE,
                                        CudaDeviceVariable<NppiBufferDescriptor> pDeviceBufferList)
        {
            NppStatus status = NPPNativeMethods.NPPi.IQA.nppiMSEBatch_8u_C3R_Advanced(pBatchSrc1.DevicePointer, pBatchSrc2.DevicePointer, pBatchSrc1.Size, oMaxSizeROI, pMSE.DevicePointer,
                pDeviceBufferList.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSEBatch_8u_C3R_Advanced", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// Buffer size for \ref nppiMSEBatch_8u_C3R.
        /// </summary>
        public SizeT MSEBatchGetBufferHostSize()
        {
            SizeT ret = 0;
            status = NPPNativeMethods.NPPi.IQA.nppiMSEBatchGetBufferHostSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMSEBatchGetBufferHostSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// 3 channel 8-bit unsigned PSNR for a batch of image pairs for a single ROI.
        /// Provided oSizeROI will be used for all images passed in pSrc1BatchList and pSrc2BatchList
        /// arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
        /// </summary>
        /// <param name="oSizeROI"></param>
        /// <param name="pBatchSrc1">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pBatchSrc2">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pPSNR">Device memory pointer to output array of the computed PSNR for nBatchSize* sizeof(Npp32f)* 3 image pairs.</param>
        /// <param name="pDeviceBufferList">Device memory pointer to the list of NppiBufferDescriptor buffer descriptors specifying per image device
        /// memory buffer pointers and size as returned by at least one PSNRBatchGetBufferHostSize call.</param>
        public static void PSNRBatch(NppiSize oSizeROI, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc1, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc2, CudaDeviceVariable<float> pPSNR,
                                        CudaDeviceVariable<NppiBufferDescriptor> pDeviceBufferList)
        {
            NppStatus status = NPPNativeMethods.NPPi.IQA.nppiPSNRBatch_8u_C3R(pBatchSrc1.DevicePointer, pBatchSrc2.DevicePointer, pBatchSrc1.Size, oSizeROI, pPSNR.DevicePointer,
                pDeviceBufferList.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiPSNRBatch_8u_C3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 3 channel 8-bit unsigned PSNR for a batch of image pairs with per-image ROI
        /// </summary>
        /// <param name="oMaxSizeROI">maximum ROI width and height of ALL images in the batch.</param>
        /// <param name="pBatchSrc1">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pBatchSrc2">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pPSNR">Device memory pointer to output array of the computed PSNR for nBatchSize* sizeof(Npp32f)* 3 image pairs.</param>
        /// <param name="pDeviceBufferList">Device memory pointer to the list of NppiBufferDescriptor buffer descriptors specifying per image device
        /// memory buffer pointers and size as returned by at least one PSNRBatchGetBufferHostSize call.</param>
        public static void PSNRBatchAdvanced(NppiSize oMaxSizeROI, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc1, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc2, CudaDeviceVariable<float> pPSNR,
                                        CudaDeviceVariable<NppiBufferDescriptor> pDeviceBufferList)
        {
            NppStatus status = NPPNativeMethods.NPPi.IQA.nppiPSNRBatch_8u_C3R_Advanced(pBatchSrc1.DevicePointer, pBatchSrc2.DevicePointer, pBatchSrc1.Size, oMaxSizeROI, pPSNR.DevicePointer,
                pDeviceBufferList.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiPSNRBatch_8u_C3R_Advanced", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// Buffer size for \ref nppiPSNRBatch_8u_C3R.
        /// </summary>
        public SizeT PSNRBatchGetBufferHostSize()
        {
            SizeT ret = 0;
            status = NPPNativeMethods.NPPi.IQA.nppiPSNRBatchGetBufferHostSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiPSNRBatchGetBufferHostSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// 3 channel 8-bit unsigned WMSSSIM for a batch of image pairs for a single ROI.
        /// Provided oSizeROI will be used for all images passed in pSrc1BatchList and pSrc2BatchList
        /// arguments.API user must ensure that provided ROI (oSizeROI) does not go beyond the borders of any of provided images.
        /// </summary>
        /// <param name="oSizeROI"></param>
        /// <param name="pBatchSrc1">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pBatchSrc2">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pWMSSSIM">Device memory pointer to output array of the computed WMSSSIM for nBatchSize* sizeof(Npp32f)* 3 image pairs.</param>
        /// <param name="pDeviceBufferList">Device memory pointer to the list of NppiBufferDescriptor buffer descriptors specifying per image device
        /// memory buffer pointers and size as returned by at least one WMSSSIMBatchGetBufferHostSize call.</param>
        public static void WMSSSIMBatch(NppiSize oSizeROI, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc1, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc2, CudaDeviceVariable<float> pWMSSSIM,
                                        CudaDeviceVariable<NppiBufferDescriptor> pDeviceBufferList)
        {
            NppStatus status = NPPNativeMethods.NPPi.IQA.nppiWMSSSIMBatch_8u_C3R(pBatchSrc1.DevicePointer, pBatchSrc2.DevicePointer, pBatchSrc1.Size, oSizeROI, pWMSSSIM.DevicePointer,
                pDeviceBufferList.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWMSSSIMBatch_8u_C3R", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// 3 channel 8-bit unsigned WMSSSIM for a batch of image pairs with per-image ROI
        /// </summary>
        /// <param name="oMaxSizeROI">maximum ROI width and height of ALL images in the batch.</param>
        /// <param name="pBatchSrc1">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pBatchSrc2">device memory pointer to the list of device memory image descriptors, per image oSize must be initialized.</param>
        /// <param name="pWMSSSIM">Device memory pointer to output array of the computed WMSSSIM for nBatchSize* sizeof(Npp32f)* 3 image pairs.</param>
        /// <param name="pDeviceBufferList">Device memory pointer to the list of NppiBufferDescriptor buffer descriptors specifying per image device
        /// memory buffer pointers and size as returned by at least one WMSSSIMBatchGetBufferHostSize call.</param>
        public static void WMSSSIMBatchAdvanced(NppiSize oMaxSizeROI, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc1, CudaDeviceVariable<NppiImageDescriptor> pBatchSrc2, CudaDeviceVariable<float> pWMSSSIM,
                                        CudaDeviceVariable<NppiBufferDescriptor> pDeviceBufferList)
        {
            NppStatus status = NPPNativeMethods.NPPi.IQA.nppiWMSSSIMBatch_8u_C3R_Advanced(pBatchSrc1.DevicePointer, pBatchSrc2.DevicePointer, pBatchSrc1.Size, oMaxSizeROI, pWMSSSIM.DevicePointer,
                pDeviceBufferList.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWMSSSIMBatch_8u_C3R_Advanced", status));
            NPPException.CheckNppStatus(status, null);
        }

        /// <summary>
        /// Buffer size for \ref nppiWMSSSIMBatch_8u_C3R.
        /// </summary>
        public SizeT WMSSSIMBatchGetBufferHostSize()
        {
            SizeT ret = 0;
            status = NPPNativeMethods.NPPi.IQA.nppiWMSSSIMBatchGetBufferHostSize_8u_C3R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiWMSSSIMBatchGetBufferHostSize_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
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

        /// <summary>
        /// Buffer size (in bytes) for nppiCrossCorrFull_NormLevelAdvanced functions.
        /// </summary>
        /// <param name="oTplRoiSize">Region-of-Interest (ROI) size of the template image.</param>
        public SizeT CrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize oTplRoiSize)
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(_sizeRoi, oTplRoiSize, sizeof(float), _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        /// <summary>
        /// Buffer size (in bytes) for nppiCrossCorrSame_NormLevelAdvanced functions.
        /// </summary>
        /// <param name="oTplRoiSize">Region-of-Interest (ROI) size of the template image.</param>
        public SizeT CrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize oTplRoiSize)
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(_sizeRoi, oTplRoiSize, sizeof(float), _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        /// <summary>
        /// Buffer size (in bytes) for nppiCrossCorrValid_NormLevelAdvanced functions.
        /// </summary>
        /// <param name="oTplRoiSize">Region-of-Interest (ROI) size of the template image.</param>
        public SizeT CrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize oTplRoiSize)
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(_sizeRoi, oTplRoiSize, sizeof(float), _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize", status));
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
            status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianBorderGetBufferSize_8u_C3R(_sizeRoi, oMaskSize, ref bufferSize, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianBorderGetBufferSize_8u_C3R", status));
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
        public void Add(CudaDeviceVariable<byte> nConstant, NPPImage_8uC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_8u_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(CudaDeviceVariable<byte> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_8u_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_8u_C3IRSfs", status));
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
        public void Sub(CudaDeviceVariable<byte> nConstant, NPPImage_8uC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_8u_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(CudaDeviceVariable<byte> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_8u_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_8u_C3IRSfs", status));
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
        public void Mul(CudaDeviceVariable<byte> nConstant, NPPImage_8uC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_8u_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(CudaDeviceVariable<byte> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_8u_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_8u_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Multiply constant to image and scale by max bit width value
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void Mul(CudaDeviceVariable<byte> nConstant, NPPImage_8uC3 dest)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConstScale.nppiMulDeviceCScale_8u_C3R(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceCScale_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image and scale by max bit width value
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void Mul(CudaDeviceVariable<byte> nConstant)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConstScale.nppiMulDeviceCScale_8u_C3IR(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceCScale_8u_C3IR", status));
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
        public void Div(CudaDeviceVariable<byte> nConstant, NPPImage_8uC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_8u_C3RSfs(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_8u_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(CudaDeviceVariable<byte> nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_8u_C3IRSfs(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_8u_C3IRSfs", status));
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
        public void FilterMedianBorder(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, CudaDeviceVariable<byte> pBuffer, NppiRect filterArea = new NppiRect())
        {
            if (filterArea.Size == new NppiSize())
            {
                filterArea.Size = _sizeRoi;
            }
            status = NPPNativeMethods.NPPi.ImageMedianFilter.nppiFilterMedianBorder_8u_C3R(_devPtrRoi, _pitch, filterArea.Size, filterArea.Location, dest.DevicePointerRoi,
                                                    dest.Pitch, dest.SizeRoi, oMaskSize, oAnchor, pBuffer.DevicePointer, eBorderType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterMedianBorder_8u_C3R", status));
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
        public void FilterBoxBorderAdvanced(NPPImage_8uC3 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, CudaDeviceVariable<byte> pBuffer, NppiRect filterArea = new NppiRect())
        {
            if (filterArea.Size == new NppiSize())
            {
                filterArea.Size = _sizeRoi;
            }
            status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorderAdvanced_8u_C3R(_devPtrRoi, _pitch, filterArea.Size, filterArea.Location, dest.DevicePointerRoi,
                                                    dest.Pitch, dest.SizeRoi, oMaskSize, oAnchor, eBorderType, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorderAdvanced_8u_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// CrossCorrFull_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrFull_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevelAdvanced_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevelAdvanced_8u32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrSame_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrSame_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevelAdvanced_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevelAdvanced_8u32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrValid_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrValid_NormLevel(NPPImage_8uC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevelAdvanced_8u32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevelAdvanced_8u32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
#endif
        #endregion

        #region New in Cuda 13.0
        #region FilterCannyBorder
        /// <summary>
        /// Calculate scratch buffer size needed for the FilterCannyBorder function based on destination image SizeROI width and height.
        /// </summary>
        /// <returns></returns>
        public int FilterCannyBorderGetBufferSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.FilterCannyBorder.nppiFilterCannyBorderGetBufferSize(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterCannyBorderGetBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #endregion
        #endregion

    }
}
