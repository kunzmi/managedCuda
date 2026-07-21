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
	public partial class NPPImage_32fC2 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_32fC2(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 2;
            _isOwner = true;
            _typeSize = sizeof(float);
            _dataType = NppDataType.NPP_32F;
            _nppChannels = NppiChannels.NPP_CH_2;

            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_32f_C2(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_32f_C2", res, _pitch, _channels));

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
        public NPPImage_32fC2(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
        {
            _devPtr = devPtr;
            _devPtrRoi = _devPtr;
            _sizeOriginal.width = width;
            _sizeOriginal.height = height;
            _sizeRoi.width = width;
            _sizeRoi.height = height;
            _pitch = pitch;
            _channels = 2;
            _isOwner = isOwner;
            _typeSize = sizeof(float);
            _dataType = NppDataType.NPP_32F;
            _nppChannels = NppiChannels.NPP_CH_2;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32fC2(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_32fC2(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_32fC2(NppiSize size)
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
        public NPPImage_32fC2(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32fC2(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_32fC2()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<VectorTypes.float2> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<VectorTypes.float2>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<VectorTypes.float2>(NPPImage_32fC2 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_32fC2(CudaPitchedDeviceVariable<VectorTypes.float2> img)
        {
            return img.ToNPPImage();
        }
        #endregion

        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="channel">Channel number. This number is added to the dst pointer</param>
        public void Copy(NPPImage_32fC1 dst, int channel)
        {
            if (channel < 0 | channel >= _channels) throw new ArgumentOutOfRangeException("channel", "channel must be in range [0..1].");
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_32f_C2C1R(_devPtrRoi + channel * _typeSize, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32f_C2C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nppAlphaOp">alpha compositing operation</param>
        public void AlphaComp(NPPImage_32fC2 src2, NPPImage_32fC2 dest, NppiAlphaOp nppAlphaOp)
        {
            status = NPPNativeMethods.NPPi.AlphaComp.nppiAlphaComp_32f_AC1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaComp_32f_AC1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set</param>
        public void Set(float[] nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32f_C2R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32f_C2R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

    }
}
