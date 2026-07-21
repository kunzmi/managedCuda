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
    public partial class NPPImage_8sC3 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_8sC3(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 3;
            _isOwner = true;
            _typeSize = sizeof(byte);
            _dataType = NppDataType.NPP_8S;
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
        public NPPImage_8sC3(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _dataType = NppDataType.NPP_8S;
            _nppChannels = NppiChannels.NPP_CH_3;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_8sC3(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_8sC3(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_8sC3(NppiSize size)
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
        public NPPImage_8sC3(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_8sC3(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_8sC3()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<VectorTypes.char3> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<VectorTypes.char3>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<VectorTypes.char3>(NPPImage_8sC3 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_8sC3(CudaPitchedDeviceVariable<VectorTypes.char3> img)
        {
            return img.ToNPPImage();
        }
        #endregion

        #region Convert
        /// <summary>
        /// 8-bit signed to 32-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32sC3 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_8s32s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8s32s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 8-bit signed to 32-bit floating point conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32fC3 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_8s32f_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_8s32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Copy
        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Copy(NPPImage_8sC3 dst)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_8s_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_8s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set</param>
        public void Set(sbyte[] nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_8s_C3R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_8s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }

        #endregion

        #region Logical
        /// <summary>
        /// image bit shift by constant (right).
        /// </summary>
        /// <param name="nConstant">Constant</param>
        /// <param name="dest">Destination image</param>
        public void RShiftC(uint[] nConstant, NPPImage_8sC3 dest)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_8s_C3R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_8s_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// image bit shift by constant (right), inplace.
        /// </summary>
        /// <param name="nConstant">Constant</param>
        public void RShiftC(uint[] nConstant)
        {
            status = NPPNativeMethods.NPPi.RightShiftConst.nppiRShiftC_8s_C3IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiRShiftC_8s_C3IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        //new in Cuda 5.5

        #region New in Cuda 12.0

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
        /// CrossCorrFull_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrFull_NormLevel(NPPImage_8sC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevelAdvanced_8s32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevelAdvanced_8s32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrSame_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrSame_NormLevel(NPPImage_8sC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevelAdvanced_8s32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevelAdvanced_8s32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrValid_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrValid_NormLevel(NPPImage_8sC3 tpl, NPPImage_32fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevelAdvanced_8s32f_C3R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevelAdvanced_8s32f_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
#endif
        #endregion
    }
}
