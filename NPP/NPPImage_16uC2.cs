﻿// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
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


using System;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NPP
{
    /// <summary>
    /// 
    /// </summary>
    public partial class NPPImage_16uC2 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_16uC2(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 2;
            _isOwner = true;
            _typeSize = sizeof(ushort);
            _dataType = NppDataType.NPP_16U;
            _nppChannels = NppiChannels.NPP_CH_2;

            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_16u_C2(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_16u_C2", res, _pitch, _channels));

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
        public NPPImage_16uC2(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = sizeof(ushort);
            _dataType = NppDataType.NPP_16U;
            _nppChannels = NppiChannels.NPP_CH_2;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16uC2(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_16uC2(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_16uC2(NppiSize size)
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
        public NPPImage_16uC2(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16uC2(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_16uC2()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<VectorTypes.ushort2> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<VectorTypes.ushort2>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<VectorTypes.ushort2>(NPPImage_16uC2 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_16uC2(CudaPitchedDeviceVariable<VectorTypes.ushort2> img)
        {
            return img.ToNPPImage();
        }
        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set (array length = 2)</param>
        public void Set(ushort[] nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_16u_C2R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16u_C2R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        /// <summary>
        /// Image composition using image alpha values (0 - max channel pixel value).<para/>
        /// Also the function is called *AC1R, it is a two channel image with second channel as alpha.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nppAlphaOp">alpha compositing operation</param>
        public void AlphaComp(NPPImage_16uC2 src2, NPPImage_16uC2 dest, NppiAlphaOp nppAlphaOp)
        {
            status = NPPNativeMethods.NPPi.AlphaComp.nppiAlphaComp_16u_AC1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaComp_16u_AC1R", status));
            NPPException.CheckNppStatus(status, this);
        }


        #region MaxError
        /// <summary>
        /// image maximum error. User buffer is internally allocated and freed.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        public void MaxError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = MaxErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_16u_C2R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
        public void MaxError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = MaxErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_16u_C2R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaxError.
        /// </summary>
        /// <returns></returns>
        public int MaxErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_16u_C2R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_16u_C2R", status));
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
        public void AverageError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = AverageErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_16u_C2R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
        public void AverageError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = AverageErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_16u_C2R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageError.
        /// </summary>
        /// <returns></returns>
        public int AverageErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_16u_C2R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_16u_C2R", status));
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
        public void MaximumRelativeError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = MaximumRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_16u_C2R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
        public void MaximumRelativeError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = MaximumRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_16u_C2R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaximumRelativeError.
        /// </summary>
        /// <returns></returns>
        public int MaximumRelativeErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_16u_C2R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_16u_C2R", status));
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
        public void AverageRelativeError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = AverageRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_16u_C2R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
        public void AverageRelativeError(NPPImage_16uC2 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = AverageRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_16u_C2R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_16u_C2R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageRelativeError.
        /// </summary>
        /// <returns></returns>
        public int AverageRelativeErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_16u_C2R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_16u_C2R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #endregion


        #region ColorTwist
        /// <summary>
        /// An input color twist matrix with floating-point pixel values is applied
        /// within ROI.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
        public void ColorTwist(NPPImage_16uC2 dest, float[,] twistMatrix)
        {
            status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist32f_16u_C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, twistMatrix);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16u_C2R", status));
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
            status = NPPNativeMethods.NPPi.ColorProcessing.nppiColorTwist32f_16u_C2IR(_devPtrRoi, _pitch, _sizeRoi, aTwist);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16u_C2IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion
    }
}
