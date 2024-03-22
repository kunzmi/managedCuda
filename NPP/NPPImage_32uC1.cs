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


using System;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NPP
{
    /// <summary>
    /// 
    /// </summary>
    public partial class NPPImage_32uC1 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_32uC1(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 1;
            _isOwner = true;
            _typeSize = sizeof(uint);
            _dataType = NppDataType.NPP_32U;
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
        public NPPImage_32uC1(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = sizeof(uint);
            _dataType = NppDataType.NPP_32U;
            _nppChannels = NppiChannels.NPP_CH_1;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32uC1(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_32uC1(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_32uC1(NppiSize size)
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
        public NPPImage_32uC1(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_32uC1(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_32uC1()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<uint> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<uint>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<uint>(NPPImage_32uC1 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_32uC1(CudaPitchedDeviceVariable<uint> img)
        {
            return img.ToNPPImage();
        }
        #endregion

        #region Convert
        /// <summary>
        /// 32-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="roundMode">Round mode</param>
        /// <param name="scaleFactor">scaling factor</param>
        public void Convert(NPPImage_8uC1 dst, NppRoundMode roundMode, int scaleFactor)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32u8u_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32u8u_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit unsigned to 8-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="roundMode">Round mode</param>
        /// <param name="scaleFactor">scaling factor</param>
        public void Convert(NPPImage_8sC1 dst, NppRoundMode roundMode, int scaleFactor)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32u8s_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32u8s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="roundMode">Round mode</param>
        /// <param name="scaleFactor">scaling factor</param>
        public void Convert(NPPImage_16uC1 dst, NppRoundMode roundMode, int scaleFactor)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32u16u_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32u16u_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="roundMode">Round mode</param>
        /// <param name="scaleFactor">scaling factor</param>
        public void Convert(NPPImage_16sC1 dst, NppRoundMode roundMode, int scaleFactor)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32u16s_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32u16s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        /// <param name="roundMode">Round mode</param>
        /// <param name="scaleFactor">scaling factor</param>
        public void Convert(NPPImage_32sC1 dst, NppRoundMode roundMode, int scaleFactor)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32u32s_C1RSfs(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, roundMode, scaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32u32s_C1RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// 32-bit unsigned to 32-bit float conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32fC1 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_32u32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_32u32f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Alpha composition
        /// <summary>
        /// Image composition using constant alpha.
        /// </summary>
        /// <param name="alpha1">constant alpha for this image</param>
        /// <param name="src2">2nd source image</param>
        /// <param name="alpha2">constant alpha for src2</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nppAlphaOp">alpha compositing operation</param>
        public void AlphaComp(uint alpha1, NPPImage_32uC1 src2, ushort alpha2, NPPImage_32uC1 dest, NppiAlphaOp nppAlphaOp)
        {
            status = NPPNativeMethods.NPPi.AlphaCompConst.nppiAlphaCompC_32u_C1R(_devPtrRoi, _pitch, alpha1, src2.DevicePointerRoi, src2.Pitch, alpha2, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nppAlphaOp);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAlphaCompC_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region DotProduct
        /// <summary>
        /// Device scratch buffer size (in bytes) for nppiDotProd_32u64f_C1R.
        /// </summary>
        /// <returns></returns>
        public SizeT DotProdGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.DotProd.nppiDotProdGetBufferHostSize_32u64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProdGetBufferHostSize_32u64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// One-channel 32-bit unsigned image DotProd.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pDp">Pointer to the computed dot product of the two images. (1 * sizeof(double))</param>
        /// <param name="buffer">Allocated device memory with size of at <see cref="DotProdGetBufferHostSize()"/></param>
        public void DotProduct(NPPImage_32uC1 src2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = DotProdGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_32u64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_32u64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// One-channel 32-bit unsigned image DotProd. Buffer is internally allocated and freed.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pDp">Pointer to the computed dot product of the two images. (1 * sizeof(double))</param>
        public void DotProduct(NPPImage_32uC1 src2, CudaDeviceVariable<double> pDp)
        {
            SizeT bufferSize = DotProdGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

            status = NPPNativeMethods.NPPi.DotProd.nppiDotProd_32u64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pDp.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDotProd_32u64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set</param>
        public void Set(uint nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_32u_C1R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region MaxError
        /// <summary>
        /// image maximum error. User buffer is internally allocated and freed.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        public void MaxError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = MaxErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_32u_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
        public void MaxError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = MaxErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaxError.
        /// </summary>
        /// <returns></returns>
        public SizeT MaxErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_32u_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_32u_C1R", status));
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
        public void AverageError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = AverageErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32u_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
        public void AverageError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = AverageErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageError.
        /// </summary>
        /// <returns></returns>
        public SizeT AverageErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_32u_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_32u_C1R", status));
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
        public void MaximumRelativeError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = MaximumRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32u_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
        public void MaximumRelativeError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = MaximumRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaximumRelativeError.
        /// </summary>
        /// <returns></returns>
        public SizeT MaximumRelativeErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_32u_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_32u_C1R", status));
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
        public void AverageRelativeError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = AverageRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32u_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
        public void AverageRelativeError(NPPImage_32uC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = AverageRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_32u_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageRelativeError.
        /// </summary>
        /// <returns></returns>
        public SizeT AverageRelativeErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_32u_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #endregion

        #region new in Cuda 10.2


        /// <summary>
        /// Calculate scratch buffer size needed 1 channel 32-bit unsigned integer LabelMarkersUF function based on destination image oSizeROI width and height.
        /// </summary>
        /// <returns>Required buffer size in bytes.</returns>
        public int LabelMarkersUFGetBufferSize()
        {
            int ret = 0;
            status = NPPNativeMethods.NPPi.LabelMarkers.nppiLabelMarkersUFGetBufferSize_32u_C1R(_sizeRoi, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabelMarkersUFGetBufferSize_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
            return ret;
        }

        /// <summary>
        /// 1 channel 32-bit to 32-bit unsigned integer label markers image generation.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
        public void LabelMarkersUF(NPPImage_32uC1 dest, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
        {
            status = NPPNativeMethods.NPPi.LabelMarkers.nppiLabelMarkersUF_32u_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, eNorm, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabelMarkersUF_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        ///// <summary>
        ///// Calculate scratch buffer size needed for 1 channel 32-bit unsigned integer to 16-bit unsigned integer CompressMarkerLabels function based on the number returned in pNumber from a previous nppiLabelMarkers call.
        ///// </summary>
        ///// <param name="nStartingNumber">The value returned from a previous call to the nppiLabelMarkers_8u32u function.</param>
        ///// <returns>Required buffer size in bytes.</returns>
        //public int CompressMarkerLabelsGetBufferSize32u16u(int nStartingNumber)
        //{
        //	int ret = 0;
        //	status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressMarkerLabelsGetBufferSize_32u16u_C1R(nStartingNumber, ref ret);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressMarkerLabelsGetBufferSize_32u16u_C1R", status));
        //	NPPException.CheckNppStatus(status, this);
        //	return ret;
        //}

        ///// <summary>
        ///// 1 channel 32-bit unsigned integer to 16-bit unsigned integer connected region marker label renumbering with numbering sparseness elimination.
        ///// </summary>
        ///// <param name="dest">Destination-Image</param>
        ///// <param name="nStartingNumber">The value returned from a previous call to the nppiLabelMarkers_8u32u function.</param>
        ///// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding CompressMarkerLabelsGetBufferSize call.</param>
        ///// <returns>the maximum renumbered marker label ID will be returned.</returns>
        //public int CompressMarkerLabels(NPPImage_16uC1 dest, int nStartingNumber, CudaDeviceVariable<byte> pBuffer)
        //{
        //	int pNewNumber = 0;
        //	status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressMarkerLabels_32u16u_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nStartingNumber, ref pNewNumber, pBuffer.DevicePointer);
        //	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressMarkerLabels_32u16u_C1R", status));
        //	NPPException.CheckNppStatus(status, this);
        //	return pNewNumber;
        //}
        #endregion

        #region new in Cuda 11


        /// <summary>
        /// label markers image generation with fixed destination ROI applied to all images in the batch.
        /// </summary>
        /// <param name="pSrcBatchList">source_batch_images_pointer device memory pointer to the list of device memory source image descriptors, oSize element is ignored.</param>
        /// <param name="pDstBatchList">destination_batch_images_pointer device memory pointer to the list of device memory destination image descriptors, oSize element is ignored.</param>
        /// <param name="oSizeROI">Region-of-Interest (ROI).</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        public static void LabelMarkersUFBatch(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList,
                              NppiSize oSizeROI, NppiNorm eNorm)
        {
            NppStatus status = NPPNativeMethods.NPPi.LabelMarkers.nppiLabelMarkersUFBatch_32u_C1R(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oSizeROI, eNorm);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabelMarkersUFBatch_32u_C1R", status));
            NPPException.CheckNppStatus(status, pSrcBatchList);
        }

        /// <summary>
        /// label markers image generation with per image destination ROI.
        /// </summary>
        /// <param name="pSrcBatchList">source_batch_images_pointer device memory pointer to the list of device memory source image descriptors, oSize element is ignored.</param>
        /// <param name="pDstBatchList">destination_batch_images_pointer device memory pointer to the list of device memory destination image descriptors, oSize element is ignored.</param>
        /// <param name="oMaxSizeROI">maximum ROI width and height of ALL images in the batch.</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        public static void LabelMarkersUFBatch_Advanced(CudaDeviceVariable<NppiImageDescriptor> pSrcBatchList, CudaDeviceVariable<NppiImageDescriptor> pDstBatchList,
                              NppiSize oMaxSizeROI, NppiNorm eNorm)
        {
            NppStatus status = NPPNativeMethods.NPPi.LabelMarkers.nppiLabelMarkersUFBatch_32u_C1R_Advanced(pSrcBatchList.DevicePointer, pDstBatchList.DevicePointer, pSrcBatchList.Size, oMaxSizeROI, eNorm);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLabelMarkersUFBatch_32u_C1R_Advanced", status));
            NPPException.CheckNppStatus(status, pSrcBatchList);
        }

        /// <summary>
        /// Calculate scratch buffer size needed for 1 channel 32-bit unsigned integer CompressMarkerLabels function based on the number returned in pNumber from a previous nppiLabelMarkers call.
        /// </summary>
        /// <param name="nStartingNumber">The value returned from a previous call to the nppiLabelMarkers_32u function.</param>
        public static int CompressMarkerLabelsGetBufferSize(int nStartingNumber)
        {
            int ret = 0;
            NppStatus status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressMarkerLabelsGetBufferSize_32u_C1R(nStartingNumber, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressMarkerLabelsGetBufferSize_32u_C1R", status));
            NPPException.CheckNppStatus(status, null);
            return ret;
        }

        /// <summary>
        /// 1 channel 32-bit unsigned integer in place connected region marker label renumbering for output from nppiLabelMarkersUF functions only with numbering sparseness elimination.<para/>
        /// Note that the image in this function must be allocated with cudaMalloc() and NOT cudaMallocPitch(). <para/>
        /// Also the pitch MUST be set to oSizeROI.width * sizeof(Npp32u).  And the image pointer and oSizeROI values MUST match those used when nppiLabelMarkersUF was called.<para/>
        /// </summary>
        /// <param name="nStartingNumber">The value returned from a previous call to the nppiLabelMarkers_8u32u function.</param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding CompressMarkerLabelsGetBufferSize call.</param>
        /// <returns>the maximum renumbered marker label ID will be returned.</returns>
        public int CompressMarkerLabelsUF(int nStartingNumber, CudaDeviceVariable<byte> pBuffer)
        {
            int pNewNumber = 0;
            status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressMarkerLabelsUF_32u_C1IR(_devPtrRoi, _pitch, _sizeRoi, nStartingNumber, ref pNewNumber, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressMarkerLabelsUF_32u_C1IR", status));
            NPPException.CheckNppStatus(status, this);
            return pNewNumber;
        }

        #endregion

        #region New in Cuda 11.1


        /// <summary>
        /// in place flood fill.
        /// </summary>
        /// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
        /// <param name="nNewValue">Image pixel values to be used to replace matching pixels.</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
        public NppiConnectedRegion FloodFill(NppiPoint oSeed, uint nNewValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
        {
            NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
            status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFill_32u_C1IR(_devPtrRoi, _pitch, oSeed, nNewValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFill_32u_C1IR", status));
            NPPException.CheckNppStatus(status, this);
            return pConnectedRegion;
        }

        /// <summary>
        /// in place flood fill.
        /// </summary>
        /// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
        /// <param name="nNewValue">Image pixel values to be used to replace matching pixels.</param>
        /// <param name="nBoundaryValue">Image pixel values to be used for region boundary. </param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
        public NppiConnectedRegion FloodFill(NppiPoint oSeed, uint nNewValue, uint nBoundaryValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
        {
            NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
            status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillBoundary_32u_C1IR(_devPtrRoi, _pitch, oSeed, nNewValue, nBoundaryValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillBoundary_32u_C1IR", status));
            NPPException.CheckNppStatus(status, this);
            return pConnectedRegion;
        }

        /// <summary>
        /// in place flood fill.
        /// </summary>
        /// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
        /// <param name="nMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
        /// <param name="nMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
        /// <param name="nNewValue">Image pixel values to be used to replace matching pixels.</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
        public NppiConnectedRegion FloodFill(NppiPoint oSeed, uint nMin, uint nMax, uint nNewValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
        {
            NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
            status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillRange_32u_C1IR(_devPtrRoi, _pitch, oSeed, nMin, nMax, nNewValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillRange_32u_C1IR", status));
            NPPException.CheckNppStatus(status, this);
            return pConnectedRegion;
        }

        /// <summary>
        /// in place flood fill.
        /// </summary>
        /// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
        /// <param name="nMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
        /// <param name="nMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
        /// <param name="nNewValue">Image pixel values to be used to replace matching pixels.</param>
        /// <param name="nBoundaryValue">Image pixel values to be used for region boundary. </param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
        public NppiConnectedRegion FloodFill(NppiPoint oSeed, uint nMin, uint nMax, uint nNewValue, uint nBoundaryValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
        {
            NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
            status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillRangeBoundary_32u_C1IR(_devPtrRoi, _pitch, oSeed, nMin, nMax, nNewValue, nBoundaryValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillRangeBoundary_32u_C1IR", status));
            NPPException.CheckNppStatus(status, this);
            return pConnectedRegion;
        }





        /// <summary>
        /// in place flood fill.
        /// </summary>
        /// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
        /// <param name="nMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
        /// <param name="nMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
        /// <param name="nNewValue">Image pixel values to be used to replace matching pixels.</param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
        public NppiConnectedRegion FloodFillGradient(NppiPoint oSeed, uint nMin, uint nMax, uint nNewValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
        {
            NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
            status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillGradient_32u_C1IR(_devPtrRoi, _pitch, oSeed, nMin, nMax, nNewValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillGradient_32u_C1IR", status));
            NPPException.CheckNppStatus(status, this);
            return pConnectedRegion;
        }
        /// <summary>
        /// in place flood fill.
        /// </summary>
        /// <param name="oSeed">Image location of seed pixel value to be used for comparison.</param>
        /// <param name="nMin">Value of each element of tested pixel must be &gt;= the corresponding seed value - aMin value.</param>
        /// <param name="nMax">Valeu of each element of tested pixel must be &lt;= the corresponding seed value + aMax value.</param>
        /// <param name="nNewValue">Image pixel values to be used to replace matching pixels.</param>
        /// <param name="nBoundaryValue">Image pixel values to be used for region boundary. </param>
        /// <param name="eNorm">Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity. </param>
        /// <param name="pBuffer">Pointer to device memory scratch buffer at least as large as value returned by the corresponding LabelMarkersUFGetBufferSize call.</param>
        public NppiConnectedRegion FloodFillGradient(NppiPoint oSeed, uint nMin, uint nMax, uint nNewValue, uint nBoundaryValue, NppiNorm eNorm, CudaDeviceVariable<byte> pBuffer)
        {
            NppiConnectedRegion pConnectedRegion = new NppiConnectedRegion();
            status = NPPNativeMethods.NPPi.FloodFill.nppiFloodFillGradientBoundary_32u_C1IR(_devPtrRoi, _pitch, oSeed, nMin, nMax, nNewValue, nBoundaryValue, eNorm, _sizeRoi, ref pConnectedRegion, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFloodFillGradientBoundary_32u_C1IR", status));
            NPPException.CheckNppStatus(status, this);
            return pConnectedRegion;
        }




        #endregion

        #region New in Cuda 11.4
        /// <summary>
        /// Calculate the size of device memory needed for the CompressedMarkerLabelsGetInfoList function based on nMaxMarkerLabelID value 
        /// returned by previous call to CompressMarkerLabelsUF function.
        /// </summary>
        /// <param name="nMaxMarkerLabelID">value returned by previous call to CompressMarkerLabelsUF for this image.</param>
        public static uint CompressedMarkerLabelsUFGetInfoListSize(uint nMaxMarkerLabelID)
        {
            uint ret = 0;
            NppStatus status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(nMaxMarkerLabelID, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R", status));
            NPPException.CheckNppStatus(status, null);
            return ret;
        }

        /// <summary>
		/// Calculate the size of device memory needed for the CompressedMarkerLabelsUFContourGeometryLists function based on the final 
		/// value in the pContoursPixelStartingOffsetHost list 
		/// returned by previous call nppiCompressedMarkerLabelsUFInfo function.
        /// </summary>
		/// <param name="nMaxContourPixelGeometryInfoCount">the final value in the pContoursPixelStartingOffsetHost list.  </param>
        public static uint CompressedMarkerLabelsUFGetGeometryListsSize(uint nMaxContourPixelGeometryInfoCount)
        {
            uint ret = 0;
            NppStatus status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(nMaxContourPixelGeometryInfoCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R", status));
            NPPException.CheckNppStatus(status, null);
            return ret;
        }


#if ADD_MISSING_CTX
        /// <summary>
        /// 1 channel 32-bit uinteger connected region marker label renumbered from a previous call to nppiCompressMarkerLabelsUF or 
        /// nppiCmpressMarkerLabelsUFBatch functions to eliminate label ID sparseness.
        /// </summary>
        /// <param name="nMaxMarkerLabelID">the value of the maximum marker label ID returned by corresponding compress marker labels UF call. </param>
        /// <param name="pMarkerLabelsInfoList">pointer to device memory buffer at least as large as value returned by the corresponding CompressedMarkerLabelsGetInfoListSize call.</param>
        /// <param name="pContoursImage">optional output image containing contours (boundaries) around each uniquely labeled connected pixel region, set to NULL if not needed. </param>
        /// <param name="pContoursDirectionImage">optional output image containing per contour pixel direction info around each uniquely labeled connected pixel region, set to NULL if not needed. </param>
        /// <param name="pContoursTotalsInfoHost">unique per call optional host memory pointer to NppiContourTotalsInfo structure in host memory, MUST be set if pContoursDirectionImage is set. </param>
        /// <param name="pContoursPixelCountsListDev">unique per call optional device memory pointer to array of nMaxMarkerLabelID uintegers in host memory, MUST be set if pContoursDirectionImage is set. </param>
        /// <param name="pContoursPixelCountsListHost">unique per call optional host memory pointer to array of nMaxMarkerLabelID uintegers in host memory, MUST be set if pContoursDirectionImage is set. </param>
        /// <param name="pContoursPixelStartingOffsetHost">unique per call optional host memory pointer to array of uintegers returned by this call representing the starting offset index of each contour found during geometry list generation </param>
        public void CompressedMarkerLabelsUFInfo(
            uint nMaxMarkerLabelID, CudaDeviceVariable<NppiCompressedMarkerLabelsInfo> pMarkerLabelsInfoList, NPPImage_8uC1 pContoursImage,
            CudaPitchedDeviceVariable<NppiContourPixelDirectionInfo> pContoursDirectionImage, NppiContourTotalsInfo[] pContoursTotalsInfoHost,
            CudaDeviceVariable<uint> pContoursPixelCountsListDev, uint[] pContoursPixelCountsListHost, uint[] pContoursPixelStartingOffsetHost)
        {
            CUdeviceptr ptrMarkerLabelsInfoList = new CUdeviceptr();
            CUdeviceptr ptrContoursImage = new CUdeviceptr();
            CUdeviceptr ptrContoursDirectionImage = new CUdeviceptr();
            CUdeviceptr ptrContoursPixelCountsListDev = new CUdeviceptr();
            int pitchContoursImage = 0;
            int pitchContoursDirectionImage = 0;
            if (pMarkerLabelsInfoList != null)
            {
                ptrMarkerLabelsInfoList = pMarkerLabelsInfoList.DevicePointer;
            }
            if (pContoursImage != null)
            {
                ptrContoursImage = pContoursImage.DevicePointerRoi;
                pitchContoursImage = pContoursImage.Pitch;
            }
            if (pContoursDirectionImage != null)
            {
                ptrContoursDirectionImage = pContoursDirectionImage.DevicePointer;
                pitchContoursDirectionImage = pContoursDirectionImage.Pitch;
            }
            if (pContoursPixelCountsListDev != null)
            {
                ptrContoursPixelCountsListDev = pContoursPixelCountsListDev.DevicePointer;
            }


            status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressedMarkerLabelsUFInfo_32u_C1R(_devPtrRoi, _pitch, _sizeRoi, nMaxMarkerLabelID, ptrMarkerLabelsInfoList,
                ptrContoursImage, pitchContoursImage, ptrContoursDirectionImage, pitchContoursDirectionImage, pContoursTotalsInfoHost, ptrContoursPixelCountsListDev,
                pContoursPixelCountsListHost, pContoursPixelStartingOffsetHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressedMarkerLabelsUFInfo_32u_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 1 channel connected region contours image to generate contours geometry info list in host memory. 
        ///  <para/>
        /// Note that ALL input and output data for the function MUST be in device memory except where noted otherwise. 
        /// Also nFirstContourID and nLastContourID allow only a portion of the contour geometry lists in the image to be output. 
        ///  <para/>
        /// Note that the geometry list for each contour will begin at pContoursGeometryListsHost[pContoursPixelStartingOffsetHost[nContourID]		/// sizeof(NppiContourPixelGeometryInfo). 
        ///  <para/>
        /// Note that due to the nature of some imput images contour ID 0 can sometimes contain ALL contours in the image which 
        /// can significantly increase the time taken to output the geometry lists.  In these cases setting nFirstContourGeometryListID to >= 1 
        /// significantly speed up geometry list output performance and all individual contours will still be output. 
        /// </summary>
        /// <param name="pMarkerLabelsInfoListDev">pointer to device memory buffer which contains the output returned by the corresponding nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx call.</param>
        /// <param name="pMarkerLabelsInfoListHost">pointer to host memory buffer which will be output by this function with additional information added.</param>
        /// <param name="pContoursDirectionImageDev">Source-Image Pointer to output image in device memory containing per contour pixel direction info around each uniquely labeled connected pixel region returned by corresponding nppiCompressedMarkerLabelsUFInfo call. </param>
        /// <param name="pContoursPixelGeometryListsHost">pointer to host memory buffer allocated to be at least as big as size returned by corresponding nppiCompressedMarkerLabelsUFGetGeometryListsSize call. </param>
        /// <param name="pContoursPixelsFoundListHost">host memory pointer to array of nMaxMarkerLabelID uintegers returned by this call representing the number of contour pixels found during geometry list generation. </param>
        /// <param name="pContoursPixelsStartingOffsetHost">host memory pointer to array of uintegers returned by this call representing the starting offset index of each contour found during geometry list generation. </param>
        /// <param name="nMaxMarkerLabelID">the value of the maximum marker label ID returned by corresponding compress marker labels UF call. </param>
        /// <param name="nFirstContourGeometryListID">the ID of the first contour geometry list to output. </param>
        /// <param name="nLastContourGeometryListID">the ID of the last contour geometry list to output.  </param>
        /// <param name="pContoursPixelGeometryListsDev">pointer to device memory buffer allocated to be at least as big as size returned by corresponding nppiCompressedMarkerLabelsUFGetGeometryListsSize call. </param>
        /// <param name="pContoursGeometryImageHost">optional pointer to host memory image of at least oSizeROI.width * sizeof(Npp8u) * oSizeROI.height bytes, NULL if not needed.</param>
        /// <param name="nContoursGeometryImageStep">geometry image line step. </param>
        /// <param name="pContoursPixelCountsListDev">device memory pointer to array of nMaxMarkerLabelID unsigned integers returned by previous call to nppiCompressedMarkerLabelsUFContoursPixelGeometryLists_C1R_Ctx.</param>
        /// <param name="pContoursPixelsFoundListDev">device memory pointer to array of nMaxMarkerLabelID unsigned integers returned by previous call to nppiCompressedMarkerLabelsUFContoursPixelGeometryLists_C1R_Ctx.</param>
        /// <param name="pContoursPixelsStartingOffsetDev">device memory pointer to array of unsigned integers returned by this call representing the starting offset index of each contour found during geometry list generation. </param>
        /// <param name="nTotalImagePixelContourCount">the total number of contour pixels in the image returned by nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx() call. </param>
        /// <param name="pContoursBlockSegmentListDev">device memory pointer to array of NppiContourBlockSegment objects, contents will be initialized by NPP. </param>
        /// <param name="pContoursBlockSegmentListHost">host memory pointer to array of NppiContourBlockSegment objects, contents will be intialized by NPP.</param>
        /// <param name="bOutputInCounterclockwiseOrder">if nonzero then output geometry list for each contour in counterclockwise order, otherwise clockwise.</param>
        public void CompressedMarkerLabelsUFContoursGenerateGeometryLists(CudaDeviceVariable<NppiCompressedMarkerLabelsInfo> pMarkerLabelsInfoListDev, NppiCompressedMarkerLabelsInfo[] pMarkerLabelsInfoListHost,
            CudaPitchedDeviceVariable<NppiContourPixelDirectionInfo> pContoursDirectionImageDev, CudaDeviceVariable<NppiContourPixelGeometryInfo> pContoursPixelGeometryListsDev, NppiContourPixelGeometryInfo[] pContoursPixelGeometryListsHost,
                                                              byte[] pContoursGeometryImageHost, int nContoursGeometryImageStep, CudaDeviceVariable<uint> pContoursPixelCountsListDev, CudaDeviceVariable<uint> pContoursPixelsFoundListDev,
                                                              uint[] pContoursPixelsFoundListHost, CudaDeviceVariable<uint> pContoursPixelsStartingOffsetDev, uint[] pContoursPixelsStartingOffsetHost, uint nTotalImagePixelContourCount,
                                                              uint nMaxMarkerLabelID, uint nFirstContourGeometryListID, uint nLastContourGeometryListID, CudaDeviceVariable<NppiContourBlockSegment> pContoursBlockSegmentListDev,
                                                              NppiContourBlockSegment[] pContoursBlockSegmentListHost, uint bOutputInCounterclockwiseOrder)
        {
            status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R(pMarkerLabelsInfoListDev.DevicePointer, pMarkerLabelsInfoListHost,
                pContoursDirectionImageDev.DevicePointer, pContoursDirectionImageDev.Pitch, pContoursPixelGeometryListsDev.DevicePointer, pContoursPixelGeometryListsHost, pContoursGeometryImageHost,
                nContoursGeometryImageStep, pContoursPixelCountsListDev.DevicePointer, pContoursPixelsFoundListDev.DevicePointer, pContoursPixelsFoundListHost, pContoursPixelsStartingOffsetDev.DevicePointer,
                pContoursPixelsStartingOffsetHost, nTotalImagePixelContourCount, nMaxMarkerLabelID, nFirstContourGeometryListID, nLastContourGeometryListID, pContoursBlockSegmentListDev.DevicePointer,
                pContoursBlockSegmentListHost, bOutputInCounterclockwiseOrder, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
#endif
        #endregion

        #region New in Cuda 12

        /// <summary>
        /// Calculate the size of memory needed for the geometry list generation function. 
        /// </summary>
        /// <param name="pContoursPixelCountsListHost">Host memory pointer to list returned by CompressedMarkerLabelsUFInfo_32u call.</param>
        /// <param name="nTotalImagePixelContourCount">the total label count returned by the nppiCompressMarkerLabelsUF function. </param>
        /// <param name="nCompressedLabelCount">the total number of contour pixels in the image from the NppiContourTotalsInfo object returned from nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx() call.</param>
        /// <param name="nFirstContourGeometryListID">the ID of the first contour geometry list to process. </param>
        /// <param name="nLastContourGeometryListID">the ID of the last contour geometry list to process, last ID MUST be greater than first ID.</param>
        public static uint CompressedMarkerLabelsUFGetContoursBlockSegmentListSize(uint[] pContoursPixelCountsListHost, uint nTotalImagePixelContourCount, uint nCompressedLabelCount,
                                                                    uint nFirstContourGeometryListID, uint nLastContourGeometryListID)
        {
            uint ret = 0;
            NppStatus status = NPPNativeMethods.NPPi.LabelMarkers.nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(pContoursPixelCountsListHost, nTotalImagePixelContourCount,
                                                                        nCompressedLabelCount, nFirstContourGeometryListID, nLastContourGeometryListID, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R", status));
            NPPException.CheckNppStatus(status, null);
            return ret;
        }

        #endregion
    }
}
