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


using ManagedCuda.BasicTypes;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ManagedCuda.NPP
{
    /// <summary>
    /// 
    /// </summary>
    public partial class NPPImage_16scC3 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_16scC3(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 3;
            _isOwner = true;
            _typeSize = Marshal.SizeOf(typeof(Npp16sc));

            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_16sc_C3(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_16sc_C3", res, _pitch, _channels));

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
        public NPPImage_16scC3(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = Marshal.SizeOf(typeof(Npp16sc));
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16scC3(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_16scC3(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_16scC3(NppiSize size)
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
        public NPPImage_16scC3(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16scC3(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_16scC3()
        {
            Dispose(false);
        }
        #endregion

        #region Copy
        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Copy(NPPImage_16scC3 dst)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16sc_C3R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16sc_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set</param>
        public void Set(Npp16sc[] nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_16sc_C3R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16sc_C3R", status));
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
        public void Add(NPPImage_16scC3 src2, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_16sc_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(NPPImage_16scC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_16sc_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16sc_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(Npp16sc[] nConstant, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16sc_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Add(Npp16sc[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16sc_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16sc_C3IRSfs", status));
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
        public void Sub(NPPImage_16scC3 src2, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_16sc_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(NPPImage_16scC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_16sc_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16sc_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(Npp16sc[] nConstant, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16sc_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Sub(Npp16sc[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16sc_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16sc_C3IRSfs", status));
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
        public void Mul(NPPImage_16scC3 src2, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_16sc_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(NPPImage_16scC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_16sc_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16sc_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(Npp16sc[] nConstant, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16sc_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Mul(Npp16sc[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16sc_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16sc_C3IRSfs", status));
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
        public void Div(NPPImage_16scC3 src2, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_16sc_C3RSfs(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(NPPImage_16scC3 src2, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_16sc_C3IRSfs(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16sc_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(Npp16sc[] nConstant, NPPImage_16scC3 dest, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16sc_C3RSfs(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16sc_C3RSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="nScaleFactor">scaling factor</param>
        public void Div(Npp16sc[] nConstant, int nScaleFactor)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16sc_C3IRSfs(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16sc_C3IRSfs", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region MaxError
        /// <summary>
        /// image maximum error. User buffer is internally allocated and freed.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        public void MaxError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = MaxErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_16sc_C3R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
        public void MaxError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = MaxErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_16sc_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaxError.
        /// </summary>
        /// <returns></returns>
        public SizeT MaxErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_16sc_C3R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_16sc_C3R", status));
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
        public void AverageError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = AverageErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_16sc_C3R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
        public void AverageError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = AverageErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_16sc_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageError.
        /// </summary>
        /// <returns></returns>
        public SizeT AverageErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_16sc_C3R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_16sc_C3R", status));
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
        public void MaximumRelativeError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = MaximumRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_16sc_C3R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
        public void MaximumRelativeError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = MaximumRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_16sc_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaximumRelativeError.
        /// </summary>
        /// <returns></returns>
        public SizeT MaximumRelativeErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_16sc_C3R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_16sc_C3R", status));
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
        public void AverageRelativeError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError)
        {
            SizeT bufferSize = AverageRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_16sc_C3R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
        public void AverageRelativeError(NPPImage_16scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            SizeT bufferSize = AverageRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_16sc_C3R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_16sc_C3R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageRelativeError.
        /// </summary>
        /// <returns></returns>
        public SizeT AverageRelativeErrorGetBufferHostSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_16sc_C3R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_16sc_C3R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #endregion
    }
}
