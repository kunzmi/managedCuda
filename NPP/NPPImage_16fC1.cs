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
    public partial class NPPImage_16fC1 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_16fC1(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 1;
            _isOwner = true;
            _typeSize = sizeof(ushort);
            _dataType = NppDataType.NPP_16F;
            _nppChannels = NppiChannels.NPP_CH_1;

            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_16u_C1(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_16u_C1", res, _pitch, _channels));

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
        public NPPImage_16fC1(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = sizeof(ushort);
            _dataType = NppDataType.NPP_16F;
            _nppChannels = NppiChannels.NPP_CH_1;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16fC1(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_16fC1(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_16fC1(NppiSize size)
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
        public NPPImage_16fC1(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16fC1(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_16fC1()
        {
            Dispose(false);
        }
        #endregion

        #region Add
        /// <summary>
        /// Image addition.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Add(NPPImage_16fC1 src2, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_16f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image addition.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Add(NPPImage_16fC1 src2)
        {
            status = NPPNativeMethods.NPPi.Add.nppiAdd_16f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Add constant to image.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="dest">Destination image</param>
        public void Add(float nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        public void Add(float nConstant)
        {
            status = NPPNativeMethods.NPPi.AddConst.nppiAddC_16f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Sub
        /// <summary>
        /// Image subtraction.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Sub(NPPImage_16fC1 src2, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_16f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image subtraction.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Sub(NPPImage_16fC1 src2)
        {
            status = NPPNativeMethods.NPPi.Sub.nppiSub_16f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Subtract constant to image.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="dest">Destination image</param>
        public void Sub(float nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        public void Sub(float nConstant)
        {
            status = NPPNativeMethods.NPPi.SubConst.nppiSubC_16f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Mul
        /// <summary>
        /// Image multiplication.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Mul(NPPImage_16fC1 src2, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_16f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image multiplication.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Mul(NPPImage_16fC1 src2)
        {
            status = NPPNativeMethods.NPPi.Mul.nppiMul_16f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Multiply constant to image.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void Mul(float nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void Mul(float nConstant)
        {
            status = NPPNativeMethods.NPPi.MulConst.nppiMulC_16f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Div
        /// <summary>
        /// Image division.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void Div(NPPImage_16fC1 src2, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_16f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// In place image division.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        public void Div(NPPImage_16fC1 src2)
        {
            status = NPPNativeMethods.NPPi.Div.nppiDiv_16f_C1IR(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Divide constant to image.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void Div(float nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16f_C1R(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void Div(float nConstant)
        {
            status = NPPNativeMethods.NPPi.DivConst.nppiDivC_16f_C1IR(nConstant, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Abs
        /// <summary>
        /// Image absolute value.
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Abs(NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Abs.nppiAbs_16f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Image absolute value. In place.
        /// </summary>
        public void Abs()
        {
            status = NPPNativeMethods.NPPi.Abs.nppiAbs_16f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbs_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region AbsDiff
        /// <summary>
        /// Absolute difference of this minus src2.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void AbsDiff(NPPImage_16fC1 src2, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.AbsDiff.nppiAbsDiff_16f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAbsDiff_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region AddProduct
        /// <summary>
        /// Image product added to in place floating point destination image.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="dest">Destination image</param>
        public void AddProduct(NPPImage_16fC1 src2, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.AddProduct.nppiAddProduct_16f_C1IR(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddProduct_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Ln
        /// <summary>
        /// Natural logarithm.
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Ln(NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Ln.nppiLn_16f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Natural logarithm.
        /// </summary>
        public void Ln()
        {
            status = NPPNativeMethods.NPPi.Ln.nppiLn_16f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiLn_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Sqr
        /// <summary>
        /// Image squared.
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Sqr(NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Inplace image squared.
        /// </summary>
        public void Sqr()
        {
            status = NPPNativeMethods.NPPi.Sqr.nppiSqr_16f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqr_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Sqrt
        /// <summary>
        /// Image square root.
        /// </summary>
        /// <param name="dest">Destination image</param>
        public void Sqrt(NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16f_C1R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// Inplace image square root.
        /// </summary>
        public void Sqrt()
        {
            status = NPPNativeMethods.NPPi.Sqrt.nppiSqrt_16f_C1IR(_devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSqrt_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Copy
        /// <summary>
        /// Image copy.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Copy(NPPImage_16fC1 dst)
        {
            status = NPPNativeMethods.NPPi.MemCopy.nppiCopy_16f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Set
        /// <summary>
        /// Set pixel values to nValue.
        /// </summary>
        /// <param name="nValue">Value to be set</param>
        public void Set(float nValue)
        {
            status = NPPNativeMethods.NPPi.MemSet.nppiSet_16f_C1R(nValue, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Convert
        /// <summary>
        /// 16-bit floating point to 32-bit conversion.
        /// </summary>
        /// <param name="dst">Destination image</param>
        public void Convert(NPPImage_32fC1 dst)
        {
            status = NPPNativeMethods.NPPi.BitDepthConversion.nppiConvert_16f32f_C1R(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiConvert_16f32f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region New in Cuda 12.0
#if ADD_MISSING_CTX
        #region Add
        /// <summary>
        /// Add constant to image.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        /// <param name="dest">Destination image</param>
        public void Add(CudaDeviceVariable<float> nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_16f_C1R(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Add constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to add</param>
        public void Add(CudaDeviceVariable<float> nConstant)
        {
            status = NPPNativeMethods.NPPi.AddDeviceConst.nppiAddDeviceC_16f_C1IR(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddDeviceC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion
        #region Sub

        /// <summary>
        /// Subtract constant to image.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        /// <param name="dest">Destination image</param>
        public void Sub(CudaDeviceVariable<float> nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_16f_C1R(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Subtract constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value to subtract</param>
        public void Sub(CudaDeviceVariable<float> nConstant)
        {
            status = NPPNativeMethods.NPPi.SubDeviceConst.nppiSubDeviceC_16f_C1IR(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubDeviceC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Mul

        /// <summary>
        /// Multiply constant to image.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void Mul(CudaDeviceVariable<float> nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_16f_C1R(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Multiply constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void Mul(CudaDeviceVariable<float> nConstant)
        {
            status = NPPNativeMethods.NPPi.MulDeviceConst.nppiMulDeviceC_16f_C1IR(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulDeviceC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion

        #region Div

        /// <summary>
        /// Divide constant to image.
        /// </summary>
        /// <param name="nConstant">Value</param>
        /// <param name="dest">Destination image</param>
        public void Div(CudaDeviceVariable<float> nConstant, NPPImage_16fC1 dest)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_16f_C1R(_devPtrRoi, _pitch, nConstant.DevicePointer, dest.DevicePointerRoi, dest.Pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_16f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Divide constant to image. Inplace.
        /// </summary>
        /// <param name="nConstant">Value</param>
        public void Div(CudaDeviceVariable<float> nConstant)
        {
            status = NPPNativeMethods.NPPi.DivDeviceConst.nppiDivDeviceC_16f_C1IR(nConstant.DevicePointer, _devPtrRoi, _pitch, _sizeRoi);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivDeviceC_16f_C1IR", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion
#endif
        #endregion
    }
}
