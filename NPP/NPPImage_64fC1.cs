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
    public partial class NPPImage_64fC1 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_64fC1(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 1;
            _isOwner = true;
            _typeSize = sizeof(double);
            _dataType = NppDataType.NPP_64F;
            _nppChannels = NppiChannels.NPP_CH_1;

            //use 32fc as allocation type as NPP doesn't provide a 64f and 32fc has the same constraints...
            _devPtr = NPPNativeMethods.NPPi.MemAlloc.nppiMalloc_32fc_C1(nWidthPixels, nHeightPixels, ref _pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}, Number of color channels: {4}", DateTime.Now, "nppiMalloc_32fc_C1", res, _pitch, _channels));

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
        public NPPImage_64fC1(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _typeSize = sizeof(double);
            _dataType = NppDataType.NPP_64F;
            _nppChannels = NppiChannels.NPP_CH_1;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_64fC1(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_64fC1(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_64fC1(NppiSize size)
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
        public NPPImage_64fC1(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_64fC1(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_64fC1()
        {
            Dispose(false);
        }
        #endregion

        #region Converter operators

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        public CudaPitchedDeviceVariable<double> ToCudaPitchedDeviceVariable()
        {
            return new CudaPitchedDeviceVariable<double>(_devPtr, _sizeOriginal.width, _sizeOriginal.height, _pitch);
        }

        /// <summary>
        /// Converts a NPPImage to a CudaPitchedDeviceVariable
        /// </summary>
        /// <param name="img">NPPImage</param>
        /// <returns>CudaPitchedDeviceVariable with the same device pointer and size of NPPImage without ROI information</returns>
        public static implicit operator CudaPitchedDeviceVariable<double>(NPPImage_64fC1 img)
        {
            return img.ToCudaPitchedDeviceVariable();
        }

        /// <summary>
        /// Converts a CudaPitchedDeviceVariable to a NPPImage 
        /// </summary>
        /// <param name="img">CudaPitchedDeviceVariable</param>
        /// <returns>NPPImage with the same device pointer and size of CudaPitchedDeviceVariable with ROI set to full image</returns>
        public static implicit operator NPPImage_64fC1(CudaPitchedDeviceVariable<double> img)
        {
            return img.ToNPPImage();
        }
        #endregion

        #region MaxError
        /// <summary>
        /// image maximum error. User buffer is internally allocated and freed.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        public void MaxError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = MaxErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
        public void MaxError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = MaxErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaxError.
        /// </summary>
        /// <returns></returns>
        public int MaxErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_64f_C1R", status));
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
        public void AverageError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = AverageErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
        public void AverageError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = AverageErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageError.nppiAverageError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageError.
        /// </summary>
        /// <returns></returns>
        public int AverageErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_64f_C1R", status));
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
        public void MaximumRelativeError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = MaximumRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
        public void MaximumRelativeError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = MaximumRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaximumRelativeError.
        /// </summary>
        /// <returns></returns>
        public int MaximumRelativeErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_64f_C1R", status));
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
        public void AverageRelativeError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError)
        {
            int bufferSize = AverageRelativeErrorGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
        public void AverageRelativeError(NPPImage_64fC1 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = AverageRelativeErrorGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeError_64f_C1R(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageRelativeError.
        /// </summary>
        /// <returns></returns>
        public int AverageRelativeErrorGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #endregion


        #region new in Cuda 12.0

        /// <summary>
        /// Scratch-buffer size for SignedDistanceTransformPBA 64 bit floating point output.
        /// </summary>
        /// <returns></returns>
        public SizeT SignedDistanceTransformPBAGet64BufferSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiSignedDistanceTransformPBAGet64fBufferSize(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSignedDistanceTransformPBAGet64fBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        /// <summary>
        /// Scratch-buffer size for DistanceTransformPBA.
        /// </summary>
        /// <returns></returns>
        public SizeT DistanceTransformPBAGetBufferSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiDistanceTransformPBAGetBufferSize(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDistanceTransformPBAGetBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        /// <summary>
        /// Calculate scratch buffer size needed for the DistanceTransformPBA function based antialiasing on destination image SizeROI width and height.
        /// </summary>
        /// <returns></returns>
        public SizeT DistanceTransformPBAGetAntialiasingBufferSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiDistanceTransformPBAGetAntialiasingBufferSize(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDistanceTransformPBAGetAntialiasingBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        /// <summary>
        /// Calculate scratch buffer size needed for the SignedDistanceTransformPBA function based antialiasing on destination image SizeROI width and height.
        /// </summary>
        /// <returns></returns>
        public SizeT SignedDistanceTransformPBAGetAntialiasingBufferSize()
        {
            SizeT bufferSize = 0;
            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiSignedDistanceTransformPBAGetAntialiasingBufferSize(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSignedDistanceTransformPBAGetAntialiasingBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// Returns the required size of host memory buffer needed by 64 bit nppiFilterBoxBorderAdvanced functions.
        /// </summary>
        public int FilterBoxBorderAdvancedGetDeviceBufferSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(_sizeRoi, _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// Buffer size (in bytes) for nppiCrossCorrFull_NormLevelAdvanced functions.
        /// </summary>
        /// <param name="oTplRoiSize">Region-of-Interest (ROI) size of the template image.</param>
        public int CrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize oTplRoiSize)
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(_sizeRoi, oTplRoiSize, sizeof(double), _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        /// <summary>
        /// Buffer size (in bytes) for nppiCrossCorrSame_NormLevelAdvanced functions.
        /// </summary>
        /// <param name="oTplRoiSize">Region-of-Interest (ROI) size of the template image.</param>
        public int CrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize oTplRoiSize)
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(_sizeRoi, oTplRoiSize, sizeof(double), _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        /// <summary>
        /// Buffer size (in bytes) for nppiCrossCorrValid_NormLevelAdvanced functions.
        /// </summary>
        /// <param name="oTplRoiSize">Region-of-Interest (ROI) size of the template image.</param>
        public int CrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize oTplRoiSize)
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(_sizeRoi, oTplRoiSize, sizeof(double), _channels, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
#if ADD_MISSING_CTX

        /// <summary>
        /// 1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram 
        /// and 64-bit floating point transform with optional sub-pixel shifts.
        /// <para/>
        /// For this particular version of the function acceptable input pixel intensities are less than or equal to 0.0f for those fully outside of connected 
        /// pixel regions, intensities with fractional parts between 0.0f and 1.0f representing the percentage of connected pixel region sub-pixel coverage within a 
        /// particular pixel (region contour), and intensities greater than or equal to 1.0f for pixels that are fully contained within closed connected pixel regions. 
        /// This function executes in two passes, the first pass prioritizes pixels outside of closed regions, the second pass 
        /// prioritizes pixels within closed regions.  The two passes are then merged on output. The function assumes that fully 
        /// covered pixels have centers located at sub-pixel locations of .5,.5. In general, object exterior distances are output as negative 
        /// numbers progressing to positive and object interior distances are output as positive numbers progressing to negative.
        /// </summary>
        /// <param name="nCutoffValue">source image pixel values &lt; nCutoffValue will be considered fully outside of pixel regions (and set to -1).</param>
        /// <param name="nSubPixelXShift">final transform distances will be shifted in the X direction by this sub-pixel fraction. </param>
        /// <param name="nSubPixelYShift">final transform distances will be shifted in the Y direction by this sub-pixel fraction. </param>
        /// <param name="pDstVoronoi">device memory voronoi diagram destination_image_pointer or NULL for no voronoi output.</param>
        /// <param name="pDstVoronoiIndices">device memory voronoi diagram destination_image_pointer or NULL for no voronoi indices output.</param>
        /// <param name="pDstVoronoiManhattanRelativeDistances">device memory voronoi relative Manhattan distances destination_image_pointer or NULL for no voronoi Manhattan output.</param>
        /// <param name="pDstTransform">device memory true euclidean distance transform destination_image_pointer or NULL for no transform output.</param>
        /// <param name="pBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiSignedDistanceTransformPBAGet64fBufferSize() above)</param>
        /// <param name="pAntialiasingDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpAntialiasingBufferSize (see nppiSignedDistanceTransformPBAGetAntialiasingBufferSize() above) or NULL if not Antialiasing</param>
        public void SignedDistanceTransformPBA(double nCutoffValue,
                                           double nSubPixelXShift, double nSubPixelYShift, NPPImage_16sC1 pDstVoronoi, NPPImage_16sC1 pDstVoronoiIndices,
                                                        NPPImage_16sC1 pDstVoronoiManhattanRelativeDistances, NPPImage_64fC1 pDstTransform, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<byte> pAntialiasingDeviceBuffer)
        {
            CUdeviceptr dstVoronoi = new CUdeviceptr();
            CUdeviceptr dstTransform = new CUdeviceptr();
            CUdeviceptr dstVoronoiIndices = new CUdeviceptr();
            CUdeviceptr dstVoronoiManhattenDistances = new CUdeviceptr();
            CUdeviceptr antiAlias = new CUdeviceptr();
            int pitchVoronoi = 0;
            int pitchTransform = 0;
            int pitchVoronoiIndices = 0;
            int pitchVoronoiManhattenDistances = 0;

            if (pDstVoronoi != null)
            {
                dstVoronoi = pDstVoronoi.DevicePointerRoi;
                pitchVoronoi = pDstVoronoi.Pitch;
            }
            if (pDstTransform != null)
            {
                dstTransform = pDstTransform.DevicePointerRoi;
                pitchTransform = pDstTransform.Pitch;
            }
            if (pDstVoronoiIndices != null)
            {
                dstVoronoiIndices = pDstVoronoiIndices.DevicePointerRoi;
                pitchVoronoiIndices = pDstVoronoiIndices.Pitch;
            }
            if (pDstVoronoiManhattanRelativeDistances != null)
            {
                dstVoronoiManhattenDistances = pDstVoronoiManhattanRelativeDistances.DevicePointerRoi;
                pitchVoronoiManhattenDistances = pDstVoronoiManhattanRelativeDistances.Pitch;
            }
            if (pAntialiasingDeviceBuffer != null)
            {
                antiAlias = pAntialiasingDeviceBuffer.DevicePointer;
            }

            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiSignedDistanceTransformPBA_64f_C1R(_devPtrRoi, _pitch, nCutoffValue, nSubPixelXShift, nSubPixelYShift, dstVoronoi, pitchVoronoi, dstVoronoiIndices, pitchVoronoiIndices, dstVoronoiManhattenDistances, pitchVoronoiManhattenDistances, dstTransform, pitchTransform, _sizeRoi, pBuffer.DevicePointer, antiAlias);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSignedDistanceTransformPBA_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram 
        /// and 64-bit floating point transform with optional sub-pixel shifts. 
        /// <para/>
        /// For this particular version of the function acceptable input pixel intensities are less than or equal to 0.0f for those fully outside of connected 
        /// pixel regions, intensities with fractional parts between 0.0f and 1.0f representing the percentage of connected pixel region sub-pixel coverage within a 
        /// particular pixel (region contour), and intensities greater than or equal to 1.0f for pixels that are fully contained within closed connected pixel regions. 
        /// This function executes in two passes, the first pass prioritizes pixels outside of closed regions, the second pass 
        /// prioritizes pixels within closed regions.  The two passes are then merged on output. The function assumes that fully 
        /// covered pixels have centers located at sub-pixel locations of .5,.5. In general, object exterior distances are output as negative 
        /// numbers progressing to positive and object interior distances are output as positive numbers progressing to negative. 
        /// </summary>
        /// <param name="nCutoffValue">source image pixel values &lt; nCutoffValue will be considered fully outside of pixel regions (and set to -1).</param>
        /// <param name="nSubPixelXShift">final transform distances will be shifted in the X direction by this sub-pixel fraction. </param>
        /// <param name="nSubPixelYShift">final transform distances will be shifted in the Y direction by this sub-pixel fraction. </param>
        /// <param name="pDstVoronoi">device memory voronoi diagram destination_image_pointer or NULL for no voronoi output.</param>
        /// <param name="pDstVoronoiIndices">device memory voronoi diagram destination_image_pointer or NULL for no voronoi indices output.</param>
        /// <param name="pDstVoronoiAbsoluteManhattanDistances">device memory voronoi relative Manhattan distances destination_image_pointer or NULL for no voronoi Manhattan output.</param>
        /// <param name="pDstTransform">device memory true euclidean distance transform destination_image_pointer or NULL for no transform output.</param>
        /// <param name="pBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiSignedDistanceTransformPBAGet64fBufferSize() above)</param>
        /// <param name="pAntialiasingDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpAntialiasingBufferSize (see nppiSignedDistanceTransformPBAGetAntialiasingBufferSize() above) or NULL if not Antialiasing</param>
        public void SignedDistanceTransformAbsPBA(double nCutoffValue, double nSubPixelXShift, double nSubPixelYShift, NPPImage_16sC1 pDstVoronoi, NPPImage_16sC1 pDstVoronoiIndices,
                                                  NPPImage_16sC1 pDstVoronoiAbsoluteManhattanDistances, NPPImage_64fC1 pDstTransform, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<byte> pAntialiasingDeviceBuffer)
        {
            CUdeviceptr dstVoronoi = new CUdeviceptr();
            CUdeviceptr dstTransform = new CUdeviceptr();
            CUdeviceptr dstVoronoiIndices = new CUdeviceptr();
            CUdeviceptr dstVoronoiManhattenDistances = new CUdeviceptr();
            CUdeviceptr antiAlias = new CUdeviceptr();
            int pitchVoronoi = 0;
            int pitchTransform = 0;
            int pitchVoronoiIndices = 0;
            int pitchVoronoiManhattenDistances = 0;

            if (pDstVoronoi != null)
            {
                dstVoronoi = pDstVoronoi.DevicePointerRoi;
                pitchVoronoi = pDstVoronoi.Pitch;
            }
            if (pDstTransform != null)
            {
                dstTransform = pDstTransform.DevicePointerRoi;
                pitchTransform = pDstTransform.Pitch;
            }
            if (pDstVoronoiIndices != null)
            {
                dstVoronoiIndices = pDstVoronoiIndices.DevicePointerRoi;
                pitchVoronoiIndices = pDstVoronoiIndices.Pitch;
            }
            if (pDstVoronoiAbsoluteManhattanDistances != null)
            {
                dstVoronoiManhattenDistances = pDstVoronoiAbsoluteManhattanDistances.DevicePointerRoi;
                pitchVoronoiManhattenDistances = pDstVoronoiAbsoluteManhattanDistances.Pitch;
            }
            if (pAntialiasingDeviceBuffer != null)
            {
                antiAlias = pAntialiasingDeviceBuffer.DevicePointer;
            }

            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiSignedDistanceTransformAbsPBA_64f_C1R(_devPtrRoi, _pitch, nCutoffValue, nSubPixelXShift, nSubPixelYShift, dstVoronoi, pitchVoronoi, dstVoronoiIndices, pitchVoronoiIndices, dstVoronoiManhattenDistances, pitchVoronoiManhattenDistances, dstTransform, pitchTransform, _sizeRoi, pBuffer.DevicePointer, antiAlias);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSignedDistanceTransformAbsPBA_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }



        /// <summary>
        /// 1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or 
        /// optional 64-bit floating point transform with optional relative Manhattan distances.
        /// </summary>
        /// <param name="nMinSiteValue">source image pixel values >= nMinSiteValue and &lt;= nMaxSiteValue are considered sites (traditionally 0s)</param>
        /// <param name="nMaxSiteValue">source image pixel values >= nMinSiteValue and &lt;= nMaxSiteValue are considered sites (traditionally 0s)</param>
        /// <param name="pDstVoronoi">device memory voronoi diagram destination_image_pointer or NULL for no voronoi output.</param>
        /// <param name="pDstVoronoiIndices">device memory voronoi diagram destination_image_pointer or NULL for no voronoi indices output.</param>
        /// <param name="pDstVoronoiManhattanRelativeDistances">device memory voronoi relative Manhattan distances destination_image_pointer or NULL for no voronoi Manhattan output.</param>
        /// <param name="pDstTransform">device memory true euclidean distance transform destination_image_pointer or NULL for no transform output.</param>
        /// <param name="pBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiDistanceTransformPBAGet64fBufferSize() above)</param>
        /// <param name="pAntialiasingDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpAntialiasingBufferSize (see nppiDistanceTransformPBAGetAntialiasingBufferSize() above) or NULL if not Antialiasing</param>
        public void DistanceTransformPBA(double nMinSiteValue, double nMaxSiteValue, NPPImage_16sC1 pDstVoronoi, NPPImage_16sC1 pDstVoronoiIndices,
                                                        NPPImage_16sC1 pDstVoronoiManhattanRelativeDistances, NPPImage_64fC1 pDstTransform, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<byte> pAntialiasingDeviceBuffer)
        {
            CUdeviceptr dstVoronoi = new CUdeviceptr();
            CUdeviceptr dstTransform = new CUdeviceptr();
            CUdeviceptr dstVoronoiIndices = new CUdeviceptr();
            CUdeviceptr dstVoronoiManhattenDistances = new CUdeviceptr();
            CUdeviceptr antiAlias = new CUdeviceptr();
            int pitchVoronoi = 0;
            int pitchTransform = 0;
            int pitchVoronoiIndices = 0;
            int pitchVoronoiManhattenDistances = 0;

            if (pDstVoronoi != null)
            {
                dstVoronoi = pDstVoronoi.DevicePointerRoi;
                pitchVoronoi = pDstVoronoi.Pitch;
            }
            if (pDstTransform != null)
            {
                dstTransform = pDstTransform.DevicePointerRoi;
                pitchTransform = pDstTransform.Pitch;
            }
            if (pDstVoronoiIndices != null)
            {
                dstVoronoiIndices = pDstVoronoiIndices.DevicePointerRoi;
                pitchVoronoiIndices = pDstVoronoiIndices.Pitch;
            }
            if (pDstVoronoiManhattanRelativeDistances != null)
            {
                dstVoronoiManhattenDistances = pDstVoronoiManhattanRelativeDistances.DevicePointerRoi;
                pitchVoronoiManhattenDistances = pDstVoronoiManhattanRelativeDistances.Pitch;
            }
            if (pAntialiasingDeviceBuffer != null)
            {
                antiAlias = pAntialiasingDeviceBuffer.DevicePointer;
            }

            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiDistanceTransformPBA_64f_C1R(_devPtrRoi, _pitch, nMinSiteValue, nMaxSiteValue, dstVoronoi, pitchVoronoi, dstVoronoiIndices, pitchVoronoiIndices, dstVoronoiManhattenDistances, pitchVoronoiManhattenDistances, dstTransform, pitchTransform, _sizeRoi, pBuffer.DevicePointer, antiAlias);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDistanceTransformPBA_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// 1 channel 64-bit floating point grayscale to optional 1 channel 16-bit signed integer euclidean distance voronoi diagram output and/or  
        /// optional 64-bit floating point transform with optional absolute Manhattan distances
        /// </summary>
        /// <param name="nMinSiteValue">source image pixel values >= nMinSiteValue and &lt;= nMaxSiteValue are considered sites (traditionally 0s)</param>
        /// <param name="nMaxSiteValue">source image pixel values >= nMinSiteValue and &lt;= nMaxSiteValue are considered sites (traditionally 0s)</param>
        /// <param name="pDstVoronoi">device memory voronoi diagram destination_image_pointer or NULL for no voronoi output.</param>
        /// <param name="pDstVoronoiIndices">device memory voronoi diagram destination_image_pointer or NULL for no voronoi indices output.</param>
        /// <param name="pDstVoronoiAbsoluteManhattanDistances">device memory voronoi relative Manhattan distances destination_image_pointer or NULL for no voronoi Manhattan output.</param>
        /// <param name="pDstTransform">device memory true euclidean distance transform destination_image_pointer or NULL for no transform output.</param>
        /// <param name="pBuffer">pointer to scratch DEVICE memory buffer of size hpBufferSize (see nppiDistanceTransformPBAGet64fBufferSize() above)</param>
        /// <param name="pAntialiasingDeviceBuffer">pointer to scratch DEVICE memory buffer of size hpAntialiasingBufferSize (see nppiDistanceTransformPBAGetAntialiasingBufferSize() above) or NULL if not Antialiasing</param>
        public void DistanceTransformAbsPBA(double nMinSiteValue, double nMaxSiteValue, NPPImage_16sC1 pDstVoronoi, NPPImage_16sC1 pDstVoronoiIndices,
                                                  NPPImage_16sC1 pDstVoronoiAbsoluteManhattanDistances, NPPImage_64fC1 pDstTransform, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<byte> pAntialiasingDeviceBuffer)
        {
            CUdeviceptr dstVoronoi = new CUdeviceptr();
            CUdeviceptr dstTransform = new CUdeviceptr();
            CUdeviceptr dstVoronoiIndices = new CUdeviceptr();
            CUdeviceptr dstVoronoiManhattenDistances = new CUdeviceptr();
            CUdeviceptr antiAlias = new CUdeviceptr();
            int pitchVoronoi = 0;
            int pitchTransform = 0;
            int pitchVoronoiIndices = 0;
            int pitchVoronoiManhattenDistances = 0;

            if (pDstVoronoi != null)
            {
                dstVoronoi = pDstVoronoi.DevicePointerRoi;
                pitchVoronoi = pDstVoronoi.Pitch;
            }
            if (pDstTransform != null)
            {
                dstTransform = pDstTransform.DevicePointerRoi;
                pitchTransform = pDstTransform.Pitch;
            }
            if (pDstVoronoiIndices != null)
            {
                dstVoronoiIndices = pDstVoronoiIndices.DevicePointerRoi;
                pitchVoronoiIndices = pDstVoronoiIndices.Pitch;
            }
            if (pDstVoronoiAbsoluteManhattanDistances != null)
            {
                dstVoronoiManhattenDistances = pDstVoronoiAbsoluteManhattanDistances.DevicePointerRoi;
                pitchVoronoiManhattenDistances = pDstVoronoiAbsoluteManhattanDistances.Pitch;
            }
            if (pAntialiasingDeviceBuffer != null)
            {
                antiAlias = pAntialiasingDeviceBuffer.DevicePointer;
            }

            status = NPPNativeMethods.NPPi.FilterDistanceTransform.nppiDistanceTransformAbsPBA_64f_C1R(_devPtrRoi, _pitch, nMinSiteValue, nMaxSiteValue, dstVoronoi, pitchVoronoi, dstVoronoiIndices, pitchVoronoiIndices, dstVoronoiManhattenDistances, pitchVoronoiManhattenDistances, dstTransform, pitchTransform, _sizeRoi, pBuffer.DevicePointer, antiAlias);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDistanceTransformAbsPBA_64f_C1R", status));
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
        public void FilterBoxBorderAdvanced(NPPImage_64fC1 dest, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, CudaDeviceVariable<byte> pBuffer, NppiRect filterArea = new NppiRect())
        {
            if (filterArea.Size == new NppiSize())
            {
                filterArea.Size = _sizeRoi;
            }
            status = NPPNativeMethods.NPPi.LinearFixedFilters2D.nppiFilterBoxBorderAdvanced_64f_C1R(_devPtrRoi, _pitch, filterArea.Size, filterArea.Location, dest.DevicePointerRoi,
                                                    dest.Pitch, dest.SizeRoi, oMaskSize, oAnchor, eBorderType, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFilterBoxBorderAdvanced_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// CrossCorrFull_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrFull_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevelAdvanced_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevelAdvanced_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrSame_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrSame_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevelAdvanced_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevelAdvanced_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrValid_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize</param>
        public void CrossCorrValid_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevelAdvanced_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevelAdvanced_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
#endif


        /// <summary>
        /// Device scratch buffer size (in bytes) for CrossCorrFull_NormLevel.
        /// </summary>
        /// <returns></returns>
        public int FullNormLevelGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// CrossCorrFull_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Allocated device memory with size of at <see cref="FullNormLevelGetBufferHostSize()"/></param>
        public void CrossCorrFull_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = FullNormLevelGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        public void CrossCorrFull_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst)
        {
            int bufferSize = FullNormLevelGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }



        /// <summary>
        /// Device scratch buffer size (in bytes) for CrossCorrSame_NormLevel.
        /// </summary>
        /// <returns></returns>
        public int SameNormLevelGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// CrossCorrSame_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Allocated device memory with size of at <see cref="SameNormLevelGetBufferHostSize()"/></param>
        public void CrossCorrSame_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = SameNormLevelGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        public void CrossCorrSame_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst)
        {
            int bufferSize = SameNormLevelGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }




        /// <summary>
        /// Device scratch buffer size (in bytes) for CrossCorrValid_NormLevel.
        /// </summary>
        /// <returns></returns>
        public int ValidNormLevelGetBufferHostSize()
        {
            int bufferSize = 0;
            status = NPPNativeMethods.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_64f_C1R(_sizeRoi, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// CrossCorrValid_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Allocated device memory with size of at <see cref="ValidNormLevelGetBufferHostSize()"/></param>
        public void CrossCorrValid_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst, CudaDeviceVariable<byte> buffer)
        {
            int bufferSize = ValidNormLevelGetBufferHostSize();
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        public void CrossCorrValid_NormLevel(NPPImage_64fC1 tpl, NPPImage_64fC1 dst)
        {
            int bufferSize = ValidNormLevelGetBufferHostSize();
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_64f_C1R", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image CrossCorrValid.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination-Image</param>
        public void CrossCorrValid(NPPImage_64fC1 tpl, NPPImage_64fC1 dst)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image CrossCorrFull_Norm.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination-Image</param>
        public void CrossCorrFull_Norm(NPPImage_64fC1 tpl, NPPImage_64fC1 dst)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrFull_Norm_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image CrossCorrSame_Norm.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination-Image</param>
        public void CrossCorrSame_Norm(NPPImage_64fC1 tpl, NPPImage_64fC1 dst)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrSame_Norm_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image CrossCorrValid_Norm.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination-Image</param>
        public void CrossCorrValid_Norm(NPPImage_64fC1 tpl, NPPImage_64fC1 dst)
        {
            status = NPPNativeMethods.NPPi.ImageProximity.nppiCrossCorrValid_Norm_64f_C1R(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_64f_C1R", status));
            NPPException.CheckNppStatus(status, this);
        }
        #endregion
    }
}
