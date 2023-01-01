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


#define ADD_MISSING_CTX
using System;
using System.Diagnostics;

namespace ManagedCuda.NPP
{
    /// <summary>
    /// 
    /// </summary>
    public partial class NPPImage_64fC3 : NPPImageBase
    {
        #region MaxError
        /// <summary>
        /// image maximum error. User buffer is internally allocated and freed.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void MaxError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
        {
            int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_64f_C3R_Ctx", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void MaxError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
        {
            int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaxError.
        /// </summary>
        /// <returns></returns>
        public int MaxErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int bufferSize = 0;
            status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_64f_C3R_Ctx", status));
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
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void AverageError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
        {
            int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_64f_C3R_Ctx", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void AverageError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
        {
            int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageError.
        /// </summary>
        /// <returns></returns>
        public int AverageErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int bufferSize = 0;
            status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_64f_C3R_Ctx", status));
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
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void MaximumRelativeError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
        {
            int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_64f_C3R_Ctx", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image maximum relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void MaximumRelativeError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
        {
            int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for MaximumRelativeError.
        /// </summary>
        /// <returns></returns>
        public int MaximumRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int bufferSize = 0;
            status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_64f_C3R_Ctx", status));
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
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void AverageRelativeError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
        {
            int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
            status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_64f_C3R_Ctx", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image average relative error.
        /// </summary>
        /// <param name="src2">2nd source image</param>
        /// <param name="pError">Pointer to the computed error.</param>
        /// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void AverageRelativeError(NPPImage_64fC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
        {
            int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_64f_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// Device scratch buffer size (in bytes) for AverageRelativeError.
        /// </summary>
        /// <returns></returns>
        public int AverageRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int bufferSize = 0;
            status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }
        #endregion

#if ADD_MISSING_CTX

        /// <summary>
        /// CrossCorrFull_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrFull_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevelAdvanced_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevelAdvanced_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrSame_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrSame_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevelAdvanced_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevelAdvanced_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
        /// <summary>
        /// CrossCorrValid_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Pointer to the required device memory allocation. </param>
        /// <param name="bufferAdvanced">Pointer to the required device memory allocation. See nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrValid_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<byte> bufferAdvanced, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevelAdvanced_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, bufferAdvanced.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevelAdvanced_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
#endif

        /// <summary>
        /// Device scratch buffer size (in bytes) for CrossCorrFull_NormLevel.
        /// </summary>
        /// <returns></returns>
        public int FullNormLevelGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int bufferSize = 0;
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiFullNormLevelGetBufferHostSize_64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiFullNormLevelGetBufferHostSize_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// CrossCorrFull_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Allocated device memory with size of at <see cref="FullNormLevelGetBufferHostSize()"/></param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrFull_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
        {
            int bufferSize = FullNormLevelGetBufferHostSize(nppStreamCtx);
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// CrossCorrFull_NormLevel. Buffer is internally allocated and freed.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrFull_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, NppStreamContext nppStreamCtx)
        {
            int bufferSize = FullNormLevelGetBufferHostSize(nppStreamCtx);
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_NormLevel_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_NormLevel_64f_C3R_Ctx", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }



        /// <summary>
        /// Device scratch buffer size (in bytes) for CrossCorrSame_NormLevel.
        /// </summary>
        /// <returns></returns>
        public int SameNormLevelGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int bufferSize = 0;
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiSameNormLevelGetBufferHostSize_64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSameNormLevelGetBufferHostSize_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// CrossCorrSame_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Allocated device memory with size of at <see cref="SameNormLevelGetBufferHostSize()"/></param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrSame_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
        {
            int bufferSize = SameNormLevelGetBufferHostSize(nppStreamCtx);
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// CrossCorrSame_NormLevel. Buffer is internally allocated and freed.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrSame_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, NppStreamContext nppStreamCtx)
        {
            int bufferSize = SameNormLevelGetBufferHostSize(nppStreamCtx);
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_NormLevel_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_NormLevel_64f_C3R_Ctx", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }




        /// <summary>
        /// Device scratch buffer size (in bytes) for CrossCorrValid_NormLevel.
        /// </summary>
        /// <returns></returns>
        public int ValidNormLevelGetBufferHostSize(NppStreamContext nppStreamCtx)
        {
            int bufferSize = 0;
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiValidNormLevelGetBufferHostSize_64f_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiValidNormLevelGetBufferHostSize_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
            return bufferSize;
        }

        /// <summary>
        /// CrossCorrValid_NormLevel.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="buffer">Allocated device memory with size of at <see cref="ValidNormLevelGetBufferHostSize()"/></param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrValid_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
        {
            int bufferSize = ValidNormLevelGetBufferHostSize(nppStreamCtx);
            if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// CrossCorrValid_NormLevel. Buffer is internally allocated and freed.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination image</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrValid_NormLevel(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, NppStreamContext nppStreamCtx)
        {
            int bufferSize = ValidNormLevelGetBufferHostSize(nppStreamCtx);
            CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);

            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_NormLevel_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointer, dst.Pitch, buffer.DevicePointer, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_NormLevel_64f_C3R_Ctx", status));
            buffer.Dispose();
            NPPException.CheckNppStatus(status, this);
        }


        /// <summary>
        /// image CrossCorrFull_Norm.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination-Image</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrFull_Norm(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrFull_Norm_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrFull_Norm_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image CrossCorrSame_Norm.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination-Image</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrSame_Norm(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrSame_Norm_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrSame_Norm_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// image CrossCorrValid_Norm.
        /// </summary>
        /// <param name="tpl">template image.</param>
        /// <param name="dst">Destination-Image</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void CrossCorrValid_Norm(NPPImage_64fC3 tpl, NPPImage_64fC3 dst, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ImageProximity.nppiCrossCorrValid_Norm_64f_C3R_Ctx(_devPtrRoi, _pitch, _sizeRoi, tpl.DevicePointerRoi, tpl.Pitch, tpl.SizeRoi, dst.DevicePointerRoi, dst.Pitch, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCrossCorrValid_Norm_64f_C3R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }
    }
}
