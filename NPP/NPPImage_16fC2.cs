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
    public partial class NPPImage_16fC2 : NPPImageBase
    {
        #region Constructors
        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="nWidthPixels">Image width in pixels</param>
        /// <param name="nHeightPixels">Image height in pixels</param>
        public NPPImage_16fC2(int nWidthPixels, int nHeightPixels)
        {
            _sizeOriginal.width = nWidthPixels;
            _sizeOriginal.height = nHeightPixels;
            _sizeRoi.width = nWidthPixels;
            _sizeRoi.height = nHeightPixels;
            _channels = 2;
            _isOwner = true;
            _typeSize = sizeof(ushort);
            _dataType = NppDataType.NPP_16F;
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
        public NPPImage_16fC2(CUdeviceptr devPtr, int width, int height, int pitch, bool isOwner)
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
            _dataType = NppDataType.NPP_16F;
            _nppChannels = NppiChannels.NPP_CH_2;
        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of decPtr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="width">Image width in pixels</param>
        /// <param name="height">Image height in pixels</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16fC2(CUdeviceptr devPtr, int width, int height, int pitch)
            : this(devPtr, width, height, pitch, false)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr. Does not take ownership of inner image device pointer.
        /// </summary>
        /// <param name="image">NPP image</param>
        public NPPImage_16fC2(NPPImageBase image)
            : this(image.DevicePointer, image.Width, image.Height, image.Pitch, false)
        {

        }

        /// <summary>
        /// Allocates new memory on device using NPP-Api.
        /// </summary>
        /// <param name="size">Image size</param>
        public NPPImage_16fC2(NppiSize size)
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
        public NPPImage_16fC2(CUdeviceptr devPtr, NppiSize size, int pitch, bool isOwner)
            : this(devPtr, size.width, size.height, pitch, isOwner)
        {

        }

        /// <summary>
        /// Creates a new NPPImage from allocated device ptr.
        /// </summary>
        /// <param name="devPtr">Already allocated device ptr.</param>
        /// <param name="size">Image size</param>
        /// <param name="pitch">Pitch / Line step</param>
        public NPPImage_16fC2(CUdeviceptr devPtr, NppiSize size, int pitch)
            : this(devPtr, size.width, size.height, pitch)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~NPPImage_16fC2()
        {
            Dispose(false);
        }
        #endregion

        #region ColorTwist
        /// <summary>
        /// An input color twist matrix with floating-point pixel values is applied
        /// within ROI.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
        public void ColorTwist(NPPImage_16fC2 dest, float[,] twistMatrix)
        {
            status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_16f_C2R(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, twistMatrix);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16f_C2R", status));
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
            status = NPPNativeMethods.NPPi.ColorTwist.nppiColorTwist32f_16f_C2IR(_devPtrRoi, _pitch, _sizeRoi, aTwist);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16f_C2IR", status));
            NPPException.CheckNppStatus(status, this);
        }

        #endregion
    }
}