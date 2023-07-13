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
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NvJpeg
{
    public static class Constants
    {
        /// <summary>
        /// Maximum number of channels nvjpeg decoder supports
        /// </summary>
        public const int MaxComponent = 4;
        /// <summary>
        /// 
        /// </summary>
        public const int NVJPEG_FLAGS_DEFAULT = 0;
        /// <summary>
        /// use this to disable pipelining for hardware decoder backend
        /// </summary>
        public const int NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE = 1;
        /// <summary>
        /// 
        /// </summary>
        public const int NVJPEG_FLAGS_ENABLE_MEMORY_POOLS = 1 << 1;
        /// <summary>
        /// 
        /// </summary>
        public const int NVJPEG_FLAGS_BITSTREAM_STRICT = 1 << 2;
        /// <summary>
        /// 
        /// </summary>
        public const int NVJPEG_FLAGS_REDUCED_MEMORY_DECODE = 1 << 3;
        /// <summary>
        /// 
        /// </summary>
        public const int NVJPEG_FLAGS_REDUCED_MEMORY_DECODE_ZERO_COPY = 1 << 4;
        /// <summary>
        /// 
        /// </summary>
        public const int NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION = 1 << 5;
    }

    /// <summary>
    /// nvJPEG status enums, returned by nvJPEG API
    /// </summary>
    public enum nvjpegStatus
    {
        /// <summary>
        /// The API call has finished successfully. Note that many of the calls are asynchronous and some of the errors may be seen only after synchronization. 
        /// </summary>
        Success = 0,
        /// <summary>
        /// The library handle was not initialized. A call to nvjpegCreate() is required to initialize the handle.
        /// </summary>
        NotInitialized = 1,
        /// <summary>
        /// Wrong parameter was passed. For example, a null pointer as input data, or an image index not in the allowed range. 
        /// </summary>
        InvalidParameter = 2,
        /// <summary>
        /// Cannot parse the JPEG stream. Check that the encoded JPEG stream and its size parameters are correct.
        /// </summary>
        BadJPEG = 3,
        /// <summary>
        /// Attempting to decode a JPEG stream that is not supported by the nvJPEG library
        /// </summary>
        JPEGNotSupported = 4,
        /// <summary>
        /// The user-provided allocator functions, for either memory allocation or for releasing the memory, returned a non-zero code.
        /// </summary>
        AllocatorFailure = 5,
        /// <summary>
        /// Error during the execution of the device tasks.
        /// </summary>
        ExecutionFailed = 6,
        /// <summary>
        /// The device capabilities are not enough for the set of input parameters provided (input parameters such as backend, encoded stream parameters, output format).
        /// </summary>
        ArchMismatch = 7,
        /// <summary>
        /// Error during the execution of the device tasks. 
        /// </summary>
        InternalError = 8,
        /// <summary>
        /// Not supported.
        /// </summary>
        ImplementationNotSupported = 9,
        /// <summary>
        /// Incomplete bit stream.
        /// </summary>
        IncompleteBitstream = 10,
    }


    /// <summary>
    /// Enums for EXIF Orientation
    /// </summary>
    public enum nvjpegExifOrientation
    {
        Unknown = 0,
        Normal = 1,
        FlipHorizontal = 2,
        Rotate180 = 3,
        FlipVertical = 4,
        Transpose = 5,
        Rotate90 = 6,
        Transverse = 7,
        Rotate270 = 8
    }


    /// <summary>
    /// Enum identifies image chroma subsampling values stored inside JPEG input stream
    /// In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
    /// Otherwise both chroma planes are present
    /// </summary>
    public enum nvjpegChromaSubsampling
    {
        CSS_444 = 0,
        CSS_422 = 1,
        CSS_420 = 2,
        CSS_440 = 3,
        CSS_411 = 4,
        CSS_410 = 5,
        CSS_Gray = 6,
        CSS_410V = 7,
        CSS_Unknown = -1
    }


    /// <summary>
    /// Parameter of this type specifies what type of output user wants for image decoding
    /// </summary>
    public enum nvjpegOutputFormat
    {
        /// <summary>
        /// return decompressed image as it is - write planar output
        /// </summary>
        Unchanged = 0,
        /// <summary>
        /// return planar luma and chroma, assuming YCbCr colorspace
        /// </summary>
        YUV = 1,
        /// <summary>
        /// return luma component only, if YCbCr colorspace, or try to convert to grayscale, writes to 1-st channel of nvjpegImage_t
        /// </summary>
        Y = 2,
        /// <summary>
        /// convert to planar RGB 
        /// </summary>
        RGB = 3,
        /// <summary>
        /// convert to planar BGR
        /// </summary>
        BGR = 4,
        /// <summary>
        /// convert to interleaved RGB and write to 1-st channel of nvjpegImage_t
        /// </summary>
        RGBI = 5,
        /// <summary>
        /// convert to interleaved BGR and write to 1-st channel of nvjpegImage_t
        /// </summary>
        BGRI = 6,
        /// <summary>
        /// 16 bit unsigned output in interleaved format
        /// </summary>
        UnchangedU16 = 7,
    }


    /// <summary>
    /// Parameter of this type specifies what type of input user provides for encoding
    /// </summary>
    public enum nvjpegInputFormat
    {
        /// <summary>
        /// Input is RGB - will be converted to YCbCr before encoding
        /// </summary>
        RGB = 3,
        /// <summary>
        /// Input is RGB - will be converted to YCbCr before encoding
        /// </summary>
        BGR = 4,
        /// <summary>
        /// Input is interleaved RGB - will be converted to YCbCr before encoding
        /// </summary>
        RGBI = 5,
        /// <summary>
        /// Input is interleaved RGB - will be converted to YCbCr before encoding
        /// </summary>
        BGRI = 6
    }


    /// <summary>
    /// 
    /// </summary>
    public enum nvjpegBackend
    {
        /// <summary>
        /// default value
        /// </summary>
        Default = 0,
        /// <summary>
        /// uses CPU for Huffman decode
        /// </summary>
        Hybrid = 1,
        /// <summary>
        /// uses GPU assisted Huffman decode. nvjpegDecodeBatched will use GPU decoding for baseline JPEG bitstreams with
        /// interleaved scan when batch size is bigger than 50
        /// </summary>
        GPUHybrid = 2,
        /// <summary>
        /// supports baseline JPEG bitstream with single scan. 410 and 411 sub-samplings are not supported
        /// </summary>
        Hardware = 3,
        /// <summary>
        /// nvjpegDecodeBatched will support bitstream input on device memory
        /// </summary>
        GPUHybridDevice = 4,
        /// <summary>
        /// nvjpegDecodeBatched will support bitstream input on device memory
        /// </summary>
        HardwareDevice = 5,
        /// <summary>
        /// LossLessJPEG
        /// </summary>
        LossLessJPEG = 6,
    }


    /// <summary>
    /// Currently parseable JPEG encodings (SOF markers)
    /// </summary>
    public enum nvjpegJpegEncoding
    {
        Unknown = 0x0,
        BaselineDCT = 0xc0,
        ExtendedSequentialDCTHuffman = 0xc1,
        ProgressiveDCTHuffman = 0xc2,
        LossLessHuffman = 0xc3
    }

    /// <summary>
    /// 
    /// </summary>
    public enum nvjpegScaleFactor
    {
        /// <summary>
        /// decoded output is not scaled 
        /// </summary>
        Scale_None = 0,
        /// <summary>
        /// decoded output width and height is scaled by a factor of 1/2
        /// </summary>
        Scale_1_by_2 = 1,
        /// <summary>
        /// decoded output width and height is scaled by a factor of 1/4
        /// </summary>
        Scale_1_by_4 = 2,
        /// <summary>
        /// decoded output width and height is scaled by a factor of 1/8
        /// </summary>
        Scale_1_by_8 = 3,
    }

    /// <summary>
    /// Prototype for device memory allocation, modelled after cudaMalloc()
    /// </summary>
    public delegate int tDevMalloc(ref CUdeviceptr ptr, SizeT size);

    /// <summary>
    /// Prototype for device memory release
    /// </summary>
    public delegate int tDevFree(CUdeviceptr ptr);

    /// <summary>
    /// Prototype for pinned memory allocation, modelled after cudaHostAlloc()
    /// </summary>
    public delegate int tPinnedMalloc(ref IntPtr ptr, SizeT size, uint flags);

    /// <summary>
    /// Prototype for pinned memory release
    /// </summary>
    public delegate int tPinnedFree(IntPtr ptr);


    /// <summary>
    /// Prototype for device memory allocation, modelled after cudaMalloc()
    /// </summary>
    public delegate int tDevMallocV2(IntPtr ctx, ref CUdeviceptr ptr, SizeT size, CUstream stream);

    /// <summary>
    /// Prototype for device memory release
    /// </summary>
    public delegate int tDevFreeV2(IntPtr ctx, CUdeviceptr ptr, SizeT size, CUstream stream);


    /// <summary>
    /// Prototype for pinned memory allocation, modelled after cudaHostAlloc()
    /// </summary>
    public delegate int tPinnedMallocV2(IntPtr ctx, ref IntPtr ptr, SizeT size, CUstream stream);

    /// <summary>
    /// Prototype for pinned memory release
    /// </summary>
    public delegate int tPinnedFreeV2(IntPtr ctx, IntPtr ptr, SizeT size, CUstream stream);

    /// <summary>
    /// Output descriptor.
    /// Data that is written to planes depends on output forman
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegImage
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = Constants.MaxComponent)]
        public CUdeviceptr[] channel;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = Constants.MaxComponent)]
        public SizeT[] pitch;

        public nvjpegImage(CUdeviceptr aDevPtr, int aPitch)
        {
            channel = new CUdeviceptr[Constants.MaxComponent];
            pitch = new SizeT[Constants.MaxComponent];

            channel[0] = aDevPtr;
            pitch[0] = aPitch;
        }
        public nvjpegImage(CUdeviceptr[] aDevPtr, int[] aPitch)
        {
            if (aDevPtr.Length != aPitch.Length || aDevPtr.Length > 4)
            {
                throw new ArgumentException("aDevPtr and aPitch must habe same length and must be <= 4");
            }

            channel = new CUdeviceptr[Constants.MaxComponent];
            pitch = new SizeT[Constants.MaxComponent];

            for (int i = 0; i < aDevPtr.Length; i++)
            {
                channel[i] = aDevPtr[i];
                pitch[i] = aPitch[i];
            }
        }
    }

    /// <summary>
    /// Memory allocator using mentioned prototypes, provided to nvjpegCreateEx
    /// This allocator will be used for all device memory allocations inside library
    /// In any way library is doing smart allocations (reallocates memory only if needed)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegDevAllocator
    {
        public tDevMalloc dev_malloc;
        public tDevFree dev_free;
    }

    /// <summary>
    /// Pinned memory allocator using mentioned prototypes, provided to nvjpegCreate
    /// This allocator will be used for all pinned host memory allocations inside library
    /// In any way library is doing smart allocations (reallocates memory only if needed)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegPinnedAllocator
    {
        public tPinnedMalloc pinned_malloc;
        public tPinnedFree pinned_free;
    }

    /// <summary>
    /// Memory allocator using mentioned prototypes, provided to nvjpegCreateEx
    /// This allocator will be used for all device memory allocations inside library
    /// In any way library is doing smart allocations (reallocates memory only if needed)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegDevAllocatorV2
    {
        tDevMallocV2 dev_malloc;
        tDevFreeV2 dev_free;
        IntPtr dev_ctx;
    }

    /// <summary>
    /// Pinned memory allocator using mentioned prototypes, provided to nvjpegCreate
    /// This allocator will be used for all pinned host memory allocations inside library
    /// In any way library is doing smart allocations (reallocates memory only if needed)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegPinnedAllocatorV2
    {
        tPinnedMallocV2 pinned_malloc;
        tPinnedFreeV2 pinned_free;
        IntPtr pinned_ctx;
    }


    /// <summary>
    /// Opaque library handle identifier.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }


    /// <summary>
    /// Opaque jpeg decoding state handle identifier - used to store intermediate information between deccding phases
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegJpegState
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegEncoderState
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegEncoderParams
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegBufferPinned
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegBufferDevice
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// handle that stores stream information - metadata, encoded image parameters, encoded stream parameters
    /// stores everything on CPU side. This allows us parse header separately from implementation
    /// and retrieve more information on the stream. Also can be used for transcoding and transfering 
    /// metadata to encoder
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegJpegStream
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegDecodeParams
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// decode parameters structure. Used to set decode-related tweaks
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvjpegJpegDecoder
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }
}
