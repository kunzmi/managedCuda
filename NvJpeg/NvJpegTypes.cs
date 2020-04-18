//	Copyright (c) 2020, Michael Kunz. All rights reserved.
//	http://kunzmi.github.io/managedCuda
//
//	This file is part of ManagedCuda.
//
//	ManagedCuda is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as 
//	published by the Free Software Foundation, either version 2.1 of the 
//	License, or (at your option) any later version.
//
//	ManagedCuda is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.

using System;
using System.Collections.Generic;
using System.Text;
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
        public const int NVJPEG_FLAGS_DEFAULT = 0;
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
        BGRI = 6
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
        /// nvjpegDecodeBatched will use GPU decoding for baseline JPEG images with 
        /// interleaved scan when batch size is bigger than 100, batched multi phase APIs 
        /// will use CPU Huffman decode. All Single Image APIs will use GPU assisted huffman decode
        /// </summary>
        GPUHybrid = 2,
    }


    /// <summary>
    /// Currently parseable JPEG encodings (SOF markers)
    /// </summary>
    public enum nvjpegJpegEncoding
    {
        Unknown = 0x0,
        BaselineDCT = 0xc0,
        ProgressiveDCTHuffman = 0xc2
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
    /// Prototype for device memory release
    /// </summary>
    public delegate int tPinnedFree(IntPtr ptr);

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
        public int[] pitch;

        public nvjpegImage(CUdeviceptr aDevPtr, int aPitch)
        {
            channel = new CUdeviceptr[Constants.MaxComponent];
            pitch = new int[Constants.MaxComponent];

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
            pitch = new int[Constants.MaxComponent];

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
