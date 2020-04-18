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
using ManagedCuda.VectorTypes;

namespace ManagedCuda.NvJpeg
{
    /// <summary>
    /// C# wrapper for nvjpeg.h
    /// </summary>
    public class NvJpegNativeMethods
    {
        internal const string NVJPEG_API_DLL_NAME = "nvjpeg64_10";



        /// <summary>
        /// returns library's property values, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
        /// </summary>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegGetProperty(libraryPropertyType type, ref int value);


        /// <summary>
        /// returns CUDA Toolkit property values that was used for building library, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegGetCudartProperty(libraryPropertyType type, ref int value);


        /// <summary>
        /// Initalization of nvjpeg handle with default backend and default memory allocators.
		/// </summary>
        /// <param name="handle">Codec instance, use for other calls</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegCreateSimple(ref nvjpegHandle handle);

      
        /// <summary>
        /// Initalization of nvjpeg handle with additional parameters. This handle is used for all consecutive nvjpeg calls
		/// </summary>
        /// <param name="backend">Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.</param>
        /// <param name="dev_allocator">Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)</param>
        /// <param name="pinned_allocator">Pointer to nvjpegPinnedAllocator. If NULL - use default cuda calls (cudaHostAlloc/cudaFreeHost)</param>
        /// <param name="flags">Parameters for the operation. Must be 0.</param>
        /// <param name="handle">Codec instance, use for other calls</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegCreateEx(nvjpegBackend backend,
                ref nvjpegDevAllocator dev_allocator,
                ref nvjpegPinnedAllocator pinned_allocator,
                uint flags,
                ref nvjpegHandle handle);

        /// <summary>
        /// Initalization of nvjpeg handle with additional parameters. This handle is used for all consecutive nvjpeg calls
        /// </summary>
        /// <param name="backend">Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.</param>
        /// <param name="dev_allocator">Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)</param>
        /// <param name="pinned_allocator">Pointer to nvjpegPinnedAllocator. If NULL - use default cuda calls (cudaHostAlloc/cudaFreeHost)</param>
        /// <param name="flags">Parameters for the operation. Must be 0.</param>
        /// <param name="handle">Codec instance, use for other calls</param>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegCreateEx(nvjpegBackend backend,
                IntPtr dev_allocator,
                ref nvjpegPinnedAllocator pinned_allocator,
                uint flags,
                ref nvjpegHandle handle);

        /// <summary>
        /// Initalization of nvjpeg handle with additional parameters. This handle is used for all consecutive nvjpeg calls
        /// </summary>
        /// <param name="backend">Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.</param>
        /// <param name="dev_allocator">Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)</param>
        /// <param name="pinned_allocator">Pointer to nvjpegPinnedAllocator. If NULL - use default cuda calls (cudaHostAlloc/cudaFreeHost)</param>
        /// <param name="flags">Parameters for the operation. Must be 0.</param>
        /// <param name="handle">Codec instance, use for other calls</param>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegCreateEx(nvjpegBackend backend,
                ref nvjpegDevAllocator dev_allocator,
                IntPtr pinned_allocator,
                uint flags,
                ref nvjpegHandle handle);

        /// <summary>
        /// Initalization of nvjpeg handle with additional parameters. This handle is used for all consecutive nvjpeg calls
        /// </summary>
        /// <param name="backend">Backend to use. Currently Default or Hybrid (which is the same at the moment) is supported.</param>
        /// <param name="dev_allocator">Pointer to nvjpegDevAllocator. If NULL - use default cuda calls (cudaMalloc/cudaFree)</param>
        /// <param name="pinned_allocator">Pointer to nvjpegPinnedAllocator. If NULL - use default cuda calls (cudaHostAlloc/cudaFreeHost)</param>
        /// <param name="flags">Parameters for the operation. Must be 0.</param>
        /// <param name="handle">Codec instance, use for other calls</param>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegCreateEx(nvjpegBackend backend,
                IntPtr dev_allocator,
                IntPtr pinned_allocator,
                uint flags,
                ref nvjpegHandle handle);


        /// <summary>
        /// Release the handle and resources.
        /// </summary>
        /// <param name="handle">instance handle to release</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDestroy(nvjpegHandle handle);

        
        /// <summary>
        /// Sets padding for device memory allocations. After success on this call any device memory allocation would be padded to the multiple of specified number of bytes. 
        /// </summary>
        /// <param name="padding">padding size</param>
        /// <param name="handle">instance handle to release </param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegSetDeviceMemoryPadding(SizeT padding, nvjpegHandle handle);

        
        /// <summary>
        /// Retrieves padding for device memory allocations
        /// </summary>
        /// <param name="padding">padding size currently used in handle.</param>
        /// <param name="handle">instance handle to release </param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegGetDeviceMemoryPadding(ref SizeT padding, nvjpegHandle handle);


        /// <summary>
        /// Sets padding for pinned host memory allocations. After success on this call any pinned host memory allocation would be padded to the multiple of specified number of bytes. 
        /// </summary>
        /// <param name="padding">padding size</param>
        /// <param name="handle">instance handle to release</param>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegSetPinnedMemoryPadding(SizeT padding, nvjpegHandle handle);


        /// <summary>
        /// Retrieves padding for pinned host memory allocations
        /// </summary>
        /// <param name="padding">padding size currently used in handle.</param>
        /// <param name="handle">instance handle to release</param>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegGetPinnedMemoryPadding(ref SizeT padding, nvjpegHandle handle);


        /// <summary>
        /// Initalization of decoding state
        /// </summary>
        /// <param name="handle">Library handle</param>
        /// <param name="jpeg_handle">Decoded jpeg image state handle</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStateCreate(nvjpegHandle handle, ref nvjpegJpegState jpeg_handle);


        /// <summary>
        /// Release the jpeg image handle.
        /// </summary>
        /// <param name="jpeg_handle">Decoded jpeg image state handle</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStateDestroy(nvjpegJpegState jpeg_handle);

        
        /// <summary>
        /// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.<para/>
        /// If less than NVJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information<para/>
        /// If the image is 3-channel, all three groups are valid.<para/>
        /// This function is thread safe.
        /// </summary>
        /// <param name="handle">Library handle</param>
        /// <param name="data">Pointer to the buffer containing the jpeg stream data to be decoded. </param>
        /// <param name="length">Length of the jpeg image buffer.</param>
        /// <param name="nComponents">Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.</param>
        /// <param name="subsampling">Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling</param>
        /// <param name="widths">pointer to NVJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded  </param>
        /// <param name="heights">pointer to NVJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded </param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegGetImageInfo(nvjpegHandle handle, IntPtr data, SizeT length,
                ref int nComponents, ref nvjpegChromaSubsampling subsampling, int[] widths, int[] heights);


        /// <summary>
        /// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.<para/>
        /// If less than NVJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information<para/>
        /// If the image is 3-channel, all three groups are valid.<para/>
        /// This function is thread safe.
        /// </summary>
        /// <param name="handle">Library handle</param>
        /// <param name="data">Pointer to the buffer containing the jpeg stream data to be decoded. </param>
        /// <param name="length">Length of the jpeg image buffer.</param>
        /// <param name="nComponents">Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.</param>
        /// <param name="subsampling">Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling</param>
        /// <param name="widths">pointer to NVJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded  </param>
        /// <param name="heights">pointer to NVJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded </param>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegGetImageInfo(nvjpegHandle handle, byte[] data, SizeT length,
            ref int nComponents, ref nvjpegChromaSubsampling subsampling, int[] widths, int[] heights);


        /// <summary>
        /// Decodes single image. Destination buffers should be large enough to be able to store 
        /// output of specified format. For each color plane sizes could be retrieved for image using nvjpegGetImageInfo()
        /// and minimum required memory buffer for each plane is nPlaneHeight*nPlanePitch where nPlanePitch >= nPlaneWidth for
        /// planar output formats and nPlanePitch >= nPlaneWidth*nOutputComponents for interleaved output format.
        /// </summary>
        /// <param name="handle">Library handle</param>
        /// <param name="jpeg_handle">Decoded jpeg image state handle</param>
        /// <param name="data">Pointer to the buffer containing the jpeg image to be decoded.</param>
        /// <param name="length">Length of the jpeg image buffer.</param>
        /// <param name="output_format">Output data format. See nvjpegOutputFormat for description</param>
        /// <param name="destination">Pointer to structure with information about output buffers. See nvjpegImage description.</param>
        /// <param name="stream">CUDA stream where to submit all GPU work</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecode(nvjpegHandle handle, nvjpegJpegState jpeg_handle, IntPtr data,
                SizeT length, nvjpegOutputFormat output_format, ref nvjpegImage destination, CUstream stream);



        //////////////////////////////////////////////
        /////////////// Batch decoding ///////////////
        //////////////////////////////////////////////

        /// <summary>
        /// Resets and initizlizes batch decoder for working on the batches of specified size
        /// Should be called once for decoding bathes of this specific size, also use to reset failed batches
        /// </summary>
        /// <param name="handle">Library handle</param>
        /// <param name="jpeg_handle">Decoded jpeg image state handle</param>
        /// <param name="batch_size">Size of the batch</param>
        /// <param name="max_cpuhreads">Maximum number of CPU threads that will be processing this batch</param>
        /// <param name="output_format">Output data format. Will be the same for every image in batch</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeBatchedInitialize(
                  nvjpegHandle handle,
                  nvjpegJpegState jpeg_handle,
                  int batch_size,
                  int max_cpuhreads,
                  nvjpegOutputFormat output_format);


        /// <summary>
        /// Decodes batch of images. Output buffers should be large enough to be able to store 
        /// outputs of specified format, see single image decoding description for details. Call to 
        /// nvjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as 
        /// parameter to this batch initialization function.
        /// </summary>
        /// <param name="handle">Library handle</param>
        /// <param name="jpeg_handle">Decoded jpeg image state handle</param>
        /// <param name="data">Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded.</param>
        /// <param name="lengths">Array of size batch_size with lengths of the jpeg images' buffers in the batch.</param>
        /// <param name="destinations">Array of size batch_size with pointers to structure with information about output buffers</param>
        /// <param name="stream">CUDA stream where to submit all GPU work</param>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeBatched(
                    nvjpegHandle handle,
                    nvjpegJpegState jpeg_handle,
                    IntPtr[] data,
                    SizeT[] lengths,
                    nvjpegImage[] destinations,
                    CUstream stream);

        /**********************************************************
        *                        Compression                      *
        **********************************************************/


        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderStateCreate(
                nvjpegHandle handle,
                ref nvjpegEncoderState encoder_state,
                CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderStateDestroy(nvjpegEncoderState encoder_state);



        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsCreate(
                nvjpegHandle handle,
                ref nvjpegEncoderParams encoder_params,
                CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsDestroy(nvjpegEncoderParams encoder_params);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsSetQuality(
                nvjpegEncoderParams encoder_params,
         int quality,
        CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsSetEncoding(
                nvjpegEncoderParams encoder_params,
                nvjpegJpegEncoding etype,
                CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsSetOptimizedHuffman(
                nvjpegEncoderParams encoder_params,
         int optimized,
        CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsSetSamplingFactors(
                nvjpegEncoderParams encoder_params,
                nvjpegChromaSubsampling chroma_subsampling,
                CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncodeGetBufferSize(
                nvjpegHandle handle,
                nvjpegEncoderParams encoder_params,
                int image_width,
                int image_height,
                ref SizeT max_stream_length);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncodeYUV(
                nvjpegHandle handle,
                nvjpegEncoderState encoder_state,
                nvjpegEncoderParams encoder_params,
                ref nvjpegImage source,
                nvjpegChromaSubsampling chroma_subsampling,
                int image_width,
                int image_height,
                CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncodeImage(
                nvjpegHandle handle,
                nvjpegEncoderState encoder_state,
                nvjpegEncoderParams encoder_params,
                ref nvjpegImage source,
                nvjpegInputFormat input_format,
                int image_width,
                int image_height,
                CUstream stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncodeRetrieveBitstream(
                nvjpegHandle handle,
                nvjpegEncoderState encoder_state,
                IntPtr data,
                ref SizeT length,
                CUstream stream);
        


        ///////////////////////////////////////////////////////////////////////////////////
        // API v2 //
        ///////////////////////////////////////////////////////////////////////////////////


        ///////////////////////////////////////////////////////////////////////////////////
        // NVJPEG buffers //
        ///////////////////////////////////////////////////////////////////////////////////


        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegBufferPinnedCreate(nvjpegHandle handle,
            ref nvjpegPinnedAllocator pinned_allocator,
            ref nvjpegBufferPinned buffer);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegBufferPinnedDestroy(nvjpegBufferPinned buffer);


        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegBufferDeviceCreate(nvjpegHandle handle,
            ref nvjpegDevAllocator device_allocator,
            ref nvjpegBufferDevice buffer);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegBufferDeviceDestroy(nvjpegBufferDevice buffer);


        /// <summary>
        /// retrieve buffer size and pointer - this allows reusing buffer when decode is not needed
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegBufferPinnedRetrieve(nvjpegBufferPinned buffer, ref SizeT size, ref IntPtr ptr);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegBufferDeviceRetrieve(nvjpegBufferDevice buffer, ref SizeT size, ref CUdeviceptr ptr);

        
        /// <summary>
        /// this allows attaching same memory buffers to different states, allowing to switch implementations without allocating extra memory
        /// </summary>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegStateAttachPinnedBuffer(nvjpegJpegState decoder_state,
            nvjpegBufferPinned pinned_buffer);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegStateAttachDeviceBuffer(nvjpegJpegState decoder_state,
            nvjpegBufferDevice device_buffer);

        ///////////////////////////////////////////////////////////////////////////////////
        // JPEG stream parameters //
        ///////////////////////////////////////////////////////////////////////////////////


        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamCreate(
            nvjpegHandle handle,
            ref nvjpegJpegStream jpeg_stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamDestroy(nvjpegJpegStream jpeg_stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamParse(
            nvjpegHandle handle,
            IntPtr data,
            SizeT length,
            int save_metadata,
            int save_stream,
            nvjpegJpegStream jpeg_stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamParse(
            nvjpegHandle handle,
            byte[] data,
            SizeT length,
            int save_metadata,
            int save_stream,
            nvjpegJpegStream jpeg_stream);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamGetJpegEncoding(
            nvjpegJpegStream jpeg_stream,
            ref nvjpegJpegEncoding jpeg_encoding);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamGetFrameDimensions(
            nvjpegJpegStream jpeg_stream,
            ref uint width,
            ref uint height);

        /// <summary>
        /// </summary>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamGetComponentsNum(
            nvjpegJpegStream jpeg_stream,
            ref uint components_num);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamGetComponentDimensions(
            nvjpegJpegStream jpeg_stream,
            uint component,
            ref uint width,
            ref uint height);


        /// <summary>
        /// if encoded is 1 color component then it assumes 4:0:0 (NVJPEG_CSS_GRAY, grayscale)<para/>
        /// if encoded is 3 color components it tries to assign one of the known subsamplings based on the components subsampling infromation<para/>
        /// in case sampling factors are not stadard or number of components is different it will return NVJPEG_CSS_UNKNOWN<para/>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegJpegStreamGetChromaSubsampling(
            nvjpegJpegStream jpeg_stream,
            ref nvjpegChromaSubsampling chroma_subsampling);

        ///////////////////////////////////////////////////////////////////////////////////
        // Decode parameters //
        ///////////////////////////////////////////////////////////////////////////////////


        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeParamsCreate(
            nvjpegHandle handle,
            ref nvjpegDecodeParams decode_params);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeParamsDestroy(nvjpegDecodeParams decode_params);


        /// <summary>
        /// set output pixel format - same value as in nvjpegDecode()
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeParamsSetOutputFormat(
            nvjpegDecodeParams decode_params,
            nvjpegOutputFormat output_format);


        /// <summary>
        /// set to desired ROI. set to (0, 0, -1, -1) to disable ROI decode (decode whole image)
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeParamsSetROI(
            nvjpegDecodeParams decode_params,
            int offset_x, int offset_y, int roi_width, int roi_height);


        /// <summary>
        /// set to true to allow conversion from CMYK to RGB or YUV that follows simple subtractive scheme
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeParamsSetAllowCMYK(
            nvjpegDecodeParams decode_params,
            int allow_cmyk);

        ///////////////////////////////////////////////////////////////////////////////////
        // Decoder helper functions //
        ///////////////////////////////////////////////////////////////////////////////////


        
        /// <summary>
        /// creates decoder implementation
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecoderCreate(nvjpegHandle nvjpeg_handle,
            nvjpegBackend implementation,
            ref nvjpegJpegDecoder decoder_handle);

        /// <summary>
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecoderDestroy(nvjpegJpegDecoder decoder_handle);

        
        /// <summary>
        /// on return sets is_supported value to 0 if decoder is capable to handle jpeg_stream 
        /// with specified decode parameters
        /// </summary>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecoderJpegSupported(nvjpegJpegDecoder decoder_handle,
            nvjpegJpegStream jpeg_stream,
            nvjpegDecodeParams decode_params,
            ref int is_supported);

         
        /// <summary>
        /// creates decoder state 
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecoderStateCreate(nvjpegHandle nvjpeg_handle,
            nvjpegJpegDecoder decoder_handle,
            ref nvjpegJpegState decoder_state);

        ///////////////////////////////////////////////////////////////////////////////////
        // Decode functions //
        ///////////////////////////////////////////////////////////////////////////////////
        
        /// <summary>
        /// starts decoding on host and save decode parameters to the state
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeJpegHost(
            nvjpegHandle handle,
            nvjpegJpegDecoder decoder,
            nvjpegJpegState decoder_state,
            nvjpegDecodeParams decode_params,
            nvjpegJpegStream jpeg_stream);

        
        /// <summary>
        /// hybrid stage of decoding image,  involves device async calls
        /// note that jpeg stream is a parameter here - because we still might need copy parts of bytestream to device
        /// </summary>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeJpegTransferToDevice(
            nvjpegHandle handle,
            nvjpegJpegDecoder decoder,
            nvjpegJpegState decoder_state,
            nvjpegJpegStream jpeg_stream,
            CUstream stream);

        
        /// <summary>
        /// finishing async operations on the device
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegDecodeJpegDevice(
            nvjpegHandle handle,
            nvjpegJpegDecoder decoder,
            nvjpegJpegState decoder_state,
            ref nvjpegImage destination,
            CUstream stream);

        ///////////////////////////////////////////////////////////////////////////////////
        // JPEG Transcoding Functions //
        ///////////////////////////////////////////////////////////////////////////////////

        
        /// <summary>
        /// copies metadata (JFIF, APP, EXT, COM markers) from parsed stream
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsCopyMetadata(
            nvjpegEncoderState encoder_state,
            nvjpegEncoderParams encode_params,
            nvjpegJpegStream jpeg_stream,
            CUstream stream);

        
        /// <summary>
        /// copies quantization tables from parsed stream
		/// </summary>
		[DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsCopyQuantizationTables(
            nvjpegEncoderParams encode_params,
            nvjpegJpegStream jpeg_stream,
            CUstream stream);


        /// <summary>
        /// copies huffman tables from parsed stream. should require same scans structure
        /// </summary>
        [DllImport(NVJPEG_API_DLL_NAME)]
        public static extern nvjpegStatus nvjpegEncoderParamsCopyHuffmanTables(
            nvjpegEncoderState encoder_state,
            nvjpegEncoderParams encode_params,
            nvjpegJpegStream jpeg_stream,
            CUstream stream);



    }
}
