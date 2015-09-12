//	Copyright (c) 2012, Michael Kunz. All rights reserved.
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

namespace ManagedCuda
{
    /// <summary>
    /// OpenGL Interoperability
    /// </summary>
    public static class OpenGLNativeMethods
    {        
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";
       
        /// <summary>
        /// OpenGL Interoperability for CUDA >3.x
        /// </summary>
        public static class CUDA3
        {
            /// <summary>
            /// Creates a new CUDA context, initializes OpenGL interoperability, and associates the CUDA context with the calling
            /// thread. It must be called before performing any other OpenGL interoperability operations. It may fail if the needed
			/// OpenGL driver facilities are not available. For usage of the Flags parameter, see <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2"/>.
            /// </summary>
            /// <param name="pCtx">Returned CUDA context</param>
            /// <param name="Flags">Options for CUDA context creation</param>
            /// <param name="device">Device on which to create the context</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint="cuGLCtxCreate_v2")]
            public static extern CUResult cuGLCtxCreate(ref CUcontext pCtx, CUCtxFlags Flags, CUdevice device);

            /// <summary>
            /// Registers the buffer object specified by buffer for access by CUDA. A handle to the registered object is returned as
            /// <c>pCudaResource</c>. The map flags <c>Flags</c> specify the intended usage.
            /// </summary>
            /// <param name="pCudaResource">Pointer to the returned object handle</param>
            /// <param name="buffer">name of buffer object to be registered</param>
            /// <param name="Flags">Map flags</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, 
            /// <see cref="CUResult.ErrorAlreadyMapped"/>, <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsGLRegisterBuffer(ref CUgraphicsResource pCudaResource, uint buffer, CUGraphicsRegisterFlags Flags);

            /// <summary>
            /// Registers the texture or renderbuffer object specified by <c>image</c> for access by CUDA. <c>target</c> must match the type
            /// of the object. A handle to the registered object is returned as <c>pCudaResource</c>. The map flags Flags specify the
            /// intended usage. <para/>
            /// The following image classes are currently disallowed: <para/>
            /// • Textures with borders <para/>
            /// • Multisampled renderbuffers
            /// </summary>
            /// <param name="pCudaResource">Pointer to the returned object handle</param>
            /// <param name="image">name of texture or renderbuffer object to be registered</param>
            /// <param name="target">Identifies the type of object specified by <c>image</c>, and must be one of <c>GL_TEXTURE_2D</c>,
            /// <c>GL_TEXTURE_RECTANGLE</c>, <c>GL_TEXTURE_CUBE_MAP</c>, 
            /// <c>GL_TEXTURE_3D</c>, <c>GL_TEXTURE_2D_ARRAY</c>, or <c>GL_RENDERBUFFER</c>.</param>
            /// <param name="Flags">Map flags</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, 
            /// <see cref="CUResult.ErrorAlreadyMapped"/>, <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsGLRegisterImage(ref CUgraphicsResource pCudaResource, uint image, CudaOpenGLImageInteropResource.OpenGLImageTarget target, CUGraphicsRegisterFlags Flags);

            /// <summary>
            /// Returns in <c>pDevice</c> the CUDA device associated with a <c>hGpu</c>, if applicable.
            /// </summary>
            /// <param name="pDevice">Device associated with hGpu</param>
            /// <param name="hGpu">Handle to a GPU, as queried via <c>WGL_NV_gpu_affinity()</c></param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuWGLGetDevice(ref CUdevice pDevice, IntPtr hGpu);
				

            /// <summary>
            /// Gets the CUDA devices associated with the current OpenGL context.<para/>
			/// Returns in pCudaDeviceCount the number of CUDA-compatible devices 
			/// corresponding to the current OpenGL context. Also returns in pCudaDevices 
			/// at most cudaDeviceCount of the CUDA-compatible devices corresponding to 
			/// the current OpenGL context. If any of the GPUs being used by the current OpenGL
			/// context are not CUDA capable then the call will return CUDA_ERROR_NO_DEVICE.
            /// </summary>
			/// <param name="pCudaDeviceCount">Returned number of CUDA devices.</param>
			/// <param name="pCudaDevices">Returned CUDA devices.</param>
			/// <param name="cudaDeviceCount">The size of the output device array pCudaDevices.</param>
			/// <param name="deviceList">The set of devices to return.</param>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuGLGetDevices_v2")]
            public static extern CUResult cuGLGetDevices(ref uint pCudaDeviceCount, CUdevice[] pCudaDevices, uint cudaDeviceCount, CUGLDeviceList deviceList);


        }
    }
}
