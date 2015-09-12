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
    /// Direct3D 9 Interoperability
    /// </summary>
    public static class DirectX9NativeMethods
    {
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";

        /// <summary>
        /// Direct3D9 Interoperability for CUDA 3.x
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class CUDA3
        {
            /// <summary>
            /// Returns in <c>pCudaDevice</c> the CUDA-compatible device corresponding to the adapter name <c>pszAdapterName</c>
            /// obtained from <c>EnumDisplayDevices()</c> or <c>IDirect3D9::GetAdapterIdentifier()</c>.
            /// If no device on the adapter with name <c>pszAdapterName</c> is CUDA-compatible, then the call will fail.
            /// </summary>
            /// <param name="pCudaDevice">Returned CUDA device corresponding to pszAdapterName</param>
            /// <param name="pszAdapterName">Adapter name to query for device</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9GetDevice(ref CUdevice pCudaDevice, string pszAdapterName);

            /// <summary>
            /// Gets the CUDA devices corresponding to a Direct3D 9 device<para/>
            /// Returns in <c>pCudaDeviceCount</c> the number of CUDA-compatible device corresponding
            /// to the Direct3D 9 device <c>pD3D9Device</c>.
            /// Also returns in <c>pCudaDevices</c> at most <c>cudaDeviceCount</c> of the the CUDA-compatible devices
            /// corresponding to the Direct3D 9 device <c>pD3D9Device</c>.
            /// <para/>
            /// If any of the GPUs being used to render <c>pDevice</c> are not CUDA capable then the
            /// call will return <see cref="CUResult.ErrorNoDevice"/>.
            /// </summary>
            /// <param name="pCudaDeviceCount">Returned number of CUDA devices corresponding to <c>pD3D9Device</c></param>
            /// <param name="pCudaDevices">Returned CUDA devices corresponding to <c>pD3D9Device</c></param>
            /// <param name="cudaDeviceCount">The size of the output device array <c>pCudaDevices</c></param>
            /// <param name="pD3D9Device">Direct3D 9 device to query for CUDA devices</param>
            /// <param name="deviceList">The set of devices to return.</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorNoDevice"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9GetDevices(ref int pCudaDeviceCount, [Out] CUdevice[] pCudaDevices, uint cudaDeviceCount, IntPtr pD3D9Device, CUd3dXDeviceList deviceList);
            /// <summary>
            /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
            /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
            /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
            /// If <c>pCudaDevice</c> is non-NULL then the <see cref="CUdevice"/> on which this CUDA context was created will be returned in
            /// <c>pCudaDevice</c>.
            /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
            /// upon destruction of this context through <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
            /// is destroyed or encounters an error.
            /// </summary>
            /// <param name="pCtx">Returned newly created CUDA context</param>
            /// <param name="pCudaDevice">Returned pointer to the device on which the context was created</param>
            /// <param name="Flags">Context creation flags (see <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2"/> for details)</param>
            /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint="cuD3D9CtxCreate_v2")]
            public static extern CUResult cuD3D9CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);
            
            /// <summary>
            /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
            /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
            /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
            /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
            /// upon destruction of this context through <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
            /// is destroyed or encounters an error.
            /// </summary>
            /// <param name="pCtx">Returned newly created CUDA context</param>
            /// <param name="flags">Context creation flags (see <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2"/> for details)</param>
            /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
            /// <param name="cudaDevice">Returned pointer to the device on which the context was created</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9CtxCreateOnDevice(ref CUcontext pCtx, CUCtxFlags flags, IntPtr pD3DDevice, CUdevice cudaDevice);
            

            /// <summary>
            /// Registers the Direct3D 9 resource <c>pD3DResource</c> for access by CUDA and returns a CUDA handle to
            /// <c>pD3Dresource</c> in <c>pCudaResource</c>. The handle returned in <c>pCudaResource</c> may be used to map and
            /// unmap this resource until it is unregistered. On success this call will increase the internal reference count on
            /// <c>pD3DResource</c>. This reference count will be decremented when this resource is unregistered through <see cref="ManagedCuda.DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnregisterResource"/>.<para/>
            /// This call is potentially high-overhead and should not be called every frame in interactive applications.<para/>
            /// The type of pD3DResource must be one of the following:
            /// <list type="table">  
            /// <listheader><term>Type of <c>pD3DResource</c></term><description>Restriction</description></listheader>  
            /// <item><term>IDirect3DVertexBuffer9</term><description>
            /// May be accessed through a device pointer.
            /// </description></item>  
            /// <item><term>IDirect3DIndexBuffer9</term><description>
            /// May be accessed through a device pointer.
            /// </description></item>  
            /// <item><term>IDirect3DSurface9</term><description>
            /// May be accessed through an array. Only stand-alone objects of type <c>IDirect3DSurface9</c>
            /// may be explicitly shared. In particular, individual mipmap levels and faces of cube maps may not be registered
            /// directly. To access individual surfaces associated with a texture, one must register the base texture object.
            /// </description></item>  
            /// <item><term>IDirect3DBaseTexture9</term><description>
            /// Individual surfaces on this texture may be accessed through an array.
            /// </description></item> 
            /// </list> 
            /// The Flags argument may be used to specify additional parameters at register time. The only valid value for this
            /// parameter is <see cref="CUGraphicsRegisterFlags.None"/>. <para/>
            /// Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some
            /// limitations.<param/>
            /// • The primary rendertarget may not be registered with CUDA.<param/>
            /// • Resources allocated as shared may not be registered with CUDA.<param/>
            /// • Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data
            /// cannot be shared.<param/>
            /// • Surfaces of depth or stencil formats cannot be shared.<param/>
            /// If Direct3D interoperability is not initialized for this context using <see cref="cuD3D9CtxCreate"/> then
            /// <see cref="CUResult.ErrorInvalidContext"/> is returned. If <c>pD3DResource</c> is of incorrect type or is already registered then
            /// <see cref="CUResult.ErrorInvalidHandle"/> is returned. If <c>pD3DResource</c> cannot be registered then 
            /// <see cref="CUResult.ErrorUnknown"/> is returned. If <c>Flags</c> is not one of the above specified value then <see cref="CUResult.ErrorInvalidValue"/>
            /// is returned.
            /// </summary>
            /// <param name="pCudaResource">Returned graphics resource handle</param>
            /// <param name="pD3DResource">Direct3D resource to register</param>
            /// <param name="Flags">Parameters for resource registration</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsD3D9RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);
            
            /// <summary>
            /// Returns in <c>ppD3DDevice</c> the Direct3D device against which this CUDA context
            /// was created in <see cref="cuD3D9CtxCreate"/>.
            /// </summary>
            /// <param name="ppD3DDevice">Returned Direct3D device corresponding to CUDA context</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D9GetDirect3DDevice(ref IntPtr ppD3DDevice);
        }
    }

    /// <summary>
    /// Direct3D 10 Interoperability
    /// </summary>
    public static class DirectX10NativeMethods
    {
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";

        /// <summary>
        /// Direct3D10 Interoperability for CUDA 3.x
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class CUDA3
        {
            /// <summary>
            /// Returns in <c>device</c> the CUDA-compatible device corresponding to the adapter <c>pAdapter</c> obtained from 
            /// <c>IDXGIFactory::EnumAdapters</c>. This call will succeed only if a device on adapter <c>pAdapter</c> is Cuda-compatible.
            /// </summary>
            /// <param name="device">Returned CUDA device corresponding to pszAdapterName</param>
            /// <param name="pAdapter">Adapter (type: IDXGIAdapter)</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10GetDevice(ref CUdevice device, IntPtr pAdapter);

            /// <summary>
            /// Gets the CUDA devices corresponding to a Direct3D 10 device<para/>
            /// Returns in <c>pCudaDeviceCount</c> the number of CUDA-compatible device corresponding
            /// to the Direct3D 10 device <c>pD3D10Device</c>.
            /// Also returns in <c>pCudaDevices</c> at most <c>cudaDeviceCount</c> of the the CUDA-compatible devices
            /// corresponding to the Direct3D 10 device <c>pD3D10Device</c>.
            /// <para/>
            /// If any of the GPUs being used to render <c>pDevice</c> are not CUDA capable then the
            /// call will return <see cref="CUResult.ErrorNoDevice"/>.
            /// </summary>
            /// <param name="pCudaDeviceCount">Returned number of CUDA devices corresponding to <c>pD3D9Device</c></param>
            /// <param name="pCudaDevices">Returned CUDA devices corresponding to <c>pD3D9Device</c></param>
            /// <param name="cudaDeviceCount">The size of the output device array <c>pCudaDevices</c></param>
            /// <param name="pD3D10Device">Direct3D 10 device to query for CUDA devices</param>
            /// <param name="deviceList">The set of devices to return.</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorNoDevice"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10GetDevices(ref int pCudaDeviceCount, [Out] CUdevice[] pCudaDevices, uint cudaDeviceCount, IntPtr pD3D10Device, CUd3dXDeviceList deviceList);
            /// <summary>
            /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
            /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
            /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
            /// If <c>pCudaDevice</c> is non-NULL then the <see cref="CUdevice"/> on which this CUDA context was created will be returned in
            /// <c>pCudaDevice</c>.
            /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
            /// upon destruction of this context through <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
            /// is destroyed or encounters an error.
            /// </summary>
            /// <param name="pCtx">Returned newly created CUDA context</param>
            /// <param name="pCudaDevice">Returned pointer to the device on which the context was created</param>
            /// <param name="Flags">Context creation flags (see <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2"/> for details)</param>
            /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint="cuD3D10CtxCreate_v2")]
            public static extern CUResult cuD3D10CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);


            /// <summary>
            /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
            /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
            /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
            /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
            /// upon destruction of this context through <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
            /// is destroyed or encounters an error.
            /// </summary>
            /// <param name="pCtx">Returned newly created CUDA context</param>
            /// <param name="flags">Context creation flags (see <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2"/> for details)</param>
            /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
            /// <param name="cudaDevice">Returned pointer to the device on which the context was created</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10CtxCreateOnDevice(ref CUcontext pCtx, CUCtxFlags flags, IntPtr pD3DDevice, CUdevice cudaDevice);
            
            /// <summary>
            /// Registers the Direct3D 10 resource <c>pD3DResource</c> for access by CUDA and returns a CUDA handle to
            /// <c>pD3Dresource</c> in <c>pCudaResource</c>. The handle returned in <c>pCudaResource</c> may be used to map and
            /// unmap this resource until it is unregistered. On success this call will increase the internal reference count on
            /// <c>pD3DResource</c>. This reference count will be decremented when this resource is unregistered through <see cref="ManagedCuda.DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnregisterResource"/>.<para/>
            /// This call is potentially high-overhead and should not be called every frame in interactive applications.<para/>
            /// The type of pD3DResource must be one of the following:
            /// <list type="table">  
            /// <listheader><term>Type of <c>pD3DResource</c></term><description>Restriction</description></listheader>  
            /// <item><term>ID3D10Buffer</term><description>
            /// May be accessed through a device pointer.
            /// </description></item>  
            /// <item><term>ID3D10Texture1D</term><description>
            /// Individual subresources of the texture may be accessed via arrays.
            /// </description></item>  
            /// <item><term>ID3D10Texture2D</term><description>
            /// Individual subresources of the texture may be accessed via arrays.
            /// </description></item> 
            /// <item><term>ID3D10Texture3D</term><description>
            /// Individual subresources of the texture may be accessed via arrays.
            /// </description></item>  
            /// </list> 
            /// The Flags argument may be used to specify additional parameters at register time. The only valid value for this
            /// parameter is <see cref="CUGraphicsRegisterFlags.None"/>. <para/>
            /// Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some
            /// limitations.<param/>
            /// • The primary rendertarget may not be registered with CUDA.<param/>
            /// • Resources allocated as shared may not be registered with CUDA.<param/>
            /// • Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data
            /// cannot be shared.<param/>
            /// • Surfaces of depth or stencil formats cannot be shared.<param/>
            /// If Direct3D interoperability is not initialized for this context using <see cref="cuD3D10CtxCreate"/> then
            /// <see cref="CUResult.ErrorInvalidContext"/> is returned. If <c>pD3DResource</c> is of incorrect type or is already registered then
            /// <see cref="CUResult.ErrorInvalidHandle"/> is returned. If <c>pD3DResource</c> cannot be registered then 
            /// <see cref="CUResult.ErrorUnknown"/> is returned. If <c>Flags</c> is not one of the above specified value then <see cref="CUResult.ErrorInvalidValue"/>
            /// is returned.
            /// </summary>
            /// <param name="pCudaResource">Returned graphics resource handle</param>
            /// <param name="pD3DResource">Direct3D resource to register</param>
            /// <param name="Flags">Parameters for resource registration</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsD3D10RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);

            /// <summary>
            /// Returns in <c>ppD3DDevice</c> the Direct3D device against which this CUDA context
            /// was created in <see cref="cuD3D10CtxCreate"/>.
            /// </summary>
            /// <param name="ppD3DDevice">Returned Direct3D device corresponding to CUDA context</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuD3D10GetDirect3DDevice(ref IntPtr ppD3DDevice);
        }
    }

    /// <summary>
    /// Direct3D 11 Interoperability for CUDA 3.x
    /// </summary>
    [System.Security.SuppressUnmanagedCodeSecurityAttribute]
    public static class DirectX11NativeMethods
    {
        internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";

        /// <summary>
        /// Returns in <c>device</c> the CUDA-compatible device corresponding to the adapter <c>pAdapter</c> obtained from 
        /// <c>IDXGIFactory::EnumAdapters</c>. This call will succeed only if a device on adapter <c>pAdapter</c> is Cuda-compatible.
        /// </summary>
        /// <param name="device">Returned CUDA device corresponding to pszAdapterName</param>
        /// <param name="pAdapter">Adapter (type: IDXGIAdapter)</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuD3D11GetDevice(ref CUdevice device, IntPtr pAdapter);

        /// <summary>
        /// Gets the CUDA devices corresponding to a Direct3D 11 device<para/>
        /// Returns in <c>pCudaDeviceCount</c> the number of CUDA-compatible device corresponding
        /// to the Direct3D 11 device <c>pD3D11Device</c>.
        /// Also returns in <c>pCudaDevices</c> at most <c>cudaDeviceCount</c> of the the CUDA-compatible devices
        /// corresponding to the Direct3D 11 device <c>pD3D11Device</c>.
        /// <para/>
        /// If any of the GPUs being used to render <c>pDevice</c> are not CUDA capable then the
        /// call will return <see cref="CUResult.ErrorNoDevice"/>.
        /// </summary>
        /// <param name="pCudaDeviceCount">Returned number of CUDA devices corresponding to <c>pD3D9Device</c></param>
        /// <param name="pCudaDevices">Returned CUDA devices corresponding to <c>pD3D11Device</c></param>
        /// <param name="cudaDeviceCount">The size of the output device array <c>pCudaDevices</c></param>
        /// <param name="pD3D11Device">Direct3D 11 device to query for CUDA devices</param>
        /// <param name="deviceList">The set of devices to return.</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorNoDevice"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuD3D11GetDevices(ref int pCudaDeviceCount, [Out] CUdevice[] pCudaDevices, uint cudaDeviceCount, IntPtr pD3D11Device, CUd3dXDeviceList deviceList);
           
        /// <summary>
        /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
        /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
        /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
        /// If <c>pCudaDevice</c> is non-NULL then the <see cref="CUdevice"/> on which this CUDA context was created will be returned in
        /// <c>pCudaDevice</c>.
        /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
        /// upon destruction of this context through <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
        /// is destroyed or encounters an error.
        /// </summary>
        /// <param name="pCtx">Returned newly created CUDA context</param>
        /// <param name="pCudaDevice">Returned pointer to the device on which the context was created</param>
		/// <param name="Flags">Context creation flags (see <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2"/> for details)</param>
        /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint="cuD3D11CtxCreate_v2")]
        public static extern CUResult cuD3D11CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);

        /// <summary>
        /// Creates a new CUDA context, enables interoperability for that context with the Direct3D device <c>pD3DDevice</c>, and
        /// associates the created CUDA context with the calling thread. The created <see cref="CUcontext"/> will be returned in <c>pCtx</c>.
        /// Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.
        /// On success, this call will increase the internal reference count on <c>pD3DDevice</c>. This reference count will be decremented
        /// upon destruction of this context through <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxDestroy"/>. This context will cease to function if <c>pD3DDevice</c>
        /// is destroyed or encounters an error.
        /// </summary>
        /// <param name="pCtx">Returned newly created CUDA context</param>
		/// <param name="flags">Context creation flags (see <see cref="ManagedCuda.DriverAPINativeMethods.ContextManagement.cuCtxCreate_v2"/> for details)</param>
        /// <param name="pD3DDevice">Direct3D device to create interoperability context with</param>
        /// <param name="cudaDevice">Returned pointer to the device on which the context was created</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuD3D11CtxCreateOnDevice(ref CUcontext pCtx, CUCtxFlags flags, IntPtr pD3DDevice, CUdevice cudaDevice);
        
        /// <summary>
        /// Registers the Direct3D 11 resource <c>pD3DResource</c> for access by CUDA and returns a CUDA handle to
        /// <c>pD3Dresource</c> in <c>pCudaResource</c>. The handle returned in <c>pCudaResource</c> may be used to map and
        /// unmap this resource until it is unregistered. On success this call will increase the internal reference count on
        /// <c>pD3DResource</c>. This reference count will be decremented when this resource is unregistered through <see cref="ManagedCuda.DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnregisterResource"/>.<para/>
        /// This call is potentially high-overhead and should not be called every frame in interactive applications.<para/>
        /// The type of pD3DResource must be one of the following:
        /// <list type="table">  
        /// <listheader><term>Type of <c>pD3DResource</c></term><description>Restriction</description></listheader>  
        /// <item><term>ID3D11Buffer</term><description>
        /// May be accessed through a device pointer.
        /// </description></item>  
        /// <item><term>ID3D11Texture1D</term><description>
        /// Individual subresources of the texture may be accessed via arrays.
        /// </description></item>  
        /// <item><term>ID3D11Texture2D</term><description>
        /// Individual subresources of the texture may be accessed via arrays.
        /// </description></item> 
        /// <item><term>ID3D11Texture3D</term><description>
        /// Individual subresources of the texture may be accessed via arrays.
        /// </description></item>  
        /// </list> 
        /// The Flags argument may be used to specify additional parameters at register time. The only valid value for this
        /// parameter is <see cref="CUGraphicsRegisterFlags.None"/>. <para/>
        /// Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some
        /// limitations.<param/>
        /// • The primary rendertarget may not be registered with CUDA.<param/>
        /// • Resources allocated as shared may not be registered with CUDA.<param/>
        /// • Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data
        /// cannot be shared.<param/>
        /// • Surfaces of depth or stencil formats cannot be shared.<param/>
        /// If Direct3D interoperability is not initialized for this context using <see cref="cuD3D11CtxCreate"/> then
        /// <see cref="CUResult.ErrorInvalidContext"/> is returned. If <c>pD3DResource</c> is of incorrect type or is already registered then
        /// <see cref="CUResult.ErrorInvalidHandle"/> is returned. If <c>pD3DResource</c> cannot be registered then 
        /// <see cref="CUResult.ErrorUnknown"/> is returned. If <c>Flags</c> is not one of the above specified value then <see cref="CUResult.ErrorInvalidValue"/>
        /// is returned.
        /// </summary>
        /// <param name="pCudaResource">Returned graphics resource handle</param>
        /// <param name="pD3DResource">Direct3D resource to register</param>
        /// <param name="Flags">Parameters for resource registration</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuGraphicsD3D11RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);

        /// <summary>
        /// Returns in <c>ppD3DDevice</c> the Direct3D device against which this CUDA context
        /// was created in <see cref="cuD3D11CtxCreate"/>.
        /// </summary>
        /// <param name="ppD3DDevice">Returned Direct3D device corresponding to CUDA context</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
        /// <see cref="CUResult.ErrorInvalidValue"/>.
        /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuD3D11GetDirect3DDevice(ref IntPtr ppD3DDevice);
    }
}
