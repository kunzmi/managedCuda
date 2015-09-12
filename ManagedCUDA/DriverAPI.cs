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
using System.Security.Permissions;

namespace ManagedCuda
{
    /// <summary>
    /// C# wrapper for the NVIDIA CUDA Driver API (--> cuda.h)
    /// </summary>
    public static class DriverAPINativeMethods
    {
		internal const string CUDA_DRIVER_API_DLL_NAME = "nvcuda";
		internal const string CUDA_OBSOLET_4_0 = "Don't use this CUDA API call with CUDA version >= 4.0.";
		internal const string CUDA_OBSOLET_5_0 = "Don't use this CUDA API call with CUDA version >= 5.0.";

		//Per thread default stream appendices
#if _PerThreadDefaultStream
		internal const string CUDA_PTDS = "_ptds";
		internal const string CUDA_PTSZ = "_ptsz";
#else
		internal const string CUDA_PTDS = "";
		internal const string CUDA_PTSZ = "";
#endif

		/// <summary>
        /// Gives the version of the wrapped api
        /// </summary>
        public static Version Version
        {
            get { return new Version(7, 5); }
        }

        #region Initialization
        /// <summary>
        /// Initializes the driver API and must be called before any other function from the driver API. Currently, 
        /// the Flags parameter must be <see cref="CUInitializationFlags.None"/>. If <see cref="cuInit"/> has not been called, any function from the driver API will return 
        /// <see cref="CUResult.ErrorNotInitialized"/>.
        /// </summary>
        /// <remarks>Before any call to the CUDA Driver API can be done, the API must be initialized with cuInit(0).</remarks>
        /// <param name="Flags">Currently, Flags must always be <see cref="CUInitializationFlags.None"/>.</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.<remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuInit(CUInitializationFlags Flags);
        #endregion

        #region Driver Version Query
        /// <summary>
        /// Returns in <c>driverVersion</c> the version number of the installed CUDA driver. This function automatically returns
        /// <see cref="CUResult.ErrorInvalidValue"/> if the driverVersion argument is NULL.
        /// </summary>
        /// <param name="driverVersion">Returns the CUDA driver version</param>
        /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidValue"/>.<remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
        [DllImport(CUDA_DRIVER_API_DLL_NAME)]
        public static extern CUResult cuDriverGetVersion(ref int driverVersion);
        #endregion

        #region Device management
        /// <summary>
        /// Combines all API calls for device management
        /// </summary>
        public static class DeviceManagement
        {
            /// <summary>
            /// Returns in <c>device</c> a device handle given an ordinal in the range [0, <see cref="cuDeviceGetCount"/>-1].
            /// </summary>
            /// <param name="device">Returned device handle</param>
            /// <param name="ordinal">Device number to get handle for</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuDeviceGet(ref CUdevice device, int ordinal);

            /// <summary>
            /// Returns in <c>count</c> the number of devices with compute capability greater than or equal to 1.0 that are available for
            /// execution. If there is no such device, <see cref="cuDeviceGetCount"/> returns 0.
            /// </summary>
            /// <param name="count">Returned number of compute-capable devices</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuDeviceGetCount(ref int count);

            /// <summary>
            /// Returns an ASCII string identifying the device <c>dev</c> in the NULL-terminated string pointed to by name. <c>len</c> specifies
            /// the maximum length of the string that may be returned.
            /// </summary>
            /// <param name="name">Returned identifier string for the device</param>
            /// <param name="len">Maximum length of string to store in <c>name</c></param>
            /// <param name="dev">Device to get identifier string for</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuDeviceGetName([Out] byte[] name, int len, CUdevice dev);

            /// <summary>
            /// Returns in <c>major</c> and <c>minor</c> the major and minor revision numbers that define the compute capability of the
            ///device <c>dev</c>.
            /// </summary>
            /// <param name="major">Major revision number</param>
            /// <param name="minor">Minor revision number</param>
            /// <param name="dev">Device handle</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
			[Obsolete(CUDA_OBSOLET_5_0)]
            public static extern CUResult cuDeviceComputeCapability(ref int major, ref int minor, CUdevice dev);

            /// <summary>
            /// Returns in <c>bytes</c> the total amount of memory available on the device <c>dev</c> in bytes.
            /// </summary>
            /// <param name="bytes">Returned memory available on device in bytes</param>
            /// <param name="dev">Device handle</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuDeviceTotalMem_v2(ref SizeT bytes, CUdevice dev);

            /// <summary>
            /// Returns in <c>prop</c> the (basic) properties of device <c>dev</c>. See <see cref="CUDeviceProperties"/>.
            /// </summary>
            /// <param name="prop">Returned properties of device</param>
            /// <param name="dev">Device to get properties for</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			[Obsolete(CUDA_OBSOLET_5_0)]
            public static extern CUResult cuDeviceGetProperties(ref CUDeviceProperties prop, CUdevice dev);

            /// <summary>
            /// Returns in <c>pi</c> the integer value of the attribute <c>attrib</c> on device <c>dev</c>. See <see cref="CUDeviceAttribute"/>.
            /// </summary>
            /// <param name="pi">Returned device attribute value</param>
            /// <param name="attrib">Device attribute to query</param>
            /// <param name="dev">Device handle</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuDeviceGetAttribute(ref int pi, CUDeviceAttribute attrib, CUdevice dev);

			#region Missing from 4.1
			/// <summary>
			/// Returns in <c>device</c> a device handle given a PCI bus ID string.
			/// </summary>
			/// <param name="dev">Returned device handle</param>
			/// <param name="pciBusId">String in one of the following forms: <para/>
			/// [domain]:[bus]:[device].[function]<para/>
			/// [domain]:[bus]:[device]<para/>
			/// [bus]:[device].[function]<para/>
			/// where domain, bus, device, and function are all hexadecimal values</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuDeviceGetByPCIBusId(ref CUdevice dev, byte[] pciBusId);
			
			/// <summary>
			/// Returns an ASCII string identifying the device <c>dev</c> in the NULL-terminated
			/// string pointed to by <c>pciBusId</c>. <c>len</c> specifies the maximum length of the
			/// string that may be returned.
			/// </summary>
			/// <param name="pciBusId">Returned identifier string for the device in the following format
			/// [domain]:[bus]:[device].[function]<para/>
			/// where domain, bus, device, and function are all hexadecimal values.<para/>
			/// pciBusId should be large enough to store 13 characters including the NULL-terminator.</param>
			/// <param name="len">Maximum length of string to store in <c>name</c></param>
			/// <param name="dev">Device to get identifier string for</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuDeviceGetPCIBusId(byte[] pciBusId, int len, CUdevice dev);
			
			/// <summary>
			/// Takes as input a previously allocated event. This event must have been 
			/// created with the ::CU_EVENT_INTERPROCESS and ::CU_EVENT_DISABLE_TIMING 
			/// flags set. This opaque handle may be copied into other processes and
			/// opened with ::cuIpcOpenEventHandle to allow efficient hardware
			/// synchronization between GPU work in different processes.
			/// <para/>
			/// After the event has been been opened in the importing process, 
			/// ::cuEventRecord, ::cuEventSynchronize, ::cuStreamWaitEvent and 
			/// ::cuEventQuery may be used in either process. Performing operations 
			/// on the imported event after the exported event has been freed 
			/// with ::cuEventDestroy will result in undefined behavior.
			/// <para/>
			/// IPC functionality is restricted to devices with support for unified 
			/// addressing on Linux operating systems.
			/// </summary>
			/// <param name="pHandle">Pointer to a user allocated CUipcEventHandle in which to return the opaque event handle</param>
			/// <param name="cuevent">Event allocated with ::CU_EVENT_INTERPROCESS and  ::CU_EVENT_DISABLE_TIMING flags.</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorMapFailed"/></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuIpcGetEventHandle(ref CUipcEventHandle pHandle, CUevent cuevent);
			
			/// <summary>
			/// Opens an interprocess event handle exported from another process with 
			/// ::cuIpcGetEventHandle. This function returns a ::CUevent that behaves like 
			/// a locally created event with the ::CU_EVENT_DISABLE_TIMING flag specified. 
			/// This event must be freed with ::cuEventDestroy.
			/// <para/>
			/// Performing operations on the imported event after the exported event has 
			/// been freed with ::cuEventDestroy will result in undefined behavior.
			/// <para/>
			/// IPC functionality is restricted to devices with support for unified 
			/// addressing on Linux operating systems.
			/// </summary>
			/// <param name="phEvent">Returns the imported event</param>
			/// <param name="handle">Interprocess handle to open</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorMapFailed"/></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuIpcOpenEventHandle(ref CUevent phEvent, CUipcEventHandle handle);
			
			/// <summary>
			/// Takes a pointer to the base of an existing device memory allocation created 
			/// with ::cuMemAlloc and exports it for use in another process. This is a 
			/// lightweight operation and may be called multiple times on an allocation
			/// without adverse effects. 
			/// <para/>
			/// If a region of memory is freed with ::cuMemFree and a subsequent call
			/// to ::cuMemAlloc returns memory with the same device address,
			/// ::cuIpcGetMemHandle will return a unique handle for the
			///  new memory. 
			/// <para/>
			/// IPC functionality is restricted to devices with support for unified 
			/// addressing on Linux operating systems.
			/// </summary>
			/// <param name="pHandle">Pointer to user allocated ::CUipcMemHandle to return the handle in.</param>
			/// <param name="dptr">Base pointer to previously allocated device memory </param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorMapFailed"/></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuIpcGetMemHandle(ref CUipcMemHandle pHandle, CUdeviceptr dptr);
			
			/// <summary>
			/// Maps memory exported from another process with ::cuIpcGetMemHandle into
			/// the current device address space. For contexts on different devices 
			/// ::cuIpcOpenMemHandle can attempt to enable peer access between the
			/// devices as if the user called ::cuCtxEnablePeerAccess. This behavior is 
			/// controlled by the ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag. 
			/// ::cuDeviceCanAccessPeer can determine if a mapping is possible.
			/// <para/>
			/// Contexts that may open ::CUipcMemHandles are restricted in the following way.
			/// ::CUipcMemHandles from each ::CUdevice in a given process may only be opened 
			/// by one ::CUcontext per ::CUdevice per other process.
			/// <para/>
			/// Memory returned from ::cuIpcOpenMemHandle must be freed with
			/// ::cuIpcCloseMemHandle.
			/// <para/>
			/// Calling ::cuMemFree on an exported memory region before calling
			/// ::cuIpcCloseMemHandle in the importing context will result in undefined
			/// behavior.
			/// <para/>
			/// IPC functionality is restricted to devices with support for unified 
			/// addressing on Linux operating systems.
			/// </summary>
			/// <param name="pdptr">Returned device pointer</param>
			/// <param name="handle">::CUipcMemHandle to open</param>
			/// <param name="Flags">Flags for this operation. Must be specified as ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorMapFailed"/>, <see cref="CUResult.ErrorTooManyPeers"/></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuIpcOpenMemHandle(ref CUdeviceptr pdptr, CUipcMemHandle handle, uint Flags);
			
			/// <summary>
			/// Unmaps memory returnd by ::cuIpcOpenMemHandle. The original allocation
			/// in the exporting process as well as imported mappings in other processes
			/// will be unaffected.
			/// <para/>
			/// Any resources used to enable peer access will be freed if this is the
			/// last mapping using them.
			/// <para/>
			/// IPC functionality is restricted to devices with support for unified 
			///  addressing on Linux operating systems.
			/// </summary>
			/// <param name="dptr">Device pointer returned by ::cuIpcOpenMemHandle</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidHandle"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorMapFailed"/></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuIpcCloseMemHandle(CUdeviceptr dptr);

			#endregion
		}
        #endregion

        #region Context management
        /// <summary>
        /// Combines all API calls for context management
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class ContextManagement
        {
            /// <summary>
            /// Creates a new CUDA context and associates it with the calling thread. The <c>flags</c> parameter is described in <see cref="CUCtxFlags"/>. The
			/// context is created with a usage count of 1 and the caller of <see cref="cuCtxCreate_v2"/> must call <see cref="cuCtxDestroy"/> or <see cref="cuCtxDetach"/>
            /// when done using the context. If a context is already current to the thread, it is supplanted by the newly created context
            /// and may be restored by a subsequent call to <see cref="cuCtxPopCurrent"/>.
            /// </summary>
            /// <param name="pctx">Returned context handle of the new context</param>
            /// <param name="flags">Context creation flags. See <see cref="CUCtxFlags"/></param>
            /// <param name="dev">Device to create context on</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxCreate_v2(ref CUcontext pctx, CUCtxFlags flags, CUdevice dev);

            /// <summary>
            /// Destroys the CUDA context specified by <c>ctx</c>. If the context usage count is not equal to 1, or the context is current
            /// to any CPU thread other than the current one, this function fails. Floating contexts (detached from a CPU thread via
            /// <see cref="cuCtxPopCurrent"/>) may be destroyed by this function.
            /// </summary>
            /// <param name="ctx">Context to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuCtxDestroy(CUcontext ctx);

            /// <summary>
            /// Destroys the CUDA context specified by <c>ctx</c>. The context <c>ctx</c> will be destroyed regardless of how many threads it is current to.
            /// It is the responsibility of the calling function to ensure that no API call is issued to <c>ctx</c> while cuCtxDestroy_v2() is executing.
            /// If <c>ctx</c> is current to the calling thread then <c>ctx</c> will also be 
            /// popped from the current thread's context stack (as though cuCtxPopCurrent()
            /// were called).  If <c>ctx</c> is current to other threads, then <c>ctx</c> will
            /// remain current to those threads, and attempting to access <c>ctx</c> from
            /// those threads will result in the error <see cref="CUResult.ErrorContextIsDestroyed"/>.
            /// </summary>
            /// <param name="ctx">Context to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>,  <see cref="CUResult.ErrorContextIsDestroyed"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxDestroy_v2(CUcontext ctx);

            /// <summary>
            /// Increments the usage count of the context and passes back a context handle in <c>pctx</c> that must be passed to <see cref="cuCtxDetach"/>
            /// when the application is done with the context. <see cref="cuCtxAttach"/> fails if there is no context current to the
            /// thread. Currently, the <c>flags</c> parameter must be <see cref="CUCtxAttachFlags.None"/>.
            /// </summary>
            /// <param name="pctx">Returned context handle of the current context</param>
            /// <param name="flags">Context attach flags (must be <see cref="CUCtxAttachFlags.None"/>)</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuCtxAttach(ref CUcontext pctx, CUCtxAttachFlags flags);

            /// <summary>
            /// Decrements the usage count of the context <c>ctx</c>, and destroys the context if the usage count goes to 0. The context
			/// must be a handle that was passed back by <see cref="cuCtxCreate_v2"/> or <see cref="cuCtxAttach"/>, and must be current to the calling thread.
            /// </summary>
            /// <param name="ctx">Context to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuCtxDetach([In] CUcontext ctx);

            /// <summary>
            /// Pushes the given context <c>ctx</c> onto the CPU thread’s stack of current contexts. The specified context becomes the
            /// CPU thread’s current context, so all CUDA functions that operate on the current context are affected.<para/>
            /// The previous current context may be made current again by calling <see cref="cuCtxDestroy"/> or <see cref="cuCtxPopCurrent"/>.<para/>
            /// The context must be "floating," i.e. not attached to any thread. Contexts are made to float by calling <see cref="cuCtxPopCurrent"/>.
            /// </summary>
            /// <param name="ctx">Floating context to attach</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuCtxPushCurrent([In] CUcontext ctx);

            /// <summary>
            /// Pushes the given context <c>ctx</c> onto the CPU thread’s stack of current contexts. The specified context becomes the
            /// CPU thread’s current context, so all CUDA functions that operate on the current context are affected.<para/>
            /// The previous current context may be made current again by calling <see cref="cuCtxDestroy"/> or <see cref="cuCtxPopCurrent"/>.<para/>
            /// The context must be "floating," i.e. not attached to any thread. Contexts are made to float by calling <see cref="cuCtxPopCurrent"/>.
            /// </summary>
            /// <param name="ctx">Floating context to attach</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxPushCurrent_v2([In] CUcontext ctx);

            /// <summary>
            /// Pops the current CUDA context from the CPU thread. The CUDA context must have a usage count of 1. CUDA contexts
            /// have a usage count of 1 upon creation; the usage count may be incremented with <see cref="cuCtxAttach"/> and decremented
            /// with <see cref="cuCtxDetach"/>.<para/>
            /// If successful, <see cref="cuCtxPopCurrent"/> passes back the old context handle in <c>pctx</c>. That context may then be made current
            /// to a different CPU thread by calling <see cref="cuCtxPushCurrent"/>.<para/>
            /// Floating contexts may be destroyed by calling <see cref="cuCtxDestroy"/>.<para/>
			/// If a context was current to the CPU thread before <see cref="cuCtxCreate_v2"/> or <see cref="cuCtxPushCurrent"/> was called, this function makes
            /// that context current to the CPU thread again.
            /// </summary>
            /// <param name="pctx">Returned new context handle</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuCtxPopCurrent(ref  CUcontext pctx);

            /// <summary>
            /// Pops the current CUDA context from the CPU thread. The CUDA context must have a usage count of 1. CUDA contexts
            /// have a usage count of 1 upon creation; the usage count may be incremented with <see cref="cuCtxAttach"/> and decremented
            /// with <see cref="cuCtxDetach"/>.<para/>
            /// If successful, <see cref="cuCtxPopCurrent"/> passes back the old context handle in <c>pctx</c>. That context may then be made current
            /// to a different CPU thread by calling <see cref="cuCtxPushCurrent"/>.<para/>
            /// Floating contexts may be destroyed by calling <see cref="cuCtxDestroy"/>.<para/>
			/// If a context was current to the CPU thread before <see cref="cuCtxCreate_v2"/> or <see cref="cuCtxPushCurrent"/> was called, this function makes
            /// that context current to the CPU thread again.
            /// </summary>
            /// <param name="pctx">Returned new context handle</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxPopCurrent_v2(ref  CUcontext pctx);

            /// <summary>
            /// Binds the specified CUDA context to the calling CPU thread.
            /// If <c>ctx</c> is NULL then the CUDA context previously bound to the
            /// calling CPU thread is unbound and <see cref="CUResult.Success"/> is returned.
            /// <para/>
            /// If there exists a CUDA context stack on the calling CPU thread, this
            /// will replace the top of that stack with <c>ctx</c>.  
            /// If <c>ctx</c> is NULL then this will be equivalent to popping the top
            /// of the calling CPU thread's CUDA context stack (or a no-op if the
            /// calling CPU thread's CUDA context stack is empty).
            /// </summary>
            /// <param name="ctx">Context to bind to the calling CPU thread</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxSetCurrent([In] CUcontext ctx);
            
            /// <summary>
            /// Returns in <c>ctx</c> the CUDA context bound to the calling CPU thread.
            /// If no context is bound to the calling CPU thread then <c>ctx</c> is
            /// set to NULL and <see cref="CUResult.Success"/> is returned.
            /// </summary>
            /// <param name="pctx">Returned context handle</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxGetCurrent(ref CUcontext pctx);

            /// <summary>
            /// Returns in <c>device</c> the ordinal of the current context’s device.
            /// </summary>
            /// <param name="device">Returned device ID for the current context</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxGetDevice(ref CUdevice device);

            /// <summary>
            /// Blocks until the device has completed all preceding requested tasks. <see cref="cuCtxSynchronize"/> returns an error if one of the
            /// preceding tasks failed. If the context was created with the <see cref="CUCtxFlags.BlockingSync"/> flag, the CPU thread will
            /// block until the GPU context has finished its work.
            /// </summary>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxSynchronize();

            /// <summary>
            /// Returns the API version used to create <c>ctx</c> in <c>version</c>. If <c>ctx</c>
            /// is NULL, returns the API version used to create the currently bound
            /// context.<para/>
            /// This wil return the API version used to create a context (for example,
            /// 3010 or 3020), which library developers can use to direct callers to a
            /// specific API version. Note that this API version may not be the same as
            /// returned by <see cref="cuDriverGetVersion(ref int)"/>.
            /// </summary>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxGetApiVersion(CUcontext ctx, ref uint version);

            /// <summary>
            /// On devices where the L1 cache and shared memory use the same hardware
            /// resources, this function returns through <c>pconfig</c> the preferred cache configuration
            /// for the current context. This is only a preference. The driver will use
            /// the requested configuration if possible, but it is free to choose a different
            /// configuration if required to execute functions.<para/>
            /// This will return a <c>pconfig</c> of <see cref="CUFuncCache.PreferNone"/> on devices
            /// where the size of the L1 cache and shared memory are fixed.
            /// </summary>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxGetCacheConfig(ref CUFuncCache pconfig);

            /// <summary>
            /// On devices where the L1 cache and shared memory use the same hardware
            /// resources, this sets through <c>config</c> the preferred cache configuration for
            /// the current context. This is only a preference. The driver will use
            /// the requested configuration if possible, but it is free to choose a different
            /// configuration if required to execute the function. Any function preference
            /// set via <see cref="FunctionManagement.cuFuncSetCacheConfig"/> will be preferred over this context-wide
            /// setting. Setting the context-wide cache configuration to
            /// <see cref="CUFuncCache.PreferNone"/> will cause subsequent kernel launches to prefer
            /// to not change the cache configuration unless required to launch the kernel.<para/>
            /// This setting does nothing on devices where the size of the L1 cache and
            /// shared memory are fixed.<para/>
            /// Launching a kernel with a different preference than the most recent
            /// preference setting may insert a device-side synchronization point.
            /// </summary>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxSetCacheConfig(CUFuncCache config);
			
			/// <summary>
			/// Returns the current shared memory configuration for the current context.
			/// <para/>
			/// This function will return in \p pConfig the current size of shared memory banks
			/// in the current context. On devices with configurable shared memory banks, 
			/// <see cref="cuCtxSetSharedMemConfig"/> can be used to change this setting, so that all 
			/// subsequent kernel launches will by default use the new bank size. When 
			/// <see cref="cuCtxGetSharedMemConfig"/> is called on devices without configurable shared 
			/// memory, it will return the fixed bank size of the hardware.
			///<para/>
			/// The returned bank configurations can be either:
			/// - <see cref="CUsharedconfig.FourByteBankSize"/>: set shared memory bank width to
			///   be natively four bytes.
			/// - <see cref="CUsharedconfig.EightByteBankSize"/>: set shared memory bank width to
			///   be natively eight bytes.
			/// </summary>
			/// <param name="pConfig">returned shared memory configuration</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxGetSharedMemConfig(ref CUsharedconfig pConfig);

			/// <summary>
			/// Sets the shared memory configuration for the current context.<para/>
			/// On devices with configurable shared memory banks, this function will set
		    /// the context's shared memory bank size which is used for subsequent kernel 
			/// launches. <para/> 
		    /// Changed the shared memory configuration between launches may insert a device
			/// side synchronization point between those launches.<para/>
		    /// Changing the shared memory bank size will not increase shared memory usage
		    /// or affect occupancy of kernels, but may have major effects on performance. 
		    /// Larger bank sizes will allow for greater potential bandwidth to shared memory,
		    /// but will change what kinds of accesses to shared memory will result in bank 
			/// conflicts.<para/>
			/// This function will do nothing on devices with fixed shared memory bank size.
			/// <para/>
			/// The supported bank configurations are:
			/// - <see cref="CUsharedconfig.DefaultBankSize"/>: set bank width to the default initial
			///   setting (currently, four bytes).
			/// - <see cref="CUsharedconfig.FourByteBankSize"/>: set shared memory bank width to
			///   be natively four bytes.
			/// - <see cref="CUsharedconfig.EightByteBankSize"/>: set shared memory bank width to
			///   be natively eight bytes.
			/// </summary>
			/// <param name="config">requested shared memory configuration</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuCtxSetSharedMemConfig(CUsharedconfig config);

			/// <summary>
			/// Returns numerical values that correspond to the least and greatest stream priorities.<para/>
			/// Returns in <c>leastPriority</c> and <c>greatestPriority</c> the numerical values that correspond
			/// to the least and greatest stream priorities respectively. Stream priorities
			/// follow a convention where lower numbers imply greater priorities. The range of
			/// meaningful stream priorities is given by [<c>greatestPriority</c>, <c>leastPriority</c>].
			/// If the user attempts to create a stream with a priority value that is
			/// outside the meaningful range as specified by this API, the priority is
			/// automatically clamped down or up to either <c>leastPriority</c> or <c>greatestPriority</c>
			/// respectively. See ::cuStreamCreateWithPriority for details on creating a
			/// priority stream.
			/// A NULL may be passed in for <c>leastPriority</c> or <c>greatestPriority</c> if the value
			/// is not desired.
			/// This function will return '0' in both <c>leastPriority</c> and <c>greatestPriority</c> if
			/// the current context's device does not support stream priorities
			/// (see ::cuDeviceGetAttribute).
			/// </summary>
			/// <param name="leastPriority">Pointer to an int in which the numerical value for least
			/// stream priority is returned</param>
			/// <param name="greatestPriority">Pointer to an int in which the numerical value for greatest stream priority is returned</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuCtxGetStreamPriorityRange(ref int leastPriority, ref int greatestPriority);

			/// <summary>
			/// Returns the flags for the current context<para/>
			/// Returns in \p *flags the flags of the current context. See ::cuCtxCreate for flag values.
			/// </summary>
			/// <param name="flags">Pointer to store flags of current context</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuCtxGetFlags(ref CUCtxFlags flags);



			#region Primary Context

			/// <summary>
			/// Retain the primary context on the GPU.<para/>
			/// Retains the primary context on the device, creating it if necessary,
			/// increasing its usage count. The caller must call
			/// ::cuDevicePrimaryCtxRelease() when done using the context.
			/// Unlike ::cuCtxCreate() the newly created context is not pushed onto the stack.
			/// <para/>
			/// Context creation will fail with ::CUDA_ERROR_UNKNOWN if the compute mode of
			/// the device is ::CU_COMPUTEMODE_PROHIBITED. Similarly, context creation will
			/// also fail with ::CUDA_ERROR_UNKNOWN if the compute mode for the device is
			/// set to ::CU_COMPUTEMODE_EXCLUSIVE and there is already an active, non-primary,
			/// context on the device. The function ::cuDeviceGetAttribute() can be used with
			/// ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode of the
			/// device. The <i>nvidia-smi</i> tool can be used to set the compute mode for
			/// devices. Documentation for <i>nvidia-smi</i> can be obtained by passing a
			/// -h option to it.
			/// <para/> 
			/// Please note that the primary context always supports pinned allocations. Other
			/// flags can be specified by ::cuDevicePrimaryCtxSetFlags().
			/// </summary>
			/// <param name="pctx">Returned context handle of the new context</param>
			/// <param name="dev">Device for which primary context is requested</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuDevicePrimaryCtxRetain(ref CUcontext pctx, CUdevice dev);

			/// <summary>
			/// Release the primary context on the GPU<para/>
			/// Releases the primary context interop on the device by decreasing the usage
			/// count by 1. If the usage drops to 0 the primary context of device \p dev
			/// will be destroyed regardless of how many threads it is current to.
			/// <para/>
			/// Please note that unlike ::cuCtxDestroy() this method does not pop the context
			/// from stack in any circumstances.
			/// </summary>
			/// <param name="dev">Device which primary context is released</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuDevicePrimaryCtxRelease(CUdevice dev);

			/// <summary>
			/// Set flags for the primary context<para/>
			/// Sets the flags for the primary context on the device overwriting perviously
			/// set ones. If the primary context is already created
			/// ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE is returned.
			/// <para/>
			///	The three LSBs of the \p flags parameter can be used to control how the OS
			///	thread, which owns the CUDA context at the time of an API call, interacts
			///	with the OS scheduler when waiting for results from the GPU. Only one of
			///	the scheduling flags can be set when creating a context.
			/// </summary>
			/// <param name="dev">Device for which the primary context flags are set</param>
			/// <param name="flags">New flags for the device</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuDevicePrimaryCtxSetFlags(CUdevice dev, CUCtxFlags flags);

			/// <summary>
			/// Get the state of the primary context<para/>
			/// Returns in \p *flags the flags for the primary context of \p dev, and in
			/// \p *active whether it is active.  See ::cuDevicePrimaryCtxSetFlags for flag
			/// values.
			/// </summary>
			/// <param name="dev">Device to get primary context flags for</param>
			/// <param name="flags">Pointer to store flags</param>
			/// <param name="active">Pointer to store context state; 0 = inactive, 1 = active</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuDevicePrimaryCtxGetState(CUdevice dev, ref CUCtxFlags flags, ref int active);

			/// <summary>
			/// Destroy all allocations and reset all state on the primary context
			/// 
			/// Explicitly destroys and cleans up all resources associated with the current
			/// device in the current process.
			/// 
			/// Note that it is responsibility of the calling function to ensure that no
			/// other module in the process is using the device any more. For that reason
			/// it is recommended to use ::cuDevicePrimaryCtxRelease() in most cases.
			/// However it is safe for other modules to call ::cuDevicePrimaryCtxRelease()
			/// even after resetting the device.
			/// </summary>
			/// <param name="dev">Device for which primary context is destroyed</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuDevicePrimaryCtxReset(CUdevice dev);
			#endregion
		}
        #endregion

        #region Module management
        /// <summary>
        /// Combines all API calls for module management
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class ModuleManagement
        {
            /// <summary>
            /// Takes a filename <c>fname</c> and loads the corresponding module <c>module</c> into the current context. The CUDA driver API
            /// does not attempt to lazily allocate the resources needed by a module; if the memory for functions and data (constant
            /// and global) needed by the module cannot be allocated, <see cref="cuModuleLoad"/> fails. The file should be a <c>cubin</c> file as output
            /// by <c>nvcc</c> or a <c>PTX</c> file, either as output by <c>nvcc</c> or handwrtten.
            /// </summary>
            /// <param name="module">Returned module</param>
            /// <param name="fname">Filename of module to load</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorNotFound"/>,
            /// <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorFileNotFound"/>, <see cref="CUResult.ErrorSharedObjectSymbolNotFound"/>,
            /// <see cref="CUResult.ErrorSharedObjectInitFailed"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleLoad(ref CUmodule module, string fname);

            /// <summary>
            /// Takes a byte[] as <c>image</c> and loads the corresponding module <c>module</c> into the current context. The byte array may be obtained
            /// by mapping a <c>cubin</c> or <c>PTX</c> file, passing a <c>cubin</c> or <c>PTX</c> file as a <c>null</c>-terminated text string.<para/>
            /// The byte[] is a replacement for the original pointer.
            /// </summary>
            /// <param name="module">Returned module</param>
            /// <param name="image">Module data to load</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>,
            /// <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorSharedObjectSymbolNotFound"/>,
            /// <see cref="CUResult.ErrorSharedObjectInitFailed"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleLoadData(ref CUmodule module, [In] byte[] image);

            /// <summary>
            /// Takes a byte[] as <c>image</c> and loads the corresponding module <c>module</c> into the current context. The byte array may be obtained
            /// by mapping a <c>cubin</c> or <c>PTX</c> file, passing a <c>cubin</c> or <c>PTX</c> file as a <c>null</c>-terminated text string. <para/>
            /// Options are passed as an array via <c>options</c> and any corresponding parameters are passed
            /// in <c>optionValues</c>. The number of total options is supplied via <c>numOptions</c>. Any outputs will be returned via
            /// <c>optionValues</c>. Supported options are definen in <see cref="CUJITOption"/>.<para/>
            /// The options values are currently passed in <c>IntPtr</c>-type and should then be cast into their real type. This might change in future.
            /// </summary>
            /// <param name="module">Returned module</param>
            /// <param name="image">Module data to load</param>
            /// <param name="numOptions">Number of options</param>
            /// <param name="options">Options for JIT</param>
            /// <param name="optionValues">Option values for JIT</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>
            /// <see cref="CUResult.ErrorNoBinaryForGPU"/>, <see cref="CUResult.ErrorSharedObjectSymbolNotFound"/>,
            /// <see cref="CUResult.ErrorSharedObjectInitFailed"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleLoadDataEx(ref CUmodule module, [In] byte[] image, uint numOptions, [In] CUJITOption[] options, [In, Out] IntPtr[] optionValues);

            /// <summary>
            /// Takes a byte[] as <c>fatCubin</c> and loads the corresponding module <c>module</c> into the current context. The byte[]
            /// represents a <c>fat binary</c> object, which is a collection of different <c>cubin</c> files, all representing the same device code, but
            /// compiled and optimized for different architectures. Prior to CUDA 4.0, there was no documented API for constructing and using
            /// fat binary objects by programmers. Starting with CUDA 4.0, fat binary objects can be constructed by providing the -fatbin option to nvcc.
            /// More information can be found in the <c>nvcc</c> document.
            /// </summary>
            /// <param name="module">Returned module</param>
            /// <param name="fatCubin">Fat binary to load</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorNotFound"/>, <see cref="CUResult.ErrorOutOfMemory"/>
            /// <see cref="CUResult.ErrorNoBinaryForGPU"/>, <see cref="CUResult.ErrorSharedObjectSymbolNotFound"/>,
            /// <see cref="CUResult.ErrorSharedObjectInitFailed"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleLoadFatBinary(ref CUmodule module, [In] byte[] fatCubin);

            /// <summary>
            /// Unloads a module <c>hmod</c> from the current context.
            /// </summary>
            /// <param name="hmod">Module to unload</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleUnload(CUmodule hmod);

            /// <summary>
            /// Returns in <c>hfunc</c> the handle of the function of name <c>name</c> located in module <c>hmod</c>. If no function of that name
            /// exists, <see cref="cuModuleGetFunction"/> returns <see cref="CUResult.ErrorNotFound"/>.
            /// </summary>
            /// <param name="hfunc">Returned function handle</param>
            /// <param name="hmod">Module to retrieve function from</param>
            /// <param name="name">Name of function to retrieve</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorNotFound"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleGetFunction(ref CUfunction hfunc, CUmodule hmod, string name);

            /// <summary>
            /// Returns in <c>dptr</c> and <c>bytes</c> the base pointer and size of the global of name <c>name</c> located in module <c>hmod</c>. If no
			/// variable of that name exists, <see cref="cuModuleGetGlobal_v2"/> returns <see cref="CUResult.ErrorNotFound"/>. Both parameters <c>dptr</c>
            /// and <c>bytes</c> are optional. If one of them is <c>null</c>, it is ignored.
            /// </summary>
            /// <param name="dptr">Returned global device pointer</param>
            /// <param name="bytes">Returned global size in bytes</param>
            /// <param name="hmod">Module to retrieve global from</param>
            /// <param name="name">Name of global to retrieve</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorNotFound"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleGetGlobal_v2(ref CUdeviceptr dptr, ref SizeT bytes, CUmodule hmod, string name);

            /// <summary>
            /// Returns in <c>pTexRef</c> the handle of the texture reference of name <c>name</c> in the module <c>hmod</c>. If no texture reference
            /// of that name exists, <see cref="cuModuleGetSurfRef"/> returns <see cref="CUResult.ErrorNotFound"/>. This texture reference handle
            /// should not be destroyed, since it will be destroyed when the module is unloaded.
            /// </summary>
            /// <param name="pTexRef">Returned texture reference</param>
            /// <param name="hmod">Module to retrieve texture reference from</param>
            /// <param name="name">Name of texture reference to retrieve</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorNotFound"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleGetTexRef(ref CUtexref pTexRef, CUmodule hmod, string name);

            /// <summary>
            /// Returns in <c>pSurfRef</c> the handle of the surface reference of name <c>name</c> in the module <c>hmod</c>. If no surface reference
            /// of that name exists, <see cref="cuModuleGetSurfRef"/> returns <see cref="CUResult.ErrorNotFound"/>.
            /// </summary>
            /// <param name="pSurfRef">Returned surface reference</param>
            /// <param name="hmod">Module to retrieve surface reference from</param>
            /// <param name="name">Name of surface reference to retrieve</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorNotFound"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuModuleGetSurfRef(ref CUsurfref pSurfRef, CUmodule hmod, string name);
					
			/// <summary>
			/// Creates a pending JIT linker invocation.<para/>
			/// If the call is successful, the caller owns the returned CUlinkState, which should eventually be destroyed with ::cuLinkDestroy.
			/// The device code machine size (32 or 64 bit) will match the calling application.<para/>
			/// Both linker and compiler options may be specified. Compiler options will be applied to inputs to this linker action which must 
			/// be compiled from PTX. The options ::CU_JIT_WALL_TIME, 
			/// ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, and ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES will accumulate data until the CUlinkState is destroyed.<para/>
			/// <c>optionValues</c> must remain valid for the life of the CUlinkState if output options are used. No other references to inputs are maintained after this call returns.
			/// </summary>
			/// <param name="numOptions">Size of options arrays</param>
			/// <param name="options">Array of linker and compiler options</param>
			/// <param name="optionValues">Array of option values, each cast to void *</param>
			/// <param name="stateOut">On success, this will contain a CUlinkState to specify and complete this action</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuLinkCreate_v2")]
            public static extern CUResult cuLinkCreate(uint numOptions, CUJITOption[] options, [In, Out] IntPtr[] optionValues, ref CUlinkState stateOut);



			/// <summary>
			/// Add an input to a pending linker invocation.<para/>
			/// Ownership of <c>data</c> data is retained by the caller.  No reference is retained to any inputs after this call returns.<para/>
			/// This method accepts only compiler options, which are used if the data must be compiled from PTX, and does not accept any of
			/// ::CU_JIT_WALL_TIME, ::CU_JIT_INFO_LOG_BUFFER, ::CU_JIT_ERROR_LOG_BUFFER, ::CU_JIT_TARGET_FROM_CUCONTEXT, or ::CU_JIT_TARGET.
			/// </summary>
			/// <param name="state">A pending linker action.</param>
			/// <param name="type">The type of the input data.</param>
			/// <param name="data">The input data.  PTX must be NULL-terminated.</param>
			/// <param name="size">The length of the input data.</param>
			/// <param name="name">An optional name for this input in log messages.</param>
			/// <param name="numOptions">Size of options.</param>
			/// <param name="options">Options to be applied only for this input (overrides options from ::cuLinkCreate).</param>
			/// <param name="optionValues">Array of option values, each cast to void *.</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuLinkAddData_v2")]
            public static extern CUResult cuLinkAddData(CUlinkState state, CUJITInputType type, byte[] data, SizeT size, [MarshalAs(UnmanagedType.LPStr)] string name,
				uint numOptions, CUJITOption[] options, IntPtr[] optionValues);

			/// <summary>
			/// Add a file input to a pending linker invocation.<para/>
			/// No reference is retained to any inputs after this call returns.<para/>
			/// This method accepts only compiler options, which are used if the data must be compiled from PTX, and does not accept any of
			/// ::CU_JIT_WALL_TIME, ::CU_JIT_INFO_LOG_BUFFER, ::CU_JIT_ERROR_LOG_BUFFER, ::CU_JIT_TARGET_FROM_CUCONTEXT, or ::CU_JIT_TARGET.
			/// <para/>This method is equivalent to invoking ::cuLinkAddData on the contents of the file.
			/// </summary>
			/// <param name="state">A pending linker action.</param>
			/// <param name="type">The type of the input data.</param>
			/// <param name="path">Path to the input file.</param>
			/// <param name="numOptions">Size of options.</param>
			/// <param name="options">Options to be applied only for this input (overrides options from ::cuLinkCreate).</param>
			/// <param name="optionValues">Array of option values, each cast to void *.</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuLinkAddFile_v2")]
            public static extern CUResult cuLinkAddFile(CUlinkState state, CUJITInputType type, string path, uint numOptions, CUJITOption[] options, IntPtr[] optionValues);


			/// <summary>
			/// Complete a pending linker invocation.<para/>
			/// Completes the pending linker action and returns the cubin image for the linked
			/// device code, which can be used with ::cuModuleLoadData. <para/>The cubin is owned by
			/// <c>state</c>, so it should be loaded before <c>state</c> is destroyed via ::cuLinkDestroy.
			/// This call does not destroy <c>state</c>.
			/// </summary>
			/// <param name="state">A pending linker invocation</param>
			/// <param name="cubinOut">On success, this will point to the output image</param>
			/// <param name="sizeOut">Optional parameter to receive the size of the generated image</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuLinkComplete(CUlinkState state, ref IntPtr cubinOut, ref SizeT sizeOut);

			/// <summary>
			/// Destroys state for a JIT linker invocation.
			/// </summary>
			/// <param name="state">State object for the linker invocation</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuLinkDestroy(CUlinkState state);

        }
        #endregion

        #region Memory management
        /// <summary>
        /// Combines all API calls for memory management
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class MemoryManagement
        {
            /// <summary>
            /// Returns in <c>free</c> and <c>total</c> respectively, the free and total amount of memory available for allocation by the 
            /// CUDA context, in bytes.
            /// </summary>
            /// <param name="free">Returned free memory in bytes</param>
            /// <param name="total">Returned total memory in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemGetInfo_v2(ref SizeT free, ref SizeT total);

            /// <summary>
            /// Allocates <c>bytesize</c> bytes of linear memory on the device and returns in <c>dptr</c> a pointer to the allocated memory.
            /// The allocated memory is suitably aligned for any kind of variable. The memory is not cleared. If <c>bytesize</c> is 0,
			/// <see cref="cuMemAlloc_v2"/> returns <see cref="CUResult.ErrorInvalidValue"/>.
            /// </summary>
            /// <param name="dptr">Returned device pointer</param>
            /// <param name="bytesize">Requested allocation size in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemAlloc_v2(ref CUdeviceptr dptr, SizeT bytesize);

            /// <summary>
            /// Allocates at least <c>WidthInBytes * Height</c> bytes of linear memory on the device and returns in <c>dptr</c> a pointer
            /// to the allocated memory. The function may pad the allocation to ensure that corresponding pointers in any given
            /// row will continue to meet the alignment requirements for coalescing as the address is updated from row to row. <para/>
            /// <c>ElementSizeBytes</c> specifies the size of the largest reads and writes that will be performed on the memory range.<para/>
            /// <c>ElementSizeBytes</c> may be 4, 8 or 16 (since coalesced memory transactions are not possible on other data sizes). If
            /// <c>ElementSizeBytes</c> is smaller than the actual read/write size of a kernel, the kernel will run correctly, but possibly
			/// at reduced speed. The pitch returned in <c>pPitch</c> by <see cref="cuMemAllocPitch_v2"/> is the width in bytes of the allocation. The
            /// intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.<para/>
            /// Given the row and column of an array element of type T, the address is computed as:<para/>
            /// <code>T * pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;</code><para/>
			/// The pitch returned by <see cref="cuMemAllocPitch_v2"/> is guaranteed to work with <see cref="SynchronousMemcpy_v2.cuMemcpy2D_v2"/> under all circumstances. For
			/// allocations of 2D arrays, it is recommended that programmers consider performing pitch allocations using <see cref="cuMemAllocPitch_v2"/>.
            /// Due to alignment restrictions in the hardware, this is especially true if the application will be performing
            /// 2D memory copies between different regions of device memory (whether linear memory or CUDA arrays). <para/>
			/// The byte alignment of the pitch returned by <see cref="cuMemAllocPitch_v2"/> is guaranteed to match or exceed the alignment
			/// requirement for texture binding with <see cref="TextureReferenceManagement.cuTexRefSetAddress2D_v2"/>.
            /// </summary>
            /// <param name="dptr">Returned device pointer</param>
            /// <param name="pPitch">Returned pitch of allocation in bytes</param>
            /// <param name="WidthInBytes">Requested allocation width in bytes</param>
            /// <param name="Height">Requested allocation height in rows</param>
            /// <param name="ElementSizeBytes">Size of largest reads/writes for range</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemAllocPitch_v2(ref CUdeviceptr dptr, ref SizeT pPitch, SizeT WidthInBytes, SizeT Height, uint ElementSizeBytes);

            /// <summary>
			/// Frees the memory space pointed to by <c>dptr</c>, which must have been returned by a previous call to <see cref="cuMemAlloc_v2"/> or
			/// <see cref="cuMemAllocPitch_v2"/>.
            /// </summary>
            /// <param name="dptr">Pointer to memory to free</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemFree_v2(CUdeviceptr dptr);

            /// <summary>
			/// Returns the base address in <c>pbase</c> and size in <c>psize</c> of the allocation by <see cref="cuMemAlloc_v2"/> or <see cref="cuMemAllocPitch_v2"/>
            /// that contains the input pointer <c>dptr</c>. Both parameters <c>pbase</c> and <c>psize</c> are optional. If one of them is <c>null</c>, it is
            /// ignored.
            /// </summary>
            /// <param name="pbase">Returned base address</param>
            /// <param name="psize">Returned size of device memory allocation</param>
            /// <param name="dptr">Device pointer to query</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemGetAddressRange_v2(ref CUdeviceptr pbase, ref SizeT psize, CUdeviceptr dptr);

            /// <summary>
            /// Allocates <c>bytesize</c> bytes of host memory that is page-locked and accessible to the device. The driver tracks the virtual
			/// memory ranges allocated with this function and automatically accelerates calls to functions such as <see cref="SynchronousMemcpy_v2.cuMemcpyHtoD_v2(CUdeviceptr, IntPtr, SizeT)"/>.
            /// Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than
			/// pageable memory obtained with functions such as <c>malloc()</c>. Allocating excessive amounts of memory with <see cref="cuMemAllocHost_v2"/>
            /// may degrade system performance, since it reduces the amount of memory available to the system for paging.
            /// As a result, this function is best used sparingly to allocate staging areas for data exchange between host and device.
            /// </summary>
            /// <param name="pp">Returned host pointer to page-locked memory</param>
            /// <param name="bytesize">Requested allocation size in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemAllocHost_v2(ref IntPtr pp, SizeT bytesize);

            /// <summary>
			/// Frees the memory space pointed to by <c>p</c>, which must have been returned by a previous call to <see cref="cuMemAllocHost_v2"/>.
            /// </summary>
            /// <param name="p">Pointer to memory to free</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemFreeHost(IntPtr p);

            /// <summary>
            /// Allocates <c>bytesize</c> bytes of host memory that is page-locked and accessible to the device. The driver tracks the virtual
			/// memory ranges allocated with this function and automatically accelerates calls to functions such as <see cref="SynchronousMemcpy_v2.cuMemcpyHtoD_v2(CUdeviceptr, IntPtr, SizeT)"/>.
            /// Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than
            /// pageable memory obtained with functions such as <c>malloc()</c>. Allocating excessive amounts of pinned
            /// memory may degrade system performance, since it reduces the amount of memory available to the system for paging.
            /// As a result, this function is best used sparingly to allocate staging areas for data exchange between host and device.<para/>
            /// For the <c>Flags</c> parameter see <see cref="CUMemHostAllocFlags"/>.<para/>
            /// The CUDA context must have been created with the <see cref="CUCtxFlags.MapHost"/> flag in order for the <see cref="CUMemHostAllocFlags.DeviceMap"/>
            /// flag to have any effect.<para/>
            /// The <see cref="CUCtxFlags.MapHost"/> flag may be specified on CUDA contexts for devices that do not support
			/// mapped pinned memory. The failure is deferred to <see cref="cuMemHostGetDevicePointer_v2"/> because the memory may be
            /// mapped into other CUDA contexts via the <see cref="CUMemHostAllocFlags.Portable"/> flag. <para/>
            /// The memory allocated by this function must be freed with <see cref="cuMemFreeHost"/>.<para/>
            /// Note all host memory allocated using <see cref="cuMemHostAlloc"/> will automatically
            /// be immediately accessible to all contexts on all devices which support unified
            /// addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
            /// Unless the flag ::CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device pointer 
            /// that may be used to access this host memory from those contexts is always equal 
            /// to the returned host pointer <c>pp</c>.  If the flag ::CU_MEMHOSTALLOC_WRITECOMBINED
			/// is specified, then the function <see cref="cuMemHostGetDevicePointer_v2"/> must be used
            /// to query the device pointer, even if the context supports unified addressing.
            /// See \ref CUDA_UNIFIED for additional details.
            /// </summary>
            /// <param name="pp">Returned host pointer to page-locked memory</param>
            /// <param name="bytesize">Requested allocation size in bytes</param>
            /// <param name="Flags">Flags for allocation request</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemHostAlloc(ref IntPtr pp, SizeT bytesize, CUMemHostAllocFlags Flags);

            /// <summary>
            /// Passes back the device pointer <c>pdptr</c> corresponding to the mapped, pinned host buffer <c>p</c> allocated by <see cref="cuMemHostAlloc"/>.
			/// <see cref="cuMemHostGetDevicePointer_v2"/> will fail if the <see cref="CUMemHostAllocFlags.DeviceMap"/> flag was not specified at the
            /// time the memory was allocated, or if the function is called on a GPU that does not support mapped pinned memory.
            /// Flags provides for future releases. For now, it must be set to 0.
            /// </summary>
            /// <param name="pdptr">Returned device pointer</param>
            /// <param name="p">Host pointer</param>
            /// <param name="Flags">Options (must be 0)</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemHostGetDevicePointer_v2(ref CUdeviceptr pdptr, IntPtr p, int Flags);

            /// <summary>
            /// Passes back the flags <c>pFlags</c> that were specified when allocating the pinned host buffer <c>p</c> allocated by
            /// <see cref="cuMemHostAlloc"/>.<para/>
			/// <see cref="cuMemHostGetFlags"/> will fail if the pointer does not reside in an allocation performed by <see cref="cuMemAllocHost_v2"/> or
            /// <see cref="cuMemHostAlloc"/>.
            /// </summary>
            /// <param name="pFlags">Returned flags</param>
            /// <param name="p">Host pointer</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemHostGetFlags(ref CUMemHostAllocFlags pFlags, IntPtr p);

            /// <summary>
            /// Page-locks the memory range specified by <c>p</c> and <c>bytesize</c> and maps it
            /// for the device(s) as specified by <c>Flags</c>. This memory range also is added
            /// to the same tracking mechanism as ::cuMemHostAlloc to automatically accelerate
            /// calls to functions such as <see cref="SynchronousMemcpy_v2.cuMemcpyHtoD_v2(BasicTypes.CUdeviceptr, VectorTypes.dim3[], BasicTypes.SizeT)"/>. Since the memory can be accessed 
            /// directly by the device, it can be read or written with much higher bandwidth 
            /// than pageable memory that has not been registered.  Page-locking excessive
            /// amounts of memory may degrade system performance, since it reduces the amount
            /// of memory available to the system for paging. As a result, this function is
            /// best used sparingly to register staging areas for data exchange between
            /// host and device.<para/>
            /// The pointer <c>p</c> and size <c>bytesize</c> must be aligned to the host page size (4 KB).<para/>
            /// The memory page-locked by this function must be unregistered with <see cref="cuMemHostUnregister"/>
            /// </summary>
            /// <param name="p">Host pointer to memory to page-lock</param>
            /// <param name="byteSize">Size in bytes of the address range to page-lock</param>
            /// <param name="Flags">Flags for allocation request</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemHostRegister_v2")]
            public static extern CUResult cuMemHostRegister(IntPtr p, SizeT byteSize, CUMemHostRegisterFlags Flags);

            /// <summary>
            /// Unmaps the memory range whose base address is specified by <c>p</c>, and makes it pageable again.<para/>
            /// The base address must be the same one specified to <see cref="cuMemHostRegister"/>.
            /// </summary>
            /// <param name="p">Host pointer to memory to page-lock</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuMemHostUnregister(IntPtr p);

            /// <summary>
            /// Returns information about a pointer
            /// </summary>
            /// <param name="data">Returned pointer attribute value</param>
            /// <param name="attribute">Pointer attribute to query</param>
            /// <param name="ptr">Pointer</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuPointerGetAttribute(ref CUcontext data, CUPointerAttribute attribute, CUdeviceptr ptr);

            /// <summary>
            /// Returns information about a pointer
            /// </summary>
            /// <param name="data">Returned pointer attribute value</param>
            /// <param name="attribute">Pointer attribute to query</param>
            /// <param name="ptr">Pointer</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuPointerGetAttribute(ref CUMemoryType data, CUPointerAttribute attribute, CUdeviceptr ptr);

			/// <summary>
			/// Returns information about a pointer
			/// </summary>
			/// <param name="data">Returned pointer attribute value</param>
			/// <param name="attribute">Pointer attribute to query</param>
			/// <param name="ptr">Pointer</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
			/// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuPointerGetAttribute(ref CUdeviceptr data, CUPointerAttribute attribute, CUdeviceptr ptr);

			/// <summary>
			/// Returns information about a pointer
			/// </summary>
			/// <param name="data">Returned pointer attribute value</param>
			/// <param name="attribute">Pointer attribute to query</param>
			/// <param name="ptr">Pointer</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
			/// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuPointerGetAttribute(ref IntPtr data, CUPointerAttribute attribute, CUdeviceptr ptr);

			/// <summary>
			/// Returns information about a pointer
			/// </summary>
			/// <param name="data">Returned pointer attribute value</param>
			/// <param name="attribute">Pointer attribute to query</param>
			/// <param name="ptr">Pointer</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
			/// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuPointerGetAttribute(ref CudaPointerAttributeP2PTokens data, CUPointerAttribute attribute, CUdeviceptr ptr);

			/// <summary>
			/// Returns information about a pointer
			/// </summary>
			/// <param name="data">Returned pointer attribute value</param>
			/// <param name="attribute">Pointer attribute to query</param>
			/// <param name="ptr">Pointer</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
			/// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuPointerGetAttribute(ref int data, CUPointerAttribute attribute, CUdeviceptr ptr);

			/// <summary>
			/// Returns information about a pointer
			/// </summary>
			/// <param name="data">Returned pointer attribute value</param>
			/// <param name="attribute">Pointer attribute to query</param>
			/// <param name="ptr">Pointer</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/>.
			/// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuPointerGetAttribute(ref ulong data, CUPointerAttribute attribute, CUdeviceptr ptr);


			
            /// <summary>
            /// Allocates memory that will be automatically managed by the Unified Memory system
            /// <para/>
            /// Allocates <c>bytesize</c> bytes of managed memory on the device and returns in
            /// <c>dptr</c> a pointer to the allocated memory. If the device doesn't support
			/// allocating managed memory, <see cref="CUResult.ErrorNotSupported"/> is returned. Support
            /// for managed memory can be queried using the device attribute
			///  <see cref="CUDeviceAttribute.ManagedMemory"/>. The allocated memory is suitably
			/// aligned for any kind of variable. The memory is not cleared. If <c>bytesize</c>
            /// is 0, ::cuMemAllocManaged returns ::CUDA_ERROR_INVALID_VALUE. The pointer
            /// is valid on the CPU and on all GPUs in the system that support managed memory.
            /// All accesses to this pointer must obey the Unified Memory programming model.
			/// <para/>
            /// <c>flags</c> specifies the default stream association for this allocation.
			/// <c>flags</c> must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST. If
            /// ::CU_MEM_ATTACH_GLOBAL is specified, then this memory is accessible from
            /// any stream on any device. If ::CU_MEM_ATTACH_HOST is specified, then the
            /// allocation is created with initial visibility restricted to host access only;
            /// an explicit call to ::cuStreamAttachMemAsync will be required to enable access
            /// on the device.
			/// <para/>
            /// If the association is later changed via ::cuStreamAttachMemAsync to
            /// a single stream, the default association as specifed during ::cuMemAllocManaged
            /// is restored when that stream is destroyed. For __managed__ variables, the
            /// default association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a
            /// stream is an asynchronous operation, and as a result, the change to default
            /// association won't happen until all work in the stream has completed.
			/// <para/>
            /// Memory allocated with ::cuMemAllocManaged should be released with ::cuMemFree.
			/// <para/>
            /// On a multi-GPU system with peer-to-peer support, where multiple GPUs support
            /// managed memory, the physical storage is created on the GPU which is active
            /// at the time ::cuMemAllocManaged is called. All other GPUs will reference the
            /// data at reduced bandwidth via peer mappings over the PCIe bus. The Unified
            /// Memory management system does not migrate memory between GPUs.
			/// <para/>
            /// On a multi-GPU system where multiple GPUs support managed memory, but not
            /// all pairs of such GPUs have peer-to-peer support between them, the physical
            /// storage is created in 'zero-copy' or system memory. All GPUs will reference
            /// the data at reduced bandwidth over the PCIe bus. In these circumstances,
            /// use of the environment variable, CUDA_VISIBLE_DEVICES, is recommended to
            /// restrict CUDA to only use those GPUs that have peer-to-peer support. This
            /// environment variable is described in the CUDA programming guide under the
            /// "CUDA environment variables" section.
            /// </summary>
            /// <param name="dptr">Returned device pointer</param>
            /// <param name="bytesize">Requested allocation size in bytes</param>
			/// <param name="flags">Must be one of <see cref="CUmemAttach_flags.Global"/> or <see cref="CUmemAttach_flags.Host"/></param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorNotSupported"/>, , <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuMemAllocManaged(ref CUdeviceptr dptr, SizeT bytesize, CUmemAttach_flags flags);
			

			/// <summary>
			/// Set attributes on a previously allocated memory region<para/>
			/// The supported attributes are:<para/>
			/// <see cref="CUPointerAttribute.SyncMemops"/>: A boolean attribute that can either be set (1) or unset (0). When set,
			/// memory operations that are synchronous. If there are some previously initiated
			/// synchronous memory operations that are pending when this attribute is set, the
 			/// function does not return until those memory operations are complete.
			/// See further documentation in the section titled "API synchronization behavior"
			/// to learn more about cases when synchronous memory operations can
			/// exhibit asynchronous behavior.
			/// <c>value</c> will be considered as a pointer to an unsigned integer to which this attribute is to be set.
			/// </summary>
			/// <param name="value">Pointer to memory containing the value to be set</param>
			/// <param name="attribute">Pointer attribute to set</param>
			/// <param name="ptr">Pointer to a memory region allocated using CUDA memory allocation APIs</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidDevice"/></returns>.
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuPointerSetAttribute(ref int value, CUPointerAttribute attribute, CUdeviceptr ptr);

			/// <summary>
			/// Returns information about a pointer.<para/>
			/// The supported attributes are (refer to ::cuPointerGetAttribute for attribute descriptions and restrictions):
			/// <para/>
			/// - ::CU_POINTER_ATTRIBUTE_CONTEXT<para/>
			/// - ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE<para/>
			/// - ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER<para/>
			/// - ::CU_POINTER_ATTRIBUTE_HOST_POINTER<para/>
			/// - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS<para/>
			/// - ::CU_POINTER_ATTRIBUTE_BUFFER_ID<para/>
			/// - ::CU_POINTER_ATTRIBUTE_IS_MANAGED<para/>
			/// </summary>
			/// <param name="numAttributes">Number of attributes to query</param>
			/// <param name="attributes">An array of attributes to query (numAttributes and the number of attributes in this array should match)</param>
			/// <param name="data">A two-dimensional array containing pointers to memory
			/// locations where the result of each attribute query will be written to.</param>
			/// <param name="ptr">Pointer to query</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuPointerGetAttributes(uint numAttributes, CUPointerAttribute[] attributes,  IntPtr data, CUdeviceptr ptr);


        }
        #endregion

        #region Synchronous Memcpy_v2
        /// <summary>
        /// Intra-device memcpy's done with these functions may execute in parallel with the CPU,
        /// but if host memory is involved, they wait until the copy is done before returning.
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class SynchronousMemcpy_v2
        {
            //New memcpy functions in CUDA 4.0 for unified addressing
            /// <summary>
            /// Copies data between two pointers. <para/>
            /// <c>dst</c> and <c>src</c> are base pointers of the destination and source, respectively.  
            /// <c>ByteCount</c> specifies the number of bytes to copy.
            /// Note that this function infers the type of the transfer (host to host, host to 
            /// device, device to device, or device to host) from the pointer values.  This
            /// function is only allowed in contexts which support unified addressing.
            /// Note that this function is synchronous.
            /// </summary>
            /// <param name="dst">Destination unified virtual address space pointer</param>
            /// <param name="src">Source unified virtual address space pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy" + CUDA_PTDS)]
            public static extern CUResult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, SizeT ByteCount);
            
            /// <summary>
            /// Copies from device memory in one context to device memory in another
            /// context. <c>dstDevice</c> is the base device pointer of the destination memory 
            /// and <c>dstContext</c> is the destination context.  <c>srcDevice</c> is the base 
            /// device pointer of the source memory and <c>srcContext</c> is the source pointer.  
            /// <c>ByteCount</c> specifies the number of bytes to copy.
            /// <para/>
            /// Note that this function is asynchronous with respect to the host, but 
            /// serialized with respect all pending and future asynchronous work in to the 
            /// current context, <c>srcContext</c>, and <c>dstContext</c> (use <see cref="AsynchronousMemcpy_v2.cuMemcpyPeerAsync"/> 
            /// to avoid this synchronization).
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstContext">Destination context</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="srcContext">Source context</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyPeer" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, SizeT ByteCount);
            
            /// <summary>
            /// Perform a 3D memory copy according to the parameters specified in
            /// <c>pCopy</c>.  See the definition of the <see cref="CUDAMemCpy3DPeer"/> structure
            /// for documentation of its parameters.<para/>
            /// Note that this function is synchronous with respect to the host only if
            /// the source or destination memory is of type ::CU_MEMORYTYPE_HOST.
            /// Note also that this copy is serialized with respect all pending and future 
            /// asynchronous work in to the current context, the copy's source context,
            /// and the copy's destination context (use <see cref="AsynchronousMemcpy_v2.cuMemcpy3DPeerAsync"/> to avoid 
            /// this synchronization).
            /// </summary>
            /// <param name="pCopy">Parameters for the memory copy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy3DPeer" + CUDA_PTDS)]
            public static extern CUResult cuMemcpy3DPeer(ref CUDAMemCpy3DPeer pCopy);



            // 1D functions
            // system <-> device memory
            #region VectorTypesArray
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] dim3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] char1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] char2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] char3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] char4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uchar1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uchar2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uchar3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uchar4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] short1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] short2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] short3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] short4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ushort1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ushort2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ushort3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ushort4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] int1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] int2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] int3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] int4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uint1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uint2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uint3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uint4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] long1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] long2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] long3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] long4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ulong1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ulong2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ulong3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ulong4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] float1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] float2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] float3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] float4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] double1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] double2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] cuDoubleComplex[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] cuDoubleReal[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] cuFloatComplex[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] cuFloatReal[] srcHost, SizeT ByteCount);
            #endregion
            #region NumberTypesArray
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] byte[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] sbyte[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ushort[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] short[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] uint[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] int[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ulong[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] long[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] float[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] double[] srcHost, SizeT ByteCount);
            #endregion
            #region VectorTypes
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref dim3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref char1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref char2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref char3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref char4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uchar1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uchar2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uchar3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uchar4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref short1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref short2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref short3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref short4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ushort1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ushort2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ushort3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ushort4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref int1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref int2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref int3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref int4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uint1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uint2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uint3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uint4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref long1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref long2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref long3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref long4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ulong1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ulong2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ulong3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ulong4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref float1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref float2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref float3 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref float4 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref double1 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref double2 srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref cuDoubleComplex srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref cuDoubleReal srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref cuFloatComplex srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref cuFloatReal srcHost, SizeT ByteCount);
            #endregion
            #region NumberTypes
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref byte srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref sbyte srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ushort srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref short srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref uint srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref int srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref ulong srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref long srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref float srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] ref double srcHost, SizeT ByteCount);
            #endregion
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, [In] IntPtr srcHost, SizeT ByteCount);


            //Device to Host
            #region VectorTypesArray
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] dim3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] char1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] char2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] char3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] char4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uchar1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uchar2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uchar3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uchar4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] short1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] short2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] short3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] short4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ushort1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ushort2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ushort3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ushort4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] int1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] int2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] int3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] int4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uint1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uint2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uint3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uint4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] long1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] long2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] long3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] long4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ulong1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ulong2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ulong3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ulong4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] float1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] float2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] float3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] float4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] double1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] double2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] cuDoubleComplex[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] cuDoubleReal[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns> 
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] cuFloatComplex[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] cuFloatReal[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            #endregion
            #region NumberTypesArray
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] byte[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] sbyte[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ushort[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] short[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] uint[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] int[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] ulong[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] long[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] float[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] double[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            #endregion
            #region VectorTypes
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref dim3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref char1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref char2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref char3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref char4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uchar1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uchar2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uchar3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uchar4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref short1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref short2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref short3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref short4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ushort1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ushort2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ushort3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ushort4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref int1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref int2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref int3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref int4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uint1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uint2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uint3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uint4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref long1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref long2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref long3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref long4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ulong1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ulong2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ulong3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ulong4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref float1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref float2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref float3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref float4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref double1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref double2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref cuDoubleComplex dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref cuDoubleReal dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref cuFloatComplex dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref cuFloatReal dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            #endregion
            #region NumberTypes
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref byte dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref sbyte dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ushort dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref short dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref uint dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref int dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref ulong dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref long dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref float dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2(ref double dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
            #endregion
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is synchronous.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoH_v2([Out] IntPtr dstHost, CUdeviceptr srcDevice, SizeT ByteCount);

            // device <-> device memory
            /// <summary>
            /// Copies from device memory to device memory. <c>dstDevice</c> and <c>srcDevice</c> are the base pointers of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is asynchronous.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, SizeT ByteCount);

            // device <-> array memory
            /// <summary>
            /// Copies from device memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting index of the destination data. <c>srcDevice</c> specifies the base pointer of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyDtoA_v2(CUarray dstArray, SizeT dstOffset, CUdeviceptr srcDevice, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to device memory. <c>dstDevice</c> specifies the base pointer of the destination and
            /// must be naturally aligned with the CUDA array elements. <c>srcArray</c> and <c>srcOffset</c> specify the CUDA array
            /// handle and the offset in bytes into the array where the copy is to begin. <c>ByteCount</c> specifies the number of bytes to
            /// copy and must be evenly divisible by the array element size.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes. Must be evenly divisible by the array element size.</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoD_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);

            // system <-> array memory
            #region VectorTypesArray
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] dim3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] char1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] char2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] char3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] char4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uchar1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uchar2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uchar3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uchar4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] short1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] short2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] short3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] short4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ushort1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ushort2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ushort3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ushort4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] int1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] int2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] int3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] int4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uint1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uint2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uint3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uint4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] long1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] long2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] long3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] long4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ulong1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ulong2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ulong3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ulong4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] float1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] float2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] float3[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] float4[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] double1[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] double2[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] cuDoubleComplex[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] cuDoubleReal[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] cuFloatComplex[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] cuFloatReal[] srcHost, SizeT ByteCount);
            #endregion
            #region NumberTypesArray
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] byte[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] sbyte[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ushort[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] short[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] uint[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] int[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] ulong[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] long[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] float[] srcHost, SizeT ByteCount);
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] double[] srcHost, SizeT ByteCount);
            #endregion
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>pSrc</c> specifies the base address of the source. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyHtoA_v2(CUarray dstArray, SizeT dstOffset, [In] IntPtr srcHost, SizeT ByteCount);

            #region VectorTypesArray
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] dim3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] char1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] char2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] char3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] char4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uchar1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uchar2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uchar3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uchar4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] short1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] short2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] short3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] short4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ushort1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ushort2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ushort3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ushort4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] int1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] int2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] int3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] int4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uint1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uint2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uint3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uint4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] long1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] long2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] long3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] long4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ulong1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ulong2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ulong3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ulong4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] float1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] float2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] float3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] float4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] double1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] double2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] cuDoubleComplex[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] cuDoubleReal[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] cuFloatComplex[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] cuFloatReal[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            #endregion
            #region NumberTypesArray
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] byte[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] sbyte[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ushort[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] short[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] uint[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] int[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] ulong[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] long[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] float[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] double[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
            #endregion

            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.
            /// </summary>
            /// <param name="dstHost">Destination device pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoH_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoH_v2([Out] IntPtr dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);

            // array <-> array memory
            /// <summary>
            /// Copies from one 1D CUDA array to another. <c>dstArray</c> and <c>srcArray</c> specify the handles of the destination and
            /// source CUDA arrays for the copy, respectively. <c>dstOffset</c> and <c>srcOffset</c> specify the destination and source
            /// offsets in bytes into the CUDA arrays. <c>ByteCount</c> is the number of bytes to be copied. The size of the elements
            /// in the CUDA arrays need not be the same format, but the elements must be the same size; and count must be evenly
            /// divisible by that size.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoA_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpyAtoA_v2(CUarray dstArray, SizeT dstOffset, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);

            // 2D memcpy
            /// <summary>
            /// Perform a 2D memory copy according to the parameters specified in <c>pCopy</c>. See <see cref="CUDAMemCpy2D"/>.
            /// <see cref="cuMemcpy2D_v2"/> returns an error if any pitch is greater than the maximum allowed (<see cref="CUDeviceProperties.memPitch"/>).
			/// <see cref="MemoryManagement.cuMemAllocPitch_v2"/> passes back pitches that always work with <see cref="cuMemcpy2D_v2"/>. On intra-device
            /// memory copies (device <![CDATA[<->]]> device, CUDA array <![CDATA[<->]]> device, CUDA array <![CDATA[<->]]> CUDA array), <see cref="cuMemcpy2D_v2"/> may fail
			/// for pitches not computed by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>. <see cref="cuMemcpy2DUnaligned_v2"/> does not have this restriction, but
            /// may run significantly slower in the cases where <see cref="cuMemcpy2D_v2"/> would have returned an error code.
            /// </summary>
            /// <param name="pCopy">Parameters for the memory copy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy2D_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpy2D_v2(ref CUDAMemCpy2D pCopy);
            /// <summary>
            /// Perform a 2D memory copy according to the parameters specified in <c>pCopy</c>. See <see cref="CUDAMemCpy2D"/>.
            /// </summary>
            /// <param name="pCopy">Parameters for the memory copy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy2DUnaligned_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpy2DUnaligned_v2(ref CUDAMemCpy2D pCopy);

            // 3D memcpy
            /// <summary>
            /// Perform a 3D memory copy according to the parameters specified in <c>pCopy</c>. See <see cref="CUDAMemCpy3D"/>.<para/>
            /// The srcLOD and dstLOD members of the CUDAMemCpy3D structure must be set to 0.
            /// </summary>
            /// <param name="pCopy">Parameters for the memory copy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>            
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy3D_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemcpy3D_v2(ref CUDAMemCpy3D pCopy);
        }
        #endregion

        #region Asynchronous Memcpy_v2
        /// <summary>
        /// Any host memory involved must be DMA'able (e.g., allocated with cuMemAllocHost).
        /// memcpy's done with these functions execute in parallel with the CPU and, if
        /// the hardware is available, may execute in parallel with the GPU.
        /// Asynchronous memcpy must be accompanied by appropriate stream synchronization.
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class AsynchronousMemcpy_v2
        {
            //New memcpy functions in CUDA 4.0 for unified addressing
            /// <summary>
            /// Copies data between two pointers. 
            /// <c>dst</c> and <c>src</c> are base pointers of the destination and source, respectively.  
            /// <c>ByteCount</c> specifies the number of bytes to copy.
            /// Note that this function infers the type of the transfer (host to host, host to 
            /// device, device to device, or device to host) from the pointer values.  This
            /// function is only allowed in contexts which support unified addressing.
            /// Note that this function is asynchronous and can optionally be associated to 
            /// a stream by passing a non-zero <c>hStream</c> argument
            /// </summary>
            /// <param name="dst">Destination unified virtual address space pointer</param>
            /// <param name="src">Source unified virtual address space pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>   
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAsync" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, SizeT ByteCount, CUstream hStream);

            /// <summary>
            /// Copies from device memory in one context to device memory in another
            /// context. <c>dstDevice</c> is the base device pointer of the destination memory 
            /// and <c>dstContext</c> is the destination context. <c>srcDevice</c> is the base 
            /// device pointer of the source memory and <c>srcContext</c> is the source pointer.  
            /// <c>ByteCount</c> specifies the number of bytes to copy.  Note that this function
            /// is asynchronous with respect to the host and all work in other streams in
            /// other devices.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstContext">Destination context</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="srcContext">Source context</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>   
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyPeerAsync" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, SizeT ByteCount, CUstream hStream);

            /// <summary>
            /// Perform a 3D memory copy according to the parameters specified in
            /// <c>pCopy</c>.  See the definition of the <see cref="BasicTypes.CUDAMemCpy3DPeer"/> structure
            /// for documentation of its parameters.
            /// </summary>
            /// <param name="pCopy">Parameters for the memory copy</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>   
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy3DPeerAsync" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpy3DPeerAsync(ref CUDAMemCpy3DPeer pCopy, CUstream hStream);



            // 1D functions
            // system <-> device memory
            /// <summary>
            /// Copies from host memory to device memory. <c>dstDevice</c> and <c>srcHost</c> are the base addresses of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. <para/>
            /// <see cref="cuMemcpyHtoDAsync_v2(CUdeviceptr, IntPtr, SizeT, CUstream)"/> is asynchronous and can optionally be associated to a stream by passing a non-zero <c>hStream</c>
            /// argument. It only works on page-locked memory and returns an error if a pointer to pageable memory is passed as
            /// input.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>   
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoDAsync_v2" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, [In] IntPtr srcHost, SizeT ByteCount, CUstream hStream);

            //Device -> Host
            /// <summary>
            /// Copies from device to host memory. <c>dstHost</c> and <c>srcDevice</c> specify the base pointers of the destination and
            /// source, respectively. <c>ByteCount</c> specifies the number of bytes to copy.<para/>
            /// <see cref="cuMemcpyDtoHAsync_v2(IntPtr, CUdeviceptr, SizeT, CUstream)"/> is asynchronous and can optionally be associated to a stream by passing a non-zero
            /// <c>hStream</c> argument. It only works on page-locked memory and returns an error if a pointer to pageable memory
            /// is passed as input.
            /// </summary>
            /// <param name="dstHost">Destination host pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>    
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoHAsync_v2" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpyDtoHAsync_v2([Out] IntPtr dstHost, CUdeviceptr srcDevice, SizeT ByteCount, CUstream hStream);

            // device <-> device memory
            /// <summary>
            /// Copies from device memory to device memory. <c>dstDevice</c> and <c>srcDevice</c> are the base pointers of the destination
            /// and source, respectively. <c>ByteCount</c> specifies the number of bytes to copy. Note that this function is asynchronous
            /// and can optionally be associated to a stream by passing a non-zero <c>hStream</c> argument.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="srcDevice">Source device pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>    
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyDtoDAsync_v2" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, SizeT ByteCount, CUstream hStream);

            // system <-> array memory
            /// <summary>
            /// Copies from host memory to a 1D CUDA array. <c>dstArray</c> and <c>dstOffset</c> specify the CUDA array handle and
            /// starting offset in bytes of the destination data. <c>srcHost</c> specifies the base address of the source. <c>ByteCount</c>
            /// specifies the number of bytes to copy.<para/>
            /// <see cref="cuMemcpyHtoAAsync_v2(CUarray, SizeT, IntPtr, SizeT, CUstream)"/> is asynchronous and can optionally be associated to a stream by passing a non-zero
            /// <c>hStream</c> argument. It only works on page-locked memory and returns an error if a pointer to pageable memory
            /// is passed as input.
            /// </summary>
            /// <param name="dstArray">Destination array</param>
            /// <param name="dstOffset">Offset in bytes of destination array</param>
            /// <param name="srcHost">Source host pointer</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>    
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyHtoAAsync_v2" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpyHtoAAsync_v2(CUarray dstArray, SizeT dstOffset, [In] IntPtr srcHost, SizeT ByteCount, CUstream hStream);

            //Array -> Host
            /// <summary>
            /// Copies from one 1D CUDA array to host memory. <c>dstHost</c> specifies the base pointer of the destination. <c>srcArray</c>
            /// and <c>srcOffset</c> specify the CUDA array handle and starting offset in bytes of the source data. <c>ByteCount</c> specifies
            /// the number of bytes to copy.<para/>
            /// <see cref="cuMemcpyAtoHAsync_v2(IntPtr, CUarray, SizeT, SizeT, CUstream)"/> is asynchronous and can optionally be associated to a stream by passing a non-zero stream <c>hStream</c>
            /// argument. It only works on page-locked host memory and returns an error if a pointer to pageable memory is passed
            /// as input.
            /// </summary>
            /// <param name="dstHost">Destination pointer</param>
            /// <param name="srcArray">Source array</param>
            /// <param name="srcOffset">Offset in bytes of source array</param>
            /// <param name="ByteCount">Size of memory copy in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>    
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpyAtoHAsync_v2" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpyAtoHAsync_v2([Out] IntPtr dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount, CUstream hStream);

            // 2D memcpy
            /// <summary>
            /// Perform a 2D memory copy according to the parameters specified in <c>pCopy</c>. See <see cref="CUDAMemCpy2D"/>.
            /// <see cref="cuMemcpy2DAsync_v2"/> returns an error if any pitch is greater than the maximum allowed (<see cref="CUDeviceProperties.memPitch"/>).
			/// <see cref="MemoryManagement.cuMemAllocPitch_v2"/> passes back pitches that always work with <see cref="cuMemcpy2DAsync_v2"/>. On intra-device
            /// memory copies (device <![CDATA[<->]]> device, CUDA array <![CDATA[<->]]> device, CUDA array <![CDATA[<->]]> CUDA array), <see cref="cuMemcpy2DAsync_v2"/> may fail
			/// for pitches not computed by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>. <see cref="SynchronousMemcpy_v2.cuMemcpy2DUnaligned_v2"/> (not async!) does not have this restriction, but
            /// may run significantly slower in the cases where <see cref="cuMemcpy2DAsync_v2"/> would have returned an error code.
            /// </summary>
            /// <param name="pCopy">Parameters for the memory copy</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy2DAsync_v2" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpy2DAsync_v2(ref CUDAMemCpy2D pCopy, CUstream hStream);

            // 3D memcpy
            /// <summary>
            /// Perform a 3D memory copy according to the parameters specified in <c>pCopy</c>. See <see cref="CUDAMemCpy3D"/>.
            /// <see cref="cuMemcpy3DAsync_v2"/> returns an error if any pitch is greater than the maximum allowed (<see cref="CUDeviceProperties.memPitch"/>).<para/>
            /// <see cref="cuMemcpy3DAsync_v2"/> is asynchronous and can optionally be associated to a stream by passing a non-zero <c>hStream</c>
            /// argument. It only works on page-locked host memory and returns an error if a pointer to pageable memory is passed
            /// as input. <para/>
            /// The srcLOD and dstLOD members of the CUDAMemCpy3D structure must be set to 0.
            /// </summary>
            /// <param name="pCopy">Parameters for the memory copy</param>
            /// <param name="hStream">Stream indetifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>   
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemcpy3DAsync_v2" + CUDA_PTSZ)]
            public static extern CUResult cuMemcpy3DAsync_v2(ref CUDAMemCpy3D pCopy, CUstream hStream);
        }
        #endregion

        #region Memset
        /// <summary>
        /// Combines all memset API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class Memset
        {
            /// <summary>
            /// Sets the memory range of <c>N</c> 8-bit values to the specified value <c>b</c>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="b">Value to set</param>
            /// <param name="N">Number of elements</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD8_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemsetD8_v2(CUdeviceptr dstDevice, byte b, SizeT N);

            /// <summary>
            /// Sets the memory range of <c>N</c> 16-bit values to the specified value <c>us</c>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="us">Value to set</param>
            /// <param name="N">Number of elements</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD16_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemsetD16_v2(CUdeviceptr dstDevice, ushort us, SizeT N);

            /// <summary>
            /// Sets the memory range of <c>N</c> 32-bit values to the specified value <c>ui</c>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="ui">Value to set</param>
            /// <param name="N">Number of elements</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD32_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemsetD32_v2(CUdeviceptr dstDevice, uint ui, SizeT N);

            /// <summary>
            /// Sets the 2D memory range of <c>Width</c> 8-bit values to the specified value <c>b</c>. <c>Height</c> specifies the number of rows to
            /// set, and <c>dstPitch</c> specifies the number of bytes between each row. This function performs fastest when the pitch is
			/// one that has been passed back by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstPitch">Pitch of destination device pointer</param>
            /// <param name="b">Value to set</param>
            /// <param name="Width">Width of row</param>
            /// <param name="Height">Number of rows</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD2D8_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemsetD2D8_v2(CUdeviceptr dstDevice, SizeT dstPitch, byte b, SizeT Width, SizeT Height);

            /// <summary>
            /// Sets the 2D memory range of <c>Width</c> 16-bit values to the specified value <c>us</c>. <c>Height</c> specifies the number of rows to
            /// set, and <c>dstPitch</c> specifies the number of bytes between each row. This function performs fastest when the pitch is
			/// one that has been passed back by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstPitch">Pitch of destination device pointer</param>
            /// <param name="us">Value to set</param>
            /// <param name="Width">Width of row</param>
            /// <param name="Height">Number of rows</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD2D16_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemsetD2D16_v2(CUdeviceptr dstDevice, SizeT dstPitch, ushort us, SizeT Width, SizeT Height);

            /// <summary>
            /// Sets the 2D memory range of <c>Width</c> 32-bit values to the specified value <c>us</c>. <c>Height</c> specifies the number of rows to
            /// set, and <c>dstPitch</c> specifies the number of bytes between each row. This function performs fastest when the pitch is
			/// one that has been passed back by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstPitch">Pitch of destination device pointer</param>
            /// <param name="ui">Value to set</param>
            /// <param name="Width">Width of row</param>
            /// <param name="Height">Number of rows</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD2D32_v2" + CUDA_PTDS)]
            public static extern CUResult cuMemsetD2D32_v2(CUdeviceptr dstDevice, SizeT dstPitch, uint ui, SizeT Width, SizeT Height);
        }
        #endregion

        #region MemsetAsync
        /// <summary>
        /// Combines all async memset API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class MemsetAsync
        {
            /// <summary>
            /// Sets the memory range of <c>N</c> 8-bit values to the specified value <c>b</c>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="b">Value to set</param>
            /// <param name="N">Number of elements</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD8Async" + CUDA_PTSZ)]
            public static extern CUResult cuMemsetD8Async(CUdeviceptr dstDevice, byte b, SizeT N, CUstream hStream);

            /// <summary>
            /// Sets the memory range of <c>N</c> 16-bit values to the specified value <c>us</c>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="us">Value to set</param>
            /// <param name="N">Number of elements</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD16Async" + CUDA_PTSZ)]
            public static extern CUResult cuMemsetD16Async(CUdeviceptr dstDevice, ushort us, SizeT N, CUstream hStream);

            /// <summary>
            /// Sets the memory range of <c>N</c> 32-bit values to the specified value <c>ui</c>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="ui">Value to set</param>
            /// <param name="N">Number of elements</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD32Async" + CUDA_PTSZ)]
            public static extern CUResult cuMemsetD32Async(CUdeviceptr dstDevice, uint ui, SizeT N, CUstream hStream);

            /// <summary>
            /// Sets the 2D memory range of <c>Width</c> 8-bit values to the specified value <c>b</c>. <c>Height</c> specifies the number of rows to
            /// set, and <c>dstPitch</c> specifies the number of bytes between each row. This function performs fastest when the pitch is
			/// one that has been passed back by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstPitch">Pitch of destination device pointer</param>
            /// <param name="b">Value to set</param>
            /// <param name="Width">Width of row</param>
            /// <param name="Height">Number of rows</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD2D8Async" + CUDA_PTSZ)]
            public static extern CUResult cuMemsetD2D8Async(CUdeviceptr dstDevice, SizeT dstPitch, byte b, SizeT Width, SizeT Height, CUstream hStream);

            /// <summary>
            /// Sets the 2D memory range of <c>Width</c> 16-bit values to the specified value <c>us</c>. <c>Height</c> specifies the number of rows to
            /// set, and <c>dstPitch</c> specifies the number of bytes between each row. This function performs fastest when the pitch is
			/// one that has been passed back by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstPitch">Pitch of destination device pointer</param>
            /// <param name="us">Value to set</param>
            /// <param name="Width">Width of row</param>
            /// <param name="Height">Number of rows</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD2D16Async" + CUDA_PTSZ)]
            public static extern CUResult cuMemsetD2D16Async(CUdeviceptr dstDevice, SizeT dstPitch, ushort us, SizeT Width, SizeT Height, CUstream hStream);

            /// <summary>
            /// Sets the 2D memory range of <c>Width</c> 32-bit values to the specified value <c>us</c>. <c>Height</c> specifies the number of rows to
            /// set, and <c>dstPitch</c> specifies the number of bytes between each row. This function performs fastest when the pitch is
			/// one that has been passed back by <see cref="MemoryManagement.cuMemAllocPitch_v2"/>.
            /// </summary>
            /// <param name="dstDevice">Destination device pointer</param>
            /// <param name="dstPitch">Pitch of destination device pointer</param>
            /// <param name="ui">Value to set</param>
            /// <param name="Width">Width of row</param>
            /// <param name="Height">Number of rows</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuMemsetD2D32Async" + CUDA_PTSZ)]
            public static extern CUResult cuMemsetD2D32Async(CUdeviceptr dstDevice, SizeT dstPitch, uint ui, SizeT Width, SizeT Height, CUstream hStream);
        }
        #endregion

        #region Function management
        /// <summary>
        /// Combines all function / kernel API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class FunctionManagement
        {
            /// <summary>
            /// Specifies the <c>x</c>, <c>y</c>, and <c>z</c> dimensions of the thread blocks that are created when the kernel given by <c>hfunc</c> is launched.
            /// </summary>
            /// <param name="hfunc">Kernel to specify dimensions of</param>
            /// <param name="x">X dimension</param>
            /// <param name="y">Y dimension</param>
            /// <param name="z">Z dimension</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
            
            /// <summary>
            /// Sets through <c>bytes</c> the amount of dynamic shared memory that will be available to each thread block when the kernel
            /// given by <c>hfunc</c> is launched.
            /// </summary>
            /// <param name="hfunc">Kernel to specify dynamic shared-memory size for</param>
            /// <param name="bytes">Dynamic shared-memory size per thread in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuFuncSetSharedSize(CUfunction hfunc, uint bytes);
            
            /// <summary>
            /// Returns in <c>pi</c> the integer value of the attribute <c>attrib</c> on the kernel given by <c>hfunc</c>. See <see cref="CUFunctionAttribute"/>.
            /// </summary>
            /// <param name="pi">Returned attribute value</param>
            /// <param name="attrib">Attribute requested</param>
            /// <param name="hfunc">Function to query attribute of</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuFuncGetAttribute(ref int pi, CUFunctionAttribute attrib, CUfunction hfunc);
            
            /// <summary>
            /// On devices where the L1 cache and shared memory use the same hardware resources, this sets through <c>config</c>
            /// the preferred cache configuration for the device function <c>hfunc</c>. This is only a preference. The driver will use the
            /// requested configuration if possible, but it is free to choose a different configuration if required to execute <c>hfunc</c>. <para/>
            /// This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.<para/>
            /// Switching between configuration modes may insert a device-side synchronization point for streamed kernel launches.<para/>
            /// The supported cache modes are defined in <see cref="CUFuncCache"/>
            /// </summary>
            /// <param name="hfunc">Kernel to configure cache for</param>
            /// <param name="config">Requested cache configuration</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuFuncSetCacheConfig(CUfunction hfunc, CUFuncCache config);
			
			/// <summary>
			/// Sets the shared memory configuration for a device function.<para/>
			/// On devices with configurable shared memory banks, this function will 
			/// force all subsequent launches of the specified device function to have
			/// the given shared memory bank size configuration. On any given launch of the
			/// function, the shared memory configuration of the device will be temporarily
			/// changed if needed to suit the function's preferred configuration. Changes in
			/// shared memory configuration between subsequent launches of functions, 
			/// may introduce a device side synchronization point.<para/>
			/// Any per-function setting of shared memory bank size set via
			/// <see cref="cuFuncSetSharedMemConfig"/>  will override the context wide setting set with
			/// <see cref="DriverAPINativeMethods.ContextManagement.cuCtxSetSharedMemConfig"/>.<para/>
			/// Changing the shared memory bank size will not increase shared memory usage
			/// or affect occupancy of kernels, but may have major effects on performance. 
			/// Larger bank sizes will allow for greater potential bandwidth to shared memory,
			/// but will change what kinds of accesses to shared memory will result in bank 
			/// conflicts.<para/>
			/// This function will do nothing on devices with fixed shared memory bank size.<para/>
			/// The supported bank configurations are<para/> 
			/// - <see cref="CUsharedconfig.DefaultBankSize"/>: set bank width to the default initial
			///   setting (currently, four bytes).
			/// - <see cref="CUsharedconfig.FourByteBankSize"/>: set shared memory bank width to
			///   be natively four bytes.
			/// - <see cref="CUsharedconfig.EightByteBankSize"/>: set shared memory bank width to
			///   be natively eight bytes.
			/// </summary>
			/// <param name="hfunc">kernel to be given a shared memory config</param>
			/// <param name="config">requested shared memory configuration</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);

        }
        #endregion

        #region Array management
        /// <summary>
        /// Combines all array management API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class ArrayManagement
        {            
            /// <summary>
            /// Creates a CUDA array according to the <see cref="CUDAArrayDescriptor"/> structure <c>pAllocateArray</c> and returns a
            /// handle to the new CUDA array in <c>pHandle</c>.
            /// </summary>
            /// <param name="pHandle">Returned array</param>
            /// <param name="pAllocateArray">Array descriptor</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuArrayCreate_v2(ref CUarray pHandle, ref CUDAArrayDescriptor pAllocateArray );
            
            /// <summary>
            /// Returns in <c>pArrayDescriptor</c> a descriptor containing information on the format and dimensions of the CUDA
            /// array <c>hArray</c>. It is useful for subroutines that have been passed a CUDA array, but need to know the CUDA array
            /// parameters for validation or other purposes.
            /// </summary>
            /// <param name="pArrayDescriptor">Returned array descriptor</param>
            /// <param name="hArray">Array to get descriptor of</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuArrayGetDescriptor_v2( ref CUDAArrayDescriptor pArrayDescriptor, CUarray hArray );
            
            /// <summary>
            /// Destroys the CUDA array hArray.
            /// </summary>
            /// <param name="hArray">Array to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorArrayIsMapped"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuArrayDestroy( CUarray hArray );

            /// <summary>
            /// Creates a CUDA array according to the <see cref="CUDAArray3DDescriptor"/> structure <c>pAllocateArray</c> and returns
            /// a handle to the new CUDA array in <c>pHandle</c>.
            /// </summary>
            /// <param name="pHandle">Returned array</param>
            /// <param name="pAllocateArray">3D array descriptor</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuArray3DCreate_v2(ref CUarray pHandle, ref CUDAArray3DDescriptor pAllocateArray);

            /// <summary>
            /// Returns in <c>pArrayDescriptor</c> a descriptor containing information on the format and dimensions of the CUDA
            /// array <c>hArray</c>. It is useful for subroutines that have been passed a CUDA array, but need to know the CUDA array
            /// parameters for validation or other purposes.<para/>
            /// This function may be called on 1D and 2D arrays, in which case the Height and/or Depth members of the descriptor
            /// struct will be set to 0.
            /// </summary>
            /// <param name="pArrayDescriptor">Returned 3D array descriptor</param>
            /// <param name="hArray">3D array to get descriptor of</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuArray3DGetDescriptor_v2(ref CUDAArray3DDescriptor pArrayDescriptor, CUarray hArray);

			/// <summary>
			/// Creates a CUDA mipmapped array according to the ::CUDA_ARRAY3D_DESCRIPTOR structure
			/// <c>pMipmappedArrayDesc</c> and returns a handle to the new CUDA mipmapped array in <c>pHandle</c>.
			/// <c>numMipmapLevels</c> specifies the number of mipmap levels to be allocated. This value is
			/// clamped to the range [1, 1 + floor(log2(max(width, height, depth)))]. 
			/// </summary>
			/// <param name="pHandle">Returned mipmapped array</param>
			/// <param name="pMipmappedArrayDesc">mipmapped array descriptor</param>
			/// <param name="numMipmapLevels">Number of mipmap levels</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>, <see cref="CUResult.ErrorUnknown"/>. </returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuMipmappedArrayCreate(ref CUmipmappedArray pHandle, ref CUDAArray3DDescriptor pMipmappedArrayDesc, uint numMipmapLevels);

			/// <summary>
			/// Returns in <c>pLevelArray</c> a CUDA array that represents a single mipmap level
			/// of the CUDA mipmapped array <c>hMipmappedArray</c>.
			/// </summary>
			/// <param name="pLevelArray">Returned mipmap level CUDA array</param>
			/// <param name="hMipmappedArray">CUDA mipmapped array</param>
			/// <param name="level">Mipmap level</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>.</returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuMipmappedArrayGetLevel(ref CUarray pLevelArray, CUmipmappedArray hMipmappedArray, uint level);

			/// <summary>
			/// Destroys the CUDA mipmapped array <c>hMipmappedArray</c>.
			/// </summary>
			/// <param name="hMipmappedArray">Mipmapped array to destroy</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>.</returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);


        }
        #endregion

        #region Texture reference management
        /// <summary>
        /// Groups all texture reference management API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class TextureReferenceManagement
        {
            /// <summary>
            /// Creates a texture reference and returns its handle in <c>pTexRef</c>. Once created, the application must call <see cref="cuTexRefSetArray"/>
			/// or <see cref="cuTexRefSetAddress_v2"/> to associate the reference with allocated memory. Other texture reference functions
            /// are used to specify the format and interpretation (addressing, filtering, etc.) to be used when the memory is read
            /// through this texture reference. To associate the texture reference with a texture ordinal for a given function, the
            /// application should call <see cref="ParameterManagement.cuParamSetTexRef"/>.
            /// </summary>
            /// <param name="pTexRef">Returned texture reference</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefCreate( ref CUtexref pTexRef );
            
            /// <summary>
            /// Destroys the texture reference specified by <c>hTexRef</c>.
            /// </summary>
            /// <param name="hTexRef">Texture reference to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefDestroy( CUtexref hTexRef );
    
            /// <summary>
            /// Binds the CUDA array <c>hArray</c> to the texture reference <c>hTexRef</c>. Any previous address or CUDA array state
            /// associated with the texture reference is superseded by this function. Flags must be set to 
            /// <see cref="CUTexRefSetArrayFlags.OverrideFormat"/>. Any CUDA array previously bound to hTexRef is unbound.
            /// </summary>
            /// <param name="hTexRef">Texture reference to bind</param>
            /// <param name="hArray">Array to bind</param>
            /// <param name="Flags">Options (must be <see cref="CUTexRefSetArrayFlags.OverrideFormat"/>)</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, CUTexRefSetArrayFlags Flags);
			
			/// <summary>
			/// Binds the CUDA mipmapped array <c>hMipmappedArray</c> to the texture reference <c>hTexRef</c>.
			/// Any previous address or CUDA array state associated with the texture reference
			/// is superseded by this function. <c>Flags</c> must be set to <see cref="CUTexRefSetArrayFlags.OverrideFormat"/>. 
			/// Any CUDA array previously bound to <c>hTexRef</c> is unbound.
			/// </summary>
			/// <param name="hTexRef">Texture reference to bind</param>
			/// <param name="hMipmappedArray">Mipmapped array to bind</param>
			/// <param name="Flags">Options (must be <see cref="CUTexRefSetArrayFlags.OverrideFormat"/>)</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
			/// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, CUTexRefSetArrayFlags Flags);

            /// <summary>
            /// Binds a linear address range to the texture reference <c>hTexRef</c>. Any previous address or CUDA array state associated
            /// with the texture reference is superseded by this function. Any memory previously bound to <c>hTexRef</c> is unbound.<para/>
			/// Since the hardware enforces an alignment requirement on texture base addresses, <see cref="cuTexRefSetAddress_v2"/> passes back
            /// a byte offset in <c>ByteOffset</c> that must be applied to texture fetches in order to read from the desired memory. This
            /// offset must be divided by the texel size and passed to kernels that read from the texture so they can be applied to the
            /// <c>tex1Dfetch()</c> function.<para/>
			/// If the device memory pointer was returned from <see cref="MemoryManagement.cuMemAlloc_v2"/>, the offset is guaranteed to be 0 and <c>null</c> may be
            /// passed as the <c>ByteOffset</c> parameter.
            /// </summary>
            /// <param name="ByteOffset">Returned byte offset</param>
            /// <param name="hTexRef">Texture reference to bind</param>
            /// <param name="dptr">Device pointer to bind</param>
            /// <param name="bytes">Size of memory to bind in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefSetAddress_v2(ref SizeT ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, SizeT bytes);

            /// <summary>
            /// Binds a linear address range to the texture reference <c>hTexRef</c>. Any previous address or CUDA array state associated
            /// with the texture reference is superseded by this function. Any memory previously bound to <c>hTexRef</c> is unbound. <para/>
            /// Using a <c>tex2D()</c> function inside a kernel requires a call to either <see cref="cuTexRefSetArray"/> to bind the corresponding texture
			/// reference to an array, or <see cref="cuTexRefSetAddress2D_v2"/> to bind the texture reference to linear memory.<para/>
			/// Function calls to <see cref="cuTexRefSetFormat"/> cannot follow calls to <see cref="cuTexRefSetAddress2D_v2"/> for the same texture reference.<para/>
            /// It is required that <c>dptr</c> be aligned to the appropriate hardware-specific texture alignment. You can query this value
            /// using the device attribute <see cref="CUDeviceAttribute.TextureAlignment"/>. If an unaligned <c>dptr</c> is supplied,
            /// <see cref="CUResult.ErrorInvalidValue"/> is returned.
            /// </summary>
            /// <param name="hTexRef">Texture reference to bind</param>
            /// <param name="desc">Descriptor of CUDA array</param>
            /// <param name="dptr">Device pointer to bind</param>
            /// <param name="Pitch">Line pitch in bytes></param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefSetAddress2D_v2(CUtexref hTexRef, ref CUDAArrayDescriptor desc, CUdeviceptr dptr, SizeT Pitch);
           
            /// <summary>
            /// Specifies the format of the data to be read by the texture reference <c>hTexRef</c>. <c>fmt</c> and <c>NumPackedComponents</c>
            /// are exactly analogous to the Format and NumChannels members of the <see cref="CUDAArrayDescriptor"/> structure:
            /// They specify the format of each component and the number of components per array element.
            /// </summary>
            /// <param name="hTexRef">Texture reference</param>
            /// <param name="fmt">Format to set</param>
            /// <param name="NumPackedComponents">Number of components per array element</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefSetFormat( CUtexref hTexRef, CUArrayFormat fmt, int NumPackedComponents );
            
            /// <summary>
            /// Specifies the addressing mode <c>am</c> for the given dimension <c>dim</c> of the texture reference <c>hTexRef</c>. If <c>dim</c> is zero,
            /// the addressing mode is applied to the first parameter of the functions used to fetch from the texture; if <c>dim</c> is 1, the
            /// second, and so on. See <see cref="CUAddressMode"/>.<para/>
            /// Note that this call has no effect if <c>hTexRef</c> is bound to linear memory.
            /// </summary>
            /// <param name="hTexRef">Texture reference</param>
            /// <param name="dim">Dimension</param>
            /// <param name="am">Addressing mode to set</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUAddressMode am );
            
            /// <summary>
            /// Specifies the filtering mode <c>fm</c> to be used when reading memory through the texture reference <c>hTexRef</c>. See <see cref="CUFilterMode"/>.<para/>
            /// Note that this call has no effect if hTexRef is bound to linear memory.
            /// </summary>
            /// <param name="hTexRef">Texture reference</param>
            /// <param name="fm">Filtering mode to set</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefSetFilterMode( CUtexref hTexRef, CUFilterMode fm );

            /// <summary>
            /// Specifies optional flags via <c>Flags</c> to specify the behavior of data returned through the texture reference <c>hTexRef</c>. See <see cref="CUTexRefSetFlags"/>.
            /// </summary>
            /// <param name="hTexRef">Texture reference</param>
            /// <param name="Flags">Optional flags to set</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefSetFlags(CUtexref hTexRef, CUTexRefSetFlags Flags);

            /// <summary>
            /// Returns in <c>pdptr</c> the base address bound to the texture reference <c>hTexRef</c>, or returns <see cref="CUResult.ErrorInvalidValue"/>
            /// if the texture reference is not bound to any device memory range.
            /// </summary>
            /// <param name="pdptr">Returned device address</param>
            /// <param name="hTexRef">Texture reference</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefGetAddress( ref CUdeviceptr pdptr, CUtexref hTexRef );
            
            /// <summary>
            /// Returns in <c>phArray</c> the CUDA array bound to the texture reference <c>hTexRef</c>, or returns <see cref="CUResult.ErrorInvalidValue"/>
            /// if the texture reference is not bound to any CUDA array.
            /// </summary>
            /// <param name="phArray">Returned array</param>
            /// <param name="hTexRef">Texture reference</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefGetArray( ref CUarray phArray, CUtexref hTexRef );

			/// <summary>
			/// Returns in <c>phMipmappedArray</c> the CUDA mipmapped array bound to the texture 
			/// reference <c>hTexRef</c>, or returns <see cref="CUResult.ErrorInvalidValue"/> if the texture reference
			/// is not bound to any CUDA mipmapped array.
			/// </summary>
			/// <param name="phMipmappedArray">Returned mipmapped array</param>
			/// <param name="hTexRef">Texture reference</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefGetMipmappedArray(ref CUmipmappedArray phMipmappedArray, CUtexref hTexRef);
			
            /// <summary>
            /// Returns in <c>pam</c> the addressing mode corresponding to the dimension <c>dim</c> of the texture reference <c>hTexRef</c>. Currently,
            /// the only valid value for <c>dim</c> are 0 and 1.
            /// </summary>
            /// <param name="pam">Returned addressing mode</param>
            /// <param name="hTexRef">Texture reference</param>
            /// <param name="dim">Dimension</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefGetAddressMode( ref CUAddressMode pam, CUtexref hTexRef, int dim );

            /// <summary>
            /// Returns in <c>pfm</c> the filtering mode of the texture reference <c>hTexRef</c>.
            /// </summary>
            /// <param name="pfm">Returned filtering mode</param>
            /// <param name="hTexRef">Texture reference</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefGetFilterMode( ref CUFilterMode pfm, CUtexref hTexRef );

            /// <summary>
            /// Returns in <c>pFormat</c> and <c>pNumChannels</c> the format and number of components of the CUDA array bound to
            /// the texture reference <c>hTexRef</c>. If <c>pFormat</c> or <c>pNumChannels</c> is <c>null</c>, it will be ignored.
            /// </summary>
            /// <param name="pFormat">Returned format</param>
            /// <param name="pNumChannels">Returned number of components</param>
            /// <param name="hTexRef">Texture reference</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefGetFormat( ref CUArrayFormat pFormat, ref int pNumChannels, CUtexref hTexRef );

            /// <summary>
            /// Returns in <c>pFlags</c> the flags of the texture reference <c>hTexRef</c>.
            /// </summary>
            /// <param name="pFlags">Returned flags</param>
            /// <param name="hTexRef">Texture reference</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuTexRefGetFlags(ref CUTexRefSetFlags pFlags, CUtexref hTexRef);

			/// <summary>
			/// Returns the mipmap filtering mode in <c>pfm</c> that's used when reading memory through
			/// the texture reference <c>hTexRef</c>.
			/// </summary>
			/// <param name="pfm">Returned mipmap filtering mode</param>
			/// <param name="hTexRef">Texture reference</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefGetMipmapFilterMode(ref CUFilterMode pfm, CUtexref hTexRef);
			
			/// <summary>
			/// Returns the mipmap level bias in <c>pBias</c> that's added to the specified mipmap
			/// level when reading memory through the texture reference <c>hTexRef</c>.
			/// </summary>
			/// <param name="pbias">Returned mipmap level bias</param>
			/// <param name="hTexRef">Texture reference</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefGetMipmapLevelBias(ref float pbias, CUtexref hTexRef);
			
			/// <summary>
			/// Returns the min/max mipmap level clamps in <c>pminMipmapLevelClamp</c> and <c>pmaxMipmapLevelClamp</c>
			/// that's used when reading memory through the texture reference <c>hTexRef</c>. 
			/// </summary>
			/// <param name="pminMipmapLevelClamp">Returned mipmap min level clamp</param>
			/// <param name="pmaxMipmapLevelClamp">Returned mipmap max level clamp</param>
			/// <param name="hTexRef">Texture reference</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefGetMipmapLevelClamp(ref float pminMipmapLevelClamp, ref float pmaxMipmapLevelClamp, CUtexref hTexRef);
			
			/// <summary>
			/// Returns the maximum aniostropy in <c>pmaxAniso</c> that's used when reading memory through
			/// the texture reference. 
			/// </summary>
			/// <param name="pmaxAniso">Returned maximum anisotropy</param>
			/// <param name="hTexRef">Texture reference</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefGetMaxAnisotropy(ref int pmaxAniso, CUtexref hTexRef);

			/// <summary>
			/// Specifies the mipmap filtering mode <c>fm</c> to be used when reading memory through
			/// the texture reference <c>hTexRef</c>.<para/>
			/// Note that this call has no effect if <c>hTexRef</c> is not bound to a mipmapped array.
			/// </summary>
			/// <param name="hTexRef">Texture reference</param>
			/// <param name="fm">Filtering mode to set</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUFilterMode fm);
			
			/// <summary>
			/// Specifies the mipmap level bias <c>bias</c> to be added to the specified mipmap level when 
			/// reading memory through the texture reference <c>hTexRef</c>.<para/>
			/// Note that this call has no effect if <c>hTexRef</c> is not bound to a mipmapped array.
			/// </summary>
			/// <param name="hTexRef">Texture reference</param>
			/// <param name="bias">Mipmap level bias</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
			
			/// <summary>
			/// Specifies the min/max mipmap level clamps, <c>minMipmapLevelClamp</c> and <c>maxMipmapLevelClamp</c>
			/// respectively, to be used when reading memory through the texture reference 
			/// <c>hTexRef</c>.<para/>
			/// Note that this call has no effect if <c>hTexRef</c> is not bound to a mipmapped array.
			/// </summary>
			/// <param name="hTexRef">Texture reference</param>
			/// <param name="minMipmapLevelClamp">Mipmap min level clamp</param>
			/// <param name="maxMipmapLevelClamp">Mipmap max level clamp</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
			
			/// <summary>
			/// Specifies the maximum aniostropy <c>maxAniso</c> to be used when reading memory through
			/// the texture reference <c>hTexRef</c>. <para/>
			/// Note that this call has no effect if <c>hTexRef</c> is not bound to a mipmapped array.
			/// </summary>
			/// <param name="hTexRef">Texture reference</param>
			/// <param name="maxAniso">Maximum anisotropy</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, uint maxAniso);

        }
        #endregion

        #region Surface reference management
        /// <summary>
        /// Combines all surface management API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class SurfaceReferenceManagement
        {
            /// <summary>
            /// Sets the CUDA array <c>hArray</c> to be read and written by the surface reference <c>hSurfRef</c>. Any previous CUDA array
            /// state associated with the surface reference is superseded by this function. Flags must be set to <see cref="CUSurfRefSetFlags.None"/>. The 
            /// <see cref="CUDAArray3DFlags.SurfaceLDST"/> flag must have been set for the CUDA array. Any CUDA array previously bound to
            /// <c>hSurfRef</c> is unbound.
            /// </summary>
            /// <param name="hSurfRef">Surface reference handle</param>
            /// <param name="hArray">CUDA array handle</param>
            /// <param name="Flags">set to <see cref="CUSurfRefSetFlags.None"/></param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, CUSurfRefSetFlags Flags);

            /// <summary>
            /// Returns in <c>phArray</c> the CUDA array bound to the surface reference <c>hSurfRef</c>, or returns
            /// <see cref="CUResult.ErrorInvalidValue"/> if the surface reference is not bound to any CUDA array.
            /// </summary>
            /// <param name="phArray">Surface reference handle</param>
            /// <param name="hSurfRef">Surface reference handle</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuSurfRefGetArray( ref CUarray phArray, CUsurfref hSurfRef );
        }
        #endregion

        #region Parameter management
        /// <summary>
        /// Combines all kernel / function parameter management API calls
        /// </summary>
        [Obsolete(CUDA_OBSOLET_4_0)]
        public static class ParameterManagement
        {
            /// <summary>
            /// Sets through <c>numbytes</c> the total size in bytes needed by the function parameters of the kernel corresponding to
            /// <c>hfunc</c>.
            /// </summary>
            /// <param name="hfunc">Kernel to set parameter size for</param>
            /// <param name="numbytes">Size of parameter list in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetSize([In] CUfunction hfunc, [In] uint numbytes);
            
            /// <summary>
            /// Sets an integer parameter that will be specified the next time the kernel corresponding to <c>hfunc</c> will be invoked.
            /// <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add parameter to</param>
            /// <param name="offset">Offset to add parameter to argument list</param>
            /// <param name="value">Value of parameter</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSeti([In] CUfunction hfunc, [In] int offset, [In] uint value);

            /// <summary>
            /// Sets a floating-point parameter that will be specified the next time the kernel corresponding to <c>hfunc</c> will be invoked.
            /// <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add parameter to</param>
            /// <param name="offset">Offset to add parameter to argument list</param>
            /// <param name="value">Value of parameter</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetf([In] CUfunction hfunc, [In] int offset, [In] float value);

            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] IntPtr ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref byte ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref sbyte ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ushort ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref short ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uint ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref int ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ulong ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref long ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref float ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref double ptr, [In] uint numbytes);

            #region VectorTypes
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref dim3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref char1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref char2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref char3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref char4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uchar1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uchar2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uchar3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uchar4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref short1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref short2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref short3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref short4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ushort1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ushort2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ushort3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ushort4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref int1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref int2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref int3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref int4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uint1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uint2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uint3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref uint4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref long1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref long2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref long3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref long4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ulong1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ulong2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ulong3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref ulong4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref float1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref float2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref float3 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref float4 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref double1 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref double2 ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref cuDoubleComplex ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref cuDoubleReal ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref cuFloatComplex ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] ref cuFloatReal ptr, [In] uint numbytes);
            #endregion

            #region VectorTypesArrays
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] dim3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] char1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] char2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] char3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] char4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uchar1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uchar2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uchar3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uchar4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] short1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] short2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] short3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] short4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ushort1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ushort2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ushort3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ushort4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] int1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] int2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] int3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] int4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uint1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uint2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uint3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uint4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] long1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] long2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] long3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] long4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ulong1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ulong2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ulong3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ulong4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] float1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] float2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] float3[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] float4[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] double1[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] double2[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] cuDoubleComplex[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] cuDoubleReal[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] cuFloatComplex[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] cuFloatReal[] ptr, uint numbytes);
            #endregion

            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] byte[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] sbyte[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ushort[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] short[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] uint[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] int[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv(CUfunction hfunc, int offset, [In] ulong[] ptr, uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] long[] ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] float[] ptr, [In] uint numbytes);
            /// <summary>
            /// Copies an arbitrary amount of data (specified in <c>numbytes</c>) from <c>ptr</c> into the parameter space of the kernel corresponding
            /// to <c>hfunc</c>. <c>offset</c> is a byte offset.
            /// </summary>
            /// <param name="hfunc">Kernel to add data to</param>
            /// <param name="offset">Offset to add data to argument list</param>
            /// <param name="ptr">Pointer to arbitrary data</param>
            /// <param name="numbytes">Size of data to copy in bytes</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuParamSetv([In] CUfunction hfunc, [In] int offset, [In] double[] ptr, [In] uint numbytes);
            
            /// <summary>
            /// Makes the CUDA array or linear memory bound to the texture reference <c>hTexRef</c> available to a device program as a
            /// texture. In this version of CUDA, the texture-reference must be obtained via <see cref="ModuleManagement.cuModuleGetTexRef"/> and the <c>texunit</c>
            /// parameter must be set to <see cref="CUParameterTexRef.Default"/>.
            /// </summary>
            /// <param name="hfunc">Kernel to add texture-reference to</param>
            /// <param name="texunit">Texture unit (must be <see cref="CUParameterTexRef.Default"/>)</param>
            /// <param name="hTexRef">Texture-reference to add to argument list</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete("cuParamSetTexRef() has been deprecated in CUDA Toolkit 3.2 since this API entry provided no functionality.")]
            public static extern CUResult cuParamSetTexRef([In] CUfunction hfunc, [In] CUParameterTexRef texunit, [In] CUtexref hTexRef);
        }
        #endregion

        #region Launch functions
        /// <summary>
        /// Groups all kernel launch API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class Launch
        {
            /// <summary>
            /// Invokes the kernel <c>f</c> on a 1 x 1 x 1 grid of blocks. The block contains the number of threads specified by a previous
            /// call to <see cref="FunctionManagement.cuFuncSetBlockShape"/>.
            /// </summary>
            /// <param name="f">Kernel to launch</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>
            /// <see cref="CUResult.ErrorLaunchFailed"/>, <see cref="CUResult.ErrorLaunchOutOfResources"/>
            /// <see cref="CUResult.ErrorLaunchTimeout"/>, <see cref="CUResult.ErrorLaunchIncompatibleTexturing"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuLaunch([In] CUfunction f);
            
            /// <summary>
            /// Invokes the kernel <c>f</c> on a <c>grid_width</c> x <c>grid_height</c> grid of blocks. Each block contains the number of threads
            /// specified by a previous call to <see cref="FunctionManagement.cuFuncSetBlockShape"/>.
            /// </summary>
            /// <param name="f">Kernel to launch</param>
            /// <param name="grid_width">Width of grid in blocks</param>
            /// <param name="grid_height">Height of grid in blocks</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>
            /// <see cref="CUResult.ErrorLaunchFailed"/>, <see cref="CUResult.ErrorLaunchOutOfResources"/>
            /// <see cref="CUResult.ErrorLaunchTimeout"/>, <see cref="CUResult.ErrorLaunchIncompatibleTexturing"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuLaunchGrid([In] CUfunction f, [In] int grid_width, [In] int grid_height);
            
            /// <summary>
            /// Invokes the kernel <c>f</c> on a <c>grid_width</c> x <c>grid_height</c> grid of blocks. Each block contains the number of threads
            /// specified by a previous call to <see cref="FunctionManagement.cuFuncSetBlockShape"/>.<para/>
            /// <see cref="cuLaunchGridAsync"/> can optionally be associated to a stream by passing a non-zero <c>hStream</c> argument.
            /// </summary>
            /// <param name="f">Kernel to launch</param>
            /// <param name="grid_width">Width of grid in blocks</param>
            /// <param name="grid_height">Height of grid in blocks</param>
            /// <param name="hStream">Stream identifier</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>
            /// <see cref="CUResult.ErrorLaunchFailed"/>, <see cref="CUResult.ErrorLaunchOutOfResources"/>
            /// <see cref="CUResult.ErrorLaunchTimeout"/>, <see cref="CUResult.ErrorLaunchIncompatibleTexturing"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuLaunchGridAsync([In]  CUfunction f, [In]  int grid_width, [In] int grid_height, [In] CUstream hStream);

            /// <summary>
            /// Invokes the kernel <c>f</c> on a <c>gridDimX</c> x <c>gridDimY</c> x <c>gridDimZ</c>
            /// grid of blocks. Each block contains <c>blockDimX</c> x <c>blockDimY</c> x
            /// blockDimZ threads.
            /// <para/>
            /// <c>sharedMemBytes</c> sets the amount of dynamic shared memory that will be
            /// available to each thread block.
            /// <para/>
            /// <see cref="cuLaunchKernel"/> can optionally be associated to a stream by passing a
            /// non-zero <c>hStream</c> argument.
            /// <para/>
            /// Kernel parameters to <c>f</c> can be specified in one of two ways:
            /// <para/>
            /// 1) Kernel parameters can be specified via <c>kernelParams</c>. If <c>f</c>
            /// has N parameters, then <c>kernelParams</c> needs to be an array of N
            /// pointers. Each of <c>kernelParams[0]</c> through <c>kernelParams[N-1]</c>
            /// must point to a region of memory from which the actual kernel
            /// parameter will be copied.  The number of kernel parameters and their
            /// offsets and sizes do not need to be specified as that information is
            /// retrieved directly from the kernel's image.
            /// <para/>
            /// 2) Kernel parameters can also be packaged by the application into
            /// a single buffer that is passed in via the <c>extra</c> parameter.
            /// This places the burden on the application of knowing each kernel
            /// parameter's size and alignment/padding within the buffer.  
            /// 
            /// <para/>
            /// The <c>extra</c> parameter exists to allow <see cref="cuLaunchKernel"/> to take
            /// additional less commonly used arguments. <c>extra</c> specifies a list of
            /// names of extra settings and their corresponding values.  Each extra
            /// setting name is immediately followed by the corresponding value.  The
            /// list must be terminated with either NULL or ::CU_LAUNCH_PARAM_END.
            /// <para/>
            /// - ::CU_LAUNCH_PARAM_END, which indicates the end of the <c>extra</c>
            ///   array;
            /// - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
            ///   value in <c>extra</c> will be a pointer to a buffer containing all
            ///   the kernel parameters for launching kernel <c>f</c>;
            /// - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
            ///   value in <c>extra</c> will be a pointer to a size_t containing the
            ///   size of the buffer specified with ::CU_LAUNCH_PARAM_BUFFER_POINTER;
            /// <para/>
            /// The error ::CUDA_ERROR_INVALID_VALUE will be returned if kernel
            /// parameters are specified with both <c>kernelParams</c> and <c>extra</c>
            /// (i.e. both <c>kernelParams</c> and <c>extra</c> are non-NULL).
            /// <para/>
            /// Calling <see cref="cuLaunchKernel"/> sets persistent function state that is
            /// the same as function state set through the following deprecated APIs:
            ///
            ///  ::cuFuncSetBlockShape()
            ///  ::cuFuncSetSharedSize()
            ///  ::cuParamSetSize()
            ///  ::cuParamSeti()
            ///  ::cuParamSetf()
            ///  ::cuParamSetv()
            /// <para/>
            /// When the kernel <c>f</c> is launched via <see cref="cuLaunchKernel"/>, the previous
            /// block shape, shared size and parameter info associated with <c>f</c>
            /// is overwritten.
            /// <para/>
            /// Note that to use <see cref="cuLaunchKernel"/>, the kernel <c>f</c> must either have
            /// been compiled with toolchain version 3.2 or later so that it will
            /// contain kernel parameter information, or have no kernel parameters.
            /// If either of these conditions is not met, then <see cref="cuLaunchKernel"/> will
            /// return <see cref="CUResult.ErrorInvalidImage"/>.
            /// </summary>
            /// <param name="f">Kernel to launch</param>
            /// <param name="gridDimX">Width of grid in blocks</param>
            /// <param name="gridDimY">Height of grid in blocks</param>
            /// <param name="gridDimZ">Depth of grid in blocks</param>
            /// <param name="blockDimX">X dimension of each thread block</param>
            /// <param name="blockDimY">Y dimension of each thread block</param>
            /// <param name="blockDimZ">Z dimension of each thread block</param>
            /// <param name="sharedMemBytes">Dynamic shared-memory size per thread block in bytes</param>
            /// <param name="hStream">Stream identifier</param>
            /// <param name="kernelParams">Array of pointers to kernel parameters</param>
            /// <param name="extra">Extra options</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>, 
            /// <see cref="CUResult.ErrorInvalidImage"/>, <see cref="CUResult.ErrorInvalidValue"/>
            /// <see cref="CUResult.ErrorLaunchFailed"/>, <see cref="CUResult.ErrorLaunchOutOfResources"/>
            /// <see cref="CUResult.ErrorLaunchTimeout"/>, <see cref="CUResult.ErrorLaunchIncompatibleTexturing"/>, <see cref="CUResult.ErrorSharedObjectInitFailed"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuLaunchKernel" + CUDA_PTSZ)]
            public static extern CUResult cuLaunchKernel(CUfunction f,
                                uint gridDimX,
                                uint gridDimY,
                                uint gridDimZ,
                                uint blockDimX,
                                uint blockDimY,
                                uint blockDimZ,
                                uint sharedMemBytes,
                                CUstream hStream,
                                IntPtr[] kernelParams,
                                IntPtr[] extra);
        }
        #endregion

        #region Events
        /// <summary>
        /// Groups all event API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class Events
        {
            /// <summary>
            /// Creates an event <c>phEvent</c> with the flags specified via <c>Flags</c>. See <see cref="CUEventFlags"/>
            /// </summary>
            /// <param name="phEvent">Returns newly created event</param>
            /// <param name="Flags">Event creation flags</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuEventCreate(ref CUevent phEvent, CUEventFlags Flags);
            
            /// <summary>
            /// Records an event. If <c>stream</c> is non-zero, the event is recorded after all preceding operations in the stream have been
            /// completed; otherwise, it is recorded after all preceding operations in the CUDA context have been completed. Since
            /// operation is asynchronous, <see cref="cuEventQuery"/> and/or <see cref="cuEventSynchronize"/> must be used to determine when the event
            /// has actually been recorded. <para/>
            /// If <see cref="cuEventRecord"/> has previously been called and the event has not been recorded yet, this function returns
            /// <see cref="CUResult.ErrorInvalidValue"/>.
            /// </summary>
            /// <param name="hEvent">Event to record</param>
            /// <param name="hStream">Stream to record event for</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuEventRecord" + CUDA_PTSZ)]
            public static extern CUResult cuEventRecord( CUevent hEvent, CUstream hStream );
            
            /// <summary>
            /// Returns <see cref="CUResult.Success"/> if the event has actually been recorded, or <see cref="CUResult.ErrorNotReady"/> if not. If
            /// <see cref="cuEventRecord"/> has not been called on this event, the function returns <see cref="CUResult.ErrorInvalidValue"/>.
            /// </summary>
            /// <param name="hEvent">Event to query</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorNotReady"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuEventQuery( CUevent hEvent );
            
            /// <summary>
            /// Waits until the event has actually been recorded. If <see cref="cuEventRecord"/> has been called on this event, the function returns
            /// <see cref="CUResult.ErrorInvalidValue"/>. Waiting for an event that was created with the <see cref="CUEventFlags.BlockingSync"/>
            /// flag will cause the calling CPU thread to block until the event has actually been recorded. <para/>
            /// If <see cref="cuEventRecord"/> has previously been called and the event has not been recorded yet, this function returns <see cref="CUResult.ErrorInvalidValue"/>.
            /// </summary>
            /// <param name="hEvent">Event to wait for</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuEventSynchronize( CUevent hEvent );
           
            /// <summary>
            /// Destroys the event specified by <c>event</c>.
            /// </summary>
            /// <param name="hEvent">Event to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult cuEventDestroy( CUevent hEvent );
           
            /// <summary>
            /// Destroys the event specified by <c>event</c>.<para/>
            /// In the case that <c>hEvent</c> has been recorded but has not yet been completed
            /// when <see cref="cuEventDestroy"/> is called, the function will return immediately and 
            /// the resources associated with <c>hEvent</c> will be released automatically once
            /// the device has completed <c>hEvent</c>.
            /// </summary>
            /// <param name="hEvent">Event to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuEventDestroy_v2( CUevent hEvent );
            
            /// <summary>
            /// Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds). If
            /// either event has not been recorded yet, this function returns <see cref="CUResult.ErrorNotReady"/>. If either event has been
            /// recorded with a non-zero stream, the result is undefined.
            /// </summary>
            /// <param name="pMilliseconds">Returned elapsed time in milliseconds</param>
            /// <param name="hStart">Starting event</param>
            /// <param name="hEnd">Ending event</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorNotReady"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuEventElapsedTime( ref float pMilliseconds, CUevent hStart, CUevent hEnd );
        }
        #endregion

        #region Streams
        /// <summary>
        /// Groups all stream API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class Streams
        {
            /// <summary>
            /// Creates a stream and returns a handle in <c>phStream</c>. The <c>Flags</c> argument
			/// determines behaviors of the stream. Valid values for <c>Flags</c> are:
			/// - <see cref="CUStreamFlags.Default"/>: Default stream creation flag.
			/// - <see cref="CUStreamFlags.NonBlocking"/>: Specifies that work running in the created 
			/// stream may run concurrently with work in stream 0 (the NULL stream), and that
			/// the created stream should perform no implicit synchronization with stream 0.
            /// </summary>
            /// <param name="phStream">Returned newly created stream</param>
            /// <param name="Flags">Parameters for stream creation</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorOutOfMemory"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult  cuStreamCreate( ref CUstream phStream, CUStreamFlags Flags );
            
            /// <summary>
            /// Returns <see cref="CUResult.Success"/> if all operations in the stream specified by <c>hStream</c> have completed, or
            /// <see cref="CUResult.ErrorNotReady"/> if not.
            /// </summary>
            /// <param name="hStream">Stream to query status of</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorNotReady"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuStreamQuery" + CUDA_PTSZ)]
            public static extern CUResult  cuStreamQuery( CUstream hStream );
            
            /// <summary>
            /// Waits until the device has completed all operations in the stream specified by <c>hStream</c>. If the context was created
            /// with the <see cref="CUCtxFlags.BlockingSync"/> flag, the CPU thread will block until the stream is finished with all of its
            /// tasks.
            /// </summary>
            /// <param name="hStream">Stream to wait for</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuStreamSynchronize" + CUDA_PTSZ)]
            public static extern CUResult  cuStreamSynchronize( CUstream hStream );
            
            /// <summary>
            /// Destroys the stream specified by hStream.
            /// </summary>
            /// <param name="hStream">Stream to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            [Obsolete(CUDA_OBSOLET_4_0)]
            public static extern CUResult  cuStreamDestroy( CUstream hStream );           
            
            /// <summary>
            /// Destroys the stream specified by hStream.<para/>
            /// In the case that the device is still doing work in the stream <c>hStream</c>
            /// when <see cref="cuStreamDestroy"/> is called, the function will return immediately 
            /// and the resources associated with <c>hStream</c> will be released automatically 
            /// once the device has completed all work in <c>hStream</c>.
            /// </summary>
            /// <param name="hStream">Stream to destroy</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult  cuStreamDestroy_v2( CUstream hStream );           
            
            /// <summary>
            /// Make a compute stream wait on an event<para/>
            /// Makes all future work submitted to <c>hStream</c>  wait until <c>hEvent</c>
            /// reports completion before beginning execution. This synchronization
            /// will be performed efficiently on the device.
            /// <para/>
            /// The stream <c>hStream</c> will wait only for the completion of the most recent
            /// host call to <see cref="Events.cuEventRecord"/> on <c>hEvent</c>. Once this call has returned,
            /// any functions (including <see cref="Events.cuEventRecord"/> and <see cref="Events.cuEventDestroy"/> may be
            /// called on <c>hEvent</c> again, and the subsequent calls will not have any
            /// effect on <c>hStream</c>.
            /// <para/>
            /// If <c>hStream</c> is 0 (the NULL stream) any future work submitted in any stream
            /// will wait for <c>hEvent</c> to complete before beginning execution. This
            /// effectively creates a barrier for all future work submitted to the context.
            /// <para/>
            /// If <see cref="Events.cuEventRecord"/> has not been called on <c>hEvent</c>, this call acts as if
            /// the record has already completed, and so is a functional no-op.
            /// <para/><c>Flags</c> argument must be 0.
            /// </summary>
            /// <param name="hStream">Stream to destroy</param>
            /// <param name="hEvent">Event</param>
            /// <param name="Flags">Flags argument must be set 0.</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuStreamWaitEvent" + CUDA_PTSZ)]
            public static extern CUResult  cuStreamWaitEvent(CUstream hStream, CUevent hEvent, uint Flags);

			/// <summary>
			/// Adds a callback to be called on the host after all currently enqueued
			/// items in the stream have completed.  For each 
			/// cuStreamAddCallback call, the callback will be executed exactly once.
			/// The callback will block later work in the stream until it is finished.
			/// <para/>
			/// The callback may be passed <see cref="CUResult.Success"/> or an error code.  In the event
			/// of a device error, all subsequently executed callbacks will receive an
			/// appropriate <see cref="CUResult"/>.
			/// <para/>
			/// Callbacks must not make any CUDA API calls.  Attempting to use a CUDA API
			/// will result in <see cref="CUResult.ErrorNotPermitted"/>.  Callbacks must not perform any
			/// synchronization that may depend on outstanding device work or other callbacks
			/// that are not mandated to run earlier.  Callbacks without a mandated order
			/// (in independent streams) execute in undefined order and may be serialized.
			/// <para/>
			/// This API requires compute capability 1.1 or greater.  See
			/// cuDeviceGetAttribute or ::cuDeviceGetProperties to query compute
			/// capability.  Attempting to use this API with earlier compute versions will
			/// return <see cref="CUResult.ErrorNotSupported"/>.
			/// </summary>
			/// <param name="hStream">Stream to add callback to</param>
			/// <param name="callback">The function to call once preceding stream operations are complete</param>
			/// <param name="userData">User specified data to be passed to the callback function</param>
			/// <param name="flags">Reserved for future use; must be 0.</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuStreamAddCallback" + CUDA_PTSZ)]
			public static extern CUResult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, IntPtr userData, CUStreamAddCallbackFlags flags);

			/// <summary>
			/// Create a stream with the given priority<para/>
			/// Creates a stream with the specified priority and returns a handle in <c>phStream</c>. <para/>
			/// This API alters the scheduler priority of work in the stream. Work in a higher priority stream 
			/// may preempt work already executing in a low priority stream.<para/>
			/// <c>priority</c> follows a convention where lower numbers represent higher priorities.<para/>
			/// '0' represents default priority. The range of meaningful numerical priorities can
			/// be queried using <see cref="ContextManagement.cuCtxGetStreamPriorityRange"/>. If the specified priority is
			/// outside the numerical range returned by <see cref="ContextManagement.cuCtxGetStreamPriorityRange"/>,
			/// it will automatically be clamped to the lowest or the highest number in the range.
			/// </summary>
			/// <param name="phStream">Returned newly created stream</param>
			/// <param name="flags">Flags for stream creation. See ::cuStreamCreate for a list of valid flags</param>
			/// <param name="priority">Stream priority. Lower numbers represent higher priorities. <para/>
			/// See <see cref="ContextManagement.cuCtxGetStreamPriorityRange"/> for more information about meaningful stream priorities that can be passed.</param>
			/// <remarks>Stream priorities are supported only on Quadro and Tesla GPUs with compute capability 3.5 or higher.
			/// <para/>In the current implementation, only compute kernels launched in priority streams are affected by the stream's priority. <para/>
			/// Stream priorities have no effect on host-to-device and device-to-host memory operations.</remarks>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuStreamCreateWithPriority(ref CUstream phStream, CUStreamFlags flags, int priority);

			
			/// <summary>
			/// Query the priority of a given stream<para/>
			/// Query the priority of a stream created using <see cref="cuStreamCreate"/> or <see cref="cuStreamCreateWithPriority"/>
			/// and return the priority in <c>priority</c>. Note that if the stream was created with a
			/// priority outside the numerical range returned by <see cref="ContextManagement.cuCtxGetStreamPriorityRange"/>,
			/// this function returns the clamped priority.
			/// See <see cref="cuStreamCreateWithPriority"/> for details about priority clamping.
			/// </summary>
			/// <param name="hStream">Handle to the stream to be queried</param>
			/// <param name="priority">Pointer to a signed integer in which the stream's priority is returned</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuStreamGetPriority" + CUDA_PTSZ)]
			public static extern CUResult cuStreamGetPriority(CUstream hStream, ref int priority);

			/// <summary>
			/// Query the flags of a given stream<para/>
			/// Query the flags of a stream created using <see cref="cuStreamCreate"/> or <see cref="cuStreamCreateWithPriority"/>
			/// and return the flags in <c>flags</c>.
			/// </summary>
			/// <param name="hStream">Handle to the stream to be queried</param>
			/// <param name="flags">Pointer to an unsigned integer in which the stream's flags are returned. <para/>
			/// The value returned in <c>flags</c> is a logical 'OR' of all flags that
			/// were used while creating this stream. See <see cref="cuStreamCreate"/> for the list
			/// of valid flags</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuStreamGetFlags" + CUDA_PTSZ)]
			public static extern CUResult cuStreamGetFlags(CUstream hStream, ref CUStreamFlags flags);

			/// <summary>
			/// Attach memory to a stream asynchronously
			/// <para/>
			/// Enqueues an operation in <c>hStream</c> to specify stream association of
			/// <c>length</c> bytes of memory starting from <c>dptr</c>. This function is a
			/// stream-ordered operation, meaning that it is dependent on, and will
			/// only take effect when, previous work in stream has completed. Any
			/// previous association is automatically replaced.
			/// <para/>
			/// <c>dptr</c> must point to an address within managed memory space declared
			/// using the __managed__ keyword or allocated with cuMemAllocManaged.
			/// <para/>
			/// <c>length</c> must be zero, to indicate that the entire allocation's
			/// stream association is being changed. Currently, it's not possible
			/// to change stream association for a portion of an allocation.
			/// <para/>
			/// The stream association is specified using <c>flags</c> which must be
			/// one of <see cref="CUmemAttach_flags"/>.
			/// If the <see cref="CUmemAttach_flags.Global"/> flag is specified, the memory can be accessed
			/// by any stream on any device.
			/// If the <see cref="CUmemAttach_flags.Host"/> flag is specified, the program makes a guarantee
			/// that it won't access the memory on the device from any stream.
			/// If the <see cref="CUmemAttach_flags.Single"/> flag is specified, the program makes a guarantee
			/// that it will only access the memory on the device from <c>hStream</c>. It is illegal
			/// to attach singly to the NULL stream, because the NULL stream is a virtual global
			/// stream and not a specific stream. An error will be returned in this case.
			/// <para/>
			/// When memory is associated with a single stream, the Unified Memory system will
			/// allow CPU access to this memory region so long as all operations in <c>hStream</c>
			/// have completed, regardless of whether other streams are active. In effect,
			/// this constrains exclusive ownership of the managed memory region by
			/// an active GPU to per-stream activity instead of whole-GPU activity.
			/// <para/>
			/// Accessing memory on the device from streams that are not associated with
			/// it will produce undefined results. No error checking is performed by the
			/// Unified Memory system to ensure that kernels launched into other streams
			/// do not access this region. 
			/// <para/>
			/// It is a program's responsibility to order calls to <see cref="cuStreamAttachMemAsync"/>
			/// via events, synchronization or other means to ensure legal access to memory
			/// at all times. Data visibility and coherency will be changed appropriately
			/// for all kernels which follow a stream-association change.
			/// <para/>
			/// If <c>hStream</c> is destroyed while data is associated with it, the association is
			/// removed and the association reverts to the default visibility of the allocation
			/// as specified at cuMemAllocManaged. For __managed__ variables, the default
			/// association is always <see cref="CUmemAttach_flags.Global"/>. Note that destroying a stream is an
			/// asynchronous operation, and as a result, the change to default association won't
			/// happen until all work in the stream has completed.
			/// <para/>
			/// </summary>
			/// <param name="hStream">Stream in which to enqueue the attach operation</param>
			/// <param name="dptr">Pointer to memory (must be a pointer to managed memory)</param>
			/// <param name="length">Length of memory (must be zero)</param>
			/// <param name="flags">Must be one of <see cref="CUmemAttach_flags"/></param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuStreamAttachMemAsync" + CUDA_PTSZ)]
			public static extern CUResult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, SizeT length, CUmemAttach_flags flags);


        }
        #endregion

        #region Graphics interop
        /// <summary>
        /// Combines all graphics interop API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class GraphicsInterop
        {
            /// <summary>
            /// Unregisters the graphics resource <c>resource</c> so it is not accessible by CUDA unless registered again.
            /// If resource is invalid then <see cref="CUResult.ErrorInvalidHandle"/> is returned.
            /// </summary>
            /// <param name="resource">Resource to unregister</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsUnregisterResource(CUgraphicsResource resource);
            
            /// <summary>
            /// Returns in <c>pArray</c> an array through which the subresource of the mapped graphics resource resource which
            /// corresponds to array index <c>arrayIndex</c> and mipmap level <c>mipLevel</c> may be accessed. The value set in <c>pArray</c>
            /// may change every time that <c>resource</c> is mapped.<para/>
            /// If <c>resource</c> is not a texture then it cannot be accessed via an array and <see cref="CUResult.ErrorNotMappedAsArray"/>
            /// is returned. If <c>arrayIndex</c> is not a valid array index for <c>resource</c> then <see cref="CUResult.ErrorInvalidValue"/>
            /// is returned. If <c>mipLevel</c> is not a valid mipmap level for <c>resource</c> then <see cref="CUResult.ErrorInvalidValue"/>
            /// is returned. If <c>resource</c> is not mapped then <see cref="CUResult.ErrorNotMapped"/> is returned.
            /// </summary>
            /// <param name="pArray">Returned array through which a subresource of <c>resource</c> may be accessed</param>
            /// <param name="resource">Mapped resource to access</param>
            /// <param name="arrayIndex">Array index for array textures or cubemap face index as defined by <see cref="CUArrayCubemapFace"/> for
            /// cubemap textures for the subresource to access</param>
            /// <param name="mipLevel">Mipmap level for the subresource to access</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
            /// <see cref="CUResult.ErrorNotMapped"/>, <see cref="CUResult.ErrorNotMappedAsArray"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsSubResourceGetMappedArray( ref CUarray pArray, CUgraphicsResource resource, uint arrayIndex, uint mipLevel );

			/// <summary>
			/// Returns in <c>pMipmappedArray</c> a mipmapped array through which the mapped graphics 
			/// resource <c>resource</c>. The value set in <c>pMipmappedArray</c> may change every time 
			/// that <c>resource</c> is mapped.
			/// <para/>
			/// If <c>resource</c> is not a texture then it cannot be accessed via a mipmapped array and
			/// <see cref="CUResult.ErrorNotMappedAsArray"/> is returned.
			/// If <c>resource</c> is not mapped then <see cref="CUResult.ErrorNotMapped"/> is returned.
			/// </summary>
			/// <param name="pMipmappedArray">Returned mipmapped array through which <c>resource</c> may be accessed</param>
			/// <param name="resource">Mapped resource to access</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
			/// <see cref="CUResult.ErrorNotMapped"/>, <see cref="CUResult.ErrorNotMappedAsArray"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuGraphicsResourceGetMappedMipmappedArray(ref CUmipmappedArray pMipmappedArray, CUgraphicsResource resource);


            /// <summary>
            /// Returns in <c>pDevPtr</c> a pointer through which the mapped graphics resource <c>resource</c> may be accessed. Returns
            /// in <c>pSize</c> the size of the memory in bytes which may be accessed from that pointer. The value set in <c>pPointer</c> may
            /// change every time that <c>resource</c> is mapped.<para/>
            /// If <c>resource</c> is not a buffer then it cannot be accessed via a pointer and <see cref="CUResult.ErrorNotMappedAsPointer"/>
            /// is returned. If resource is not mapped then <see cref="CUResult.ErrorNotMapped"/> is returned.
            /// </summary>
            /// <param name="pDevPtr">Returned pointer through which <c>resource</c> may be accessed</param>
            /// <param name="pSize">Returned size of the buffer accessible starting at <c>pPointer</c></param>
            /// <param name="resource">Mapped resource to access</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
            /// <see cref="CUResult.ErrorNotMapped"/>, <see cref="CUResult.ErrorNotMappedAsPointer"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGraphicsResourceGetMappedPointer_v2(ref CUdeviceptr pDevPtr, ref SizeT pSize, CUgraphicsResource resource);
            
            /// <summary>
            /// Set <c>flags</c> for mapping the graphics resource <c>resource</c>.
            /// Changes to <c>flags</c> will take effect the next time <c>resource</c> is mapped. See <see cref="CUGraphicsMapResourceFlags"/>. <para/>
            /// If <c>resource</c> is presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> is returned. If
            /// <c>flags</c> is not one of the <see cref="CUGraphicsMapResourceFlags"/> values then <see cref="CUResult.ErrorInvalidValue"/> is returned.
            /// </summary>
            /// <param name="resource">Registered resource to set flags for</param>
            /// <param name="flags">Parameters for resource mapping</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
            /// <see cref="CUResult.ErrorAlreadyMapped"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuGraphicsResourceSetMapFlags_v2")]
            public static extern CUResult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, CUGraphicsMapResourceFlags flags);

            /// <summary>
            /// Maps the <c>count</c> graphics resources in <c>resources</c> for access by CUDA.<para/>
            /// The resources in <c>resources</c> may be accessed by CUDA until they are unmapped. The graphics API from which
            /// <c>resources</c> were registered should not access any resources while they are mapped by CUDA. If an application does
            /// so, the results are undefined.<para/>
            /// This function provides the synchronization guarantee that any graphics calls issued before <see cref="cuGraphicsMapResources(uint, ref CUgraphicsResource, CUstream)"/>
            /// will complete before any subsequent CUDA work issued in <c>stream</c> begins.<para/>
            /// If <c>resources</c> includes any duplicate entries then <see cref="CUResult.ErrorInvalidHandle"/> is returned. If any of
            /// <c>resources</c> are presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> is returned.
            /// </summary>
            /// <param name="count">Number of resources to map. Here: must be 1</param>
            /// <param name="resources">Resources to map for CUDA usage</param>
            /// <param name="hStream">Stream with which to synchronize</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
            /// <see cref="CUResult.ErrorAlreadyMapped"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuGraphicsMapResources" + CUDA_PTSZ)]
            public static extern CUResult cuGraphicsMapResources(uint count, ref CUgraphicsResource resources, CUstream hStream);

            /// <summary>
            /// Maps the <c>count</c> graphics resources in <c>resources</c> for access by CUDA.<para/>
            /// The resources in <c>resources</c> may be accessed by CUDA until they are unmapped. The graphics API from which
            /// <c>resources</c> were registered should not access any resources while they are mapped by CUDA. If an application does
            /// so, the results are undefined.<para/>
            /// This function provides the synchronization guarantee that any graphics calls issued before <see cref="cuGraphicsMapResources(uint, CUgraphicsResource[], CUstream)"/>
            /// will complete before any subsequent CUDA work issued in <c>stream</c> begins.<para/>
            /// If <c>resources</c> includes any duplicate entries then <see cref="CUResult.ErrorInvalidHandle"/> is returned. If any of
            /// <c>resources</c> are presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> is returned.
            /// </summary>
            /// <param name="count">Number of resources to map</param>
            /// <param name="resources">Resources to map for CUDA usage</param>
            /// <param name="hStream">Stream with which to synchronize</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
            /// <see cref="CUResult.ErrorAlreadyMapped"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuGraphicsMapResources" + CUDA_PTSZ)]
            public static extern CUResult cuGraphicsMapResources(uint count, CUgraphicsResource[] resources, CUstream hStream);

            /// <summary>
            /// Unmaps the <c>count</c> graphics resources in resources.<para/>
            /// Once unmapped, the resources in <c>resources</c> may not be accessed by CUDA until they are mapped again.<para/>
            /// This function provides the synchronization guarantee that any CUDA work issued in <c>stream</c> before <see cref="cuGraphicsUnmapResources(uint, ref CUgraphicsResource, CUstream)"/>
            /// will complete before any subsequently issued graphics work begins.<para/>
            /// If <c>resources</c> includes any duplicate entries then <see cref="CUResult.ErrorInvalidHandle"/> is returned. If any of
            /// resources are not presently mapped for access by CUDA then <see cref="CUResult.ErrorNotMapped"/> is returned.
            /// </summary>
            /// <param name="count">Number of resources to unmap. Here: must be 1</param>
            /// <param name="resources">Resources to unmap</param>
            /// <param name="hStream">Stream with which to synchronize</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
            /// <see cref="CUResult.ErrorNotMapped"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuGraphicsUnmapResources" + CUDA_PTSZ)]
            public static extern CUResult cuGraphicsUnmapResources(uint count, ref CUgraphicsResource resources, CUstream hStream);

            /// <summary>
            /// Unmaps the <c>count</c> graphics resources in resources.<para/>
            /// Once unmapped, the resources in <c>resources</c> may not be accessed by CUDA until they are mapped again.<para/>
            /// This function provides the synchronization guarantee that any CUDA work issued in <c>stream</c> before <see cref="cuGraphicsUnmapResources(uint, CUgraphicsResource[], CUstream)"/>
            /// will complete before any subsequently issued graphics work begins.<para/>
            /// If <c>resources</c> includes any duplicate entries then <see cref="CUResult.ErrorInvalidHandle"/> is returned. If any of
            /// resources are not presently mapped for access by CUDA then <see cref="CUResult.ErrorNotMapped"/> is returned.
            /// </summary>
            /// <param name="count">Number of resources to unmap</param>
            /// <param name="resources">Resources to unmap</param>
            /// <param name="hStream">Stream with which to synchronize</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidHandle"/>,
            /// <see cref="CUResult.ErrorNotMapped"/>, <see cref="CUResult.ErrorUnknown"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME, EntryPoint = "cuGraphicsUnmapResources" + CUDA_PTSZ)]
            public static extern CUResult cuGraphicsUnmapResources(uint count, CUgraphicsResource[] resources, CUstream hStream);           
        }
        #endregion   

        #region Export tables
        /// <summary>
        /// cuGetExportTable
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class ExportTables
        {
            /// <summary>
            /// No description found in the CUDA reference manual...
            /// </summary>
            /// <param name="ppExportTable"></param>
            /// <param name="pExportTableId"></param>
            /// <returns>CUDA Error Code<remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuGetExportTable(ref IntPtr ppExportTable, ref CUuuid pExportTableId );
        }
        #endregion

        #region Limits
        /// <summary>
        /// Groups all context limit API calls
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class Limits
        {
            /// <summary>
            /// Setting <c>limit</c> to <c>value</c> is a request by the application to update the current limit maintained by the context. The
            /// driver is free to modify the requested value to meet h/w requirements (this could be clamping to minimum or maximum
            /// values, rounding up to nearest element size, etc). The application can use <see cref="cuCtxGetLimit"/> to find out exactly what
            /// the limit has been set to.<para/>
            /// Setting each <see cref="CULimit"/> has its own specific restrictions, so each is discussed here:
            /// <list type="table">  
            /// <listheader><term>Value</term><description>Restriction</description></listheader>  
            /// <item><term><see cref="CULimit.StackSize"/></term><description>
            /// <see cref="CULimit.StackSize"/> controls the stack size of each GPU thread. This limit is only applicable to devices
            /// of compute capability 2.0 and higher. Attempting to set this limit on devices of compute capability less than 2.0
            /// will result in the error <see cref="CUResult.ErrorUnsupportedLimit"/> being returned.
            /// </description></item>  
            /// <item><term><see cref="CULimit.PrintfFIFOSize"/></term><description>
            /// <see cref="CULimit.PrintfFIFOSize"/> controls the size of the FIFO used by the <c>printf()</c> device system call. Setting
            /// <see cref="CULimit.PrintfFIFOSize"/> must be performed before loading any module that uses the printf() device
            /// system call, otherwise <see cref="CUResult.ErrorInvalidValue"/> will be returned. This limit is only applicable to
            /// devices of compute capability 2.0 and higher. Attempting to set this limit on devices of compute capability less
            /// than 2.0 will result in the error <see cref="CUResult.ErrorUnsupportedLimit"/> being returned.
            /// </description></item> 
			/// <item><term><see cref="CULimit.MallocHeapSize"/></term><description>
            /// <see cref="CULimit.MallocHeapSize"/> controls the size in bytes of the heap used by the ::malloc() and ::free() device system calls. Setting
            /// <see cref="CULimit.MallocHeapSize"/> must be performed before launching any kernel that uses the ::malloc() or ::free() device system calls, otherwise
            /// <see cref="CUResult.ErrorInvalidValue"/> will be returned. This limit is only applicable to
            /// devices of compute capability 2.0 and higher. Attempting to set this limit on devices of compute capability less
            /// than 2.0 will result in the error <see cref="CUResult.ErrorUnsupportedLimit"/> being returned.
            /// </description></item> 
			/// <item><term><see cref="CULimit.DevRuntimeSyncDepth"/></term><description>
            /// <see cref="CULimit.DevRuntimeSyncDepth"/> controls the maximum nesting depth of a grid at which a thread can safely call ::cudaDeviceSynchronize(). Setting
            /// this limit must be performed before any launch of a kernel that uses the
			/// device runtime and calls ::cudaDeviceSynchronize() above the default sync
			/// depth, two levels of grids. Calls to ::cudaDeviceSynchronize() will fail 
			/// with error code ::cudaErrorSyncDepthExceeded if the limitation is 
			/// violated. This limit can be set smaller than the default or up the maximum
			/// launch depth of 24. When setting this limit, keep in mind that additional
			/// levels of sync depth require the driver to reserve large amounts of device
			/// memory which can no longer be used for user allocations. If these 
			/// reservations of device memory fail, ::cuCtxSetLimit will return 
			/// <see cref="CUResult.ErrorOutOfMemory"/>, and the limit can be reset to a lower value.
			/// This limit is only applicable to devices of compute capability 3.5 and
			/// higher. Attempting to set this limit on devices of compute capability less
			/// than 3.5 will result in the error <see cref="CUResult.ErrorUnsupportedLimit"/> being 
			/// returned.
            /// </description></item> 
			/// <item><term><see cref="CULimit.DevRuntimePendingLaunchCount"/></term><description>
            /// <see cref="CULimit.DevRuntimePendingLaunchCount"/> controls the maximum number of 
			/// outstanding device runtime launches that can be made from the current
			/// context. A grid is outstanding from the point of launch up until the grid
			/// is known to have been completed. Device runtime launches which violate 
			/// this limitation fail and return ::cudaErrorLaunchPendingCountExceeded when
			/// ::cudaGetLastError() is called after launch. If more pending launches than
			/// the default (2048 launches) are needed for a module using the device
			/// runtime, this limit can be increased. Keep in mind that being able to
			/// sustain additional pending launches will require the driver to reserve
			/// larger amounts of device memory upfront which can no longer be used for
			/// allocations. If these reservations fail, ::cuCtxSetLimit will return
			/// <see cref="CUResult.ErrorOutOfMemory"/>, and the limit can be reset to a lower value.
			/// This limit is only applicable to devices of compute capability 3.5 and
			/// higher. Attempting to set this limit on devices of compute capability less
			/// than 3.5 will result in the error <see cref="CUResult.ErrorUnsupportedLimit"/> being
			/// returned. 
            /// </description></item> 
            /// </list>   
            /// </summary>
            /// <param name="limit">Limit to set</param>
            /// <param name="value">Size in bytes of limit</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnsupportedLimit"/>, .
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxSetLimit(CULimit limit, SizeT value);

            /// <summary>
            /// Returns in <c>pvalue</c> the current size of limit. See <see cref="CULimit"/>
            /// </summary>
            /// <param name="pvalue">Returned size in bytes of limit</param>
            /// <param name="limit">Limit to query</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnsupportedLimit"/>, .
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxGetLimit(ref SizeT pvalue, CULimit limit);
        }
        #endregion

        #region CudaPeerAccess
        /// <summary>
        /// Peer Context Memory Access
        /// </summary>
        [System.Security.SuppressUnmanagedCodeSecurityAttribute]
        public static class CudaPeerAccess
        {
            /// <summary>
            /// Returns in <c>canAccessPeer</c> a value of 1 if contexts on <c>dev</c> are capable of
            /// directly accessing memory from contexts on <c>peerDev</c> and 0 otherwise.
            /// If direct access of <c>peerDev</c> from <c>dev</c> is possible, then access may be
            /// enabled on two specific contexts by calling <see cref="cuCtxEnablePeerAccess"/>.
            /// </summary>
            /// <param name="canAccessPeer">Returned access capability</param>
            /// <param name="dev">Device from which allocations on peerDev are to be directly accessed.</param>
            /// <param name="peerDev">Device on which the allocations to be directly accessed by dev reside.</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidDevice"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuDeviceCanAccessPeer(ref int canAccessPeer, CUdevice dev, CUdevice peerDev);

            /// <summary>
            /// If both the current context and <c>peerContext</c> are on devices which support unified 
            /// addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING), then
            /// on success all allocations from <c>peerContext</c> will immediately be accessible
            /// by the current context.  See \ref CUDA_UNIFIED for additional
            /// details. <para/>
            /// Note that access granted by this call is unidirectional and that in order to access
            /// memory from the current context in <c>peerContext</c>, a separate symmetric call 
            /// to ::cuCtxEnablePeerAccess() is required. <para/>
            /// Returns <see cref="CUResult.ErrorInvalidDevice"/> if <see cref="cuDeviceCanAccessPeer"/> indicates
            /// that the CUdevice of the current context cannot directly access memory
            /// from the CUdevice of <c>peerContext</c>. <para/>
            /// Returns <see cref="CUResult.ErrorPeerAccessAlreadyEnabled"/> if direct access of
            /// <c>peerContext</c> from the current context has already been enabled. <para/>
            /// Returns <see cref="CUResult.ErrorInvalidContext"/> if there is no current context, <c>peerContext</c>
            /// is not a valid context, or if the current context is <c>peerContext</c>. <para/>
            /// Returns <see cref="CUResult.ErrorInvalidValue"/> if <c>Flags</c> is not 0.
            /// </summary>
            /// <param name="peerContext">Peer context to enable direct access to from the current context</param>
            /// <param name="Flags">Reserved for future use and must be set to 0</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorInvalidDevice"/>, <see cref="CUResult.ErrorPeerAccessAlreadyEnabled"/>, <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxEnablePeerAccess(CUcontext peerContext, CtxEnablePeerAccessFlags Flags);

            /// <summary>
            /// Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
            /// </summary>
            /// <param name="peerContext">Peer context to disable direct access to</param>
            /// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
            /// <see cref="CUResult.ErrorPeerAccessNotEnabled"/>, <see cref="CUResult.ErrorInvalidContext"/>.
            /// <remarks>Note that this function may also return error codes from previous, asynchronous launches.</remarks></returns>
            [DllImport(CUDA_DRIVER_API_DLL_NAME)]
            public static extern CUResult cuCtxDisablePeerAccess(CUcontext peerContext);
        }
        #endregion

		#region Texture objects
		/// <summary>
		/// Texture object management functions.
		/// </summary>
		public static class TextureObjects
		{ 
			/// <summary>
			/// Creates a texture object and returns it in <c>pTexObject</c>. <c>pResDesc</c> describes
			/// the data to texture from. <c>pTexDesc</c> describes how the data should be sampled.
			/// <c>pResViewDesc</c> is an optional argument that specifies an alternate format for
			/// the data described by <c>pResDesc</c>, and also describes the subresource region
			/// to restrict access to when texturing. <c>pResViewDesc</c> can only be specified if
			/// the type of resource is a CUDA array or a CUDA mipmapped array.
			/// </summary>
			/// <param name="pTexObject">Texture object to create</param>
			/// <param name="pResDesc">Resource descriptor</param>
			/// <param name="pTexDesc">Texture descriptor</param>
			/// <param name="pResViewDesc">Resource view descriptor</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexObjectCreate(ref CUtexObject pTexObject, ref CudaResourceDesc pResDesc, ref CudaTextureDescriptor pTexDesc, ref CudaResourceViewDesc pResViewDesc);
			
			/// <summary>
			/// Creates a texture object and returns it in <c>pTexObject</c>. <c>pResDesc</c> describes
			/// the data to texture from. <c>pTexDesc</c> describes how the data should be sampled.
			/// <c>pResViewDesc</c> is an optional argument that specifies an alternate format for
			/// the data described by <c>pResDesc</c>, and also describes the subresource region
			/// to restrict access to when texturing. <c>pResViewDesc</c> can only be specified if
			/// the type of resource is a CUDA array or a CUDA mipmapped array.
			/// </summary>
			/// <param name="pTexObject">Texture object to create</param>
			/// <param name="pResDesc">Resource descriptor</param>
			/// <param name="pTexDesc">Texture descriptor</param>
			/// <param name="pResViewDesc">Resource view descriptor (Null-Pointer)</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexObjectCreate(ref CUtexObject pTexObject, ref CudaResourceDesc pResDesc, ref CudaTextureDescriptor pTexDesc, IntPtr pResViewDesc);
			
			/// <summary>
			/// Destroys the texture object specified by <c>texObject</c>.
			/// </summary>
			/// <param name="texObject">Texture object to destroy</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexObjectDestroy(CUtexObject texObject);
			
			/// <summary>
			/// Returns the resource descriptor for the texture object specified by <c>texObject</c>.
			/// </summary>
			/// <param name="pResDesc">Resource descriptor</param>
			/// <param name="texObject">Texture object</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexObjectGetResourceDesc(ref CudaResourceDesc pResDesc, CUtexObject texObject);
			
			/// <summary>
			/// Returns the texture descriptor for the texture object specified by <c>texObject</c>.
			/// </summary>
			/// <param name="pTexDesc">Texture descriptor</param>
			/// <param name="texObject">Texture object</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexObjectGetTextureDesc(ref CudaTextureDescriptor pTexDesc, CUtexObject texObject);
			
			/// <summary>
			/// Returns the resource view descriptor for the texture object specified by <c>texObject</c>.
			/// If no resource view was set for <c>texObject</c>, the ::CUDA_ERROR_INVALID_VALUE is returned.
			/// </summary>
			/// <param name="pResViewDesc">Resource view descriptor</param>
			/// <param name="texObject">Texture object</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuTexObjectGetResourceViewDesc(ref CudaResourceViewDesc pResViewDesc, CUtexObject texObject);

		}
		#endregion

		#region Surface objects
		/// <summary>
		/// Surface object management functions.
		/// </summary>
		public static class SurfaceObjects
		{
			/// <summary>
			/// Creates a surface object and returns it in <c>pSurfObject</c>. <c>pResDesc</c> describes
			/// the data to perform surface load/stores on. ::CUDA_RESOURCE_DESC::resType must be 
			/// ::CU_RESOURCE_TYPE_ARRAY and  ::CUDA_RESOURCE_DESC::res::array::hArray
			/// must be set to a valid CUDA array handle. ::CUDA_RESOURCE_DESC::flags must be set to zero.
			/// </summary>
			/// <param name="pSurfObject">Surface object to create</param>
			/// <param name="pResDesc">Resource descriptor</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuSurfObjectCreate(ref CUsurfObject pSurfObject, ref CudaResourceDesc pResDesc);
			
			/// <summary>
			/// Destroys the surface object specified by <c>surfObject</c>.
			/// </summary>
			/// <param name="surfObject">Surface object to destroy</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuSurfObjectDestroy(CUsurfObject surfObject);

			/// <summary>
			/// Returns the resource descriptor for the surface object specified by <c>surfObject</c>.
			/// </summary>
			/// <param name="pResDesc">Resource descriptor</param>
			/// <param name="surfObject">Surface object</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuSurfObjectGetResourceDesc(ref CudaResourceDesc pResDesc, CUsurfObject surfObject);

		}
		#endregion

		#region Profiling

		/// <summary>
		/// This section describes the profiler control functions of the low-level CUDA
		/// driver application programming interface.
		/// </summary>
		public static class Profiling
		{
			/// <summary>
			/// Initialize the profiling.<para/>
			/// Using this API user can initialize the CUDA profiler by specifying
			/// the configuration file, output file and output file format. This
			/// API is generally used to profile different set of counters by
			/// looping the kernel launch. The <c>configFile</c> parameter can be used
			/// to select profiling options including profiler counters. Refer to
			/// the "Compute Command Line Profiler User Guide" for supported
			/// profiler options and counters.<para/>
			/// Limitation: The CUDA profiler cannot be initialized with this API
			/// if another profiling tool is already active, as indicated by the
			/// <see cref="CUResult.ErrorProfilerDisabled"/> return code.
			/// </summary>
			/// <param name="configFile">Name of the config file that lists the counters/options for profiling.</param>
			/// <param name="outputFile">Name of the outputFile where the profiling results will be stored.</param>
			/// <param name="outputMode">outputMode</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorProfilerDisabled"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuProfilerInitialize(string configFile, string outputFile, CUoutputMode outputMode);

			/// <summary>
			/// Enable profiling.<para/>
			/// Enables profile collection by the active profiling tool for the
			/// current context. If profiling is already enabled, then
			/// cuProfilerStart() has no effect.<para/>
			/// cuProfilerStart and cuProfilerStop APIs are used to
			/// programmatically control the profiling granularity by allowing
			/// profiling to be done only on selective pieces of code.
			/// </summary>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidContext"/>. 
			/// </returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuProfilerStart();

			/// <summary>
			/// Disables profile collection by the active profiling tool for the
			/// current context. If profiling is already disabled, then
			/// cuProfilerStop() has no effect.<para/>
			/// cuProfilerStart and cuProfilerStop APIs are used to
			/// programmatically control the profiling granularity by allowing
			/// profiling to be done only on selective pieces of code.
			/// </summary>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidContext"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuProfilerStop();
		}
		#endregion

		#region Error Handling

		/// <summary>
		/// This section describes the error handling functions of the low-level CUDA
		/// driver application programming interface.
		/// </summary>
		public static class ErrorHandling
		{
			/// <summary>
			/// Gets the string description of an error code.<para/>
			/// Sets <c>pStr</c> to the address of a NULL-terminated string description
			/// of the error code <c>error</c>.
			/// If the error code is not recognized, <see cref="CUResult.ErrorInvalidValue"/>
			/// will be returned and <c>pStr</c> will be set to the NULL address
			/// </summary>
			/// <param name="error">Error code to convert to string.</param>
			/// <param name="pStr">Address of the string pointer.</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuGetErrorString(CUResult error, ref IntPtr pStr);


			/// <summary>
			/// Gets the string representation of an error code enum name.<para/>
			/// Sets <c>pStr</c> to the address of a NULL-terminated string description
			/// of the name of the enum error code <c>error</c>.
			/// If the error code is not recognized, <see cref="CUResult.ErrorInvalidValue"/>
			/// will be returned and <c>pStr</c> will be set to the NULL address
			/// </summary>
			/// <param name="error">Error code to convert to string.</param>
			/// <param name="pStr">Address of the string pointer.</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorInvalidValue"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuGetErrorName(CUResult error, ref IntPtr pStr);

		}
		#endregion

		#region Occupancy

		/// <summary>
		/// This section describes the occupancy calculation functions of the low-level CUDA
		/// driver application programming interface.
		/// </summary>
		public static class Occupancy
		{
			/// <summary>
			/// Returns in numBlocks the number of the maximum active blocks per
			/// streaming multiprocessor.
			/// </summary>
			/// <param name="numBlocks">Returned occupancy</param>
			/// <param name="func">Kernel for which occupancy is calulated</param>
			/// <param name="blockSize">Block size the kernel is intended to be launched with</param>
			/// <param name="dynamicSMemSize">Per-block dynamic shared memory usage intended, in bytes</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuOccupancyMaxActiveBlocksPerMultiprocessor(ref int numBlocks, CUfunction func, int blockSize, SizeT dynamicSMemSize);

			/// <summary>
			/// Returns in blockSize a reasonable block size that can achieve
			/// the maximum occupancy (or, the maximum number of active warps with
			/// the fewest blocks per multiprocessor), and in minGridSize the
			/// minimum grid size to achieve the maximum occupancy.
			/// 
			/// If blockSizeLimit is 0, the configurator will use the maximum
			/// block size permitted by the device / function instead.
			/// 
			/// If per-block dynamic shared memory allocation is not needed, the
			/// user should leave both blockSizeToDynamicSMemSize and 
			/// dynamicSMemSize as 0.
			/// 
			/// If per-block dynamic shared memory allocation is needed, then if
			/// the dynamic shared memory size is constant regardless of block
			/// size, the size should be passed through dynamicSMemSize, and 
			/// blockSizeToDynamicSMemSize should be NULL.
			/// 
			/// Otherwise, if the per-block dynamic shared memory size varies with
			/// different block sizes, the user needs to provide a unary function
			/// through blockSizeToDynamicSMemSize that computes the dynamic
			/// shared memory needed by func for any given block size.
			/// dynamicSMemSize is ignored.
			/// </summary>
			/// <param name="minGridSize">Returned minimum grid size needed to achieve the maximum occupancy</param>
			/// <param name="blockSize">Returned maximum block size that can achieve the maximum occupancy</param>
			/// <param name="func">Kernel for which launch configuration is calulated</param>
			/// <param name="blockSizeToDynamicSMemSize">A function that calculates how much per-block dynamic shared memory \p func uses based on the block size</param>
			/// <param name="dynamicSMemSize">Dynamic shared memory usage intended, in bytes</param>
			/// <param name="blockSizeLimit">The maximum block size \p func is designed to handle</param>
			/// <returns>CUDA Error Codes: <see cref="CUResult.Success"/>, <see cref="CUResult.ErrorDeinitialized"/>, <see cref="CUResult.ErrorNotInitialized"/>, 
			/// <see cref="CUResult.ErrorInvalidContext"/>, <see cref="CUResult.ErrorInvalidValue"/>, <see cref="CUResult.ErrorUnknown"/>.</returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuOccupancyMaxPotentialBlockSize(ref int minGridSize, ref int blockSize, CUfunction func, del_CUoccupancyB2DSize blockSizeToDynamicSMemSize, SizeT dynamicSMemSize, int blockSizeLimit);
			
			
			/// <summary>
			/// Returns occupancy of a function<para/>
			/// Returns in \p *numBlocks the number of the maximum active blocks per
			/// streaming multiprocessor.
			/// 
			/// The \p Flags parameter controls how special cases are handled. The
			/// valid flags are:
			/// 
			/// - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
			/// ::cuOccupancyMaxActiveBlocksPerMultiprocessor;
			/// - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
			/// default behavior on platform where global caching affects
			/// occupancy. On such platforms, if caching is enabled, but
			/// per-block SM resource usage would result in zero occupancy, the
			/// occupancy calculator will calculate the occupancy as if caching
			/// is disabled. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE makes
			/// the occupancy calculator to return 0 in such cases. More information
			/// can be found about this feature in the "Unified L1/Texture Cache"
			/// section of the Maxwell tuning guide.
			/// </summary>
			/// <param name="numBlocks">Returned occupancy</param>
			/// <param name="func">Kernel for which occupancy is calculated</param>
			/// <param name="blockSize">Block size the kernel is intended to be launched with</param>
			/// <param name="dynamicSMemSize">Per-block dynamic shared memory usage intended, in bytes</param>
			/// <param name="flags">Requested behavior for the occupancy calculator</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref int numBlocks, CUfunction func, int blockSize, SizeT dynamicSMemSize, CUoccupancy_flags flags);

			/// <summary>
			/// Suggest a launch configuration with reasonable occupancy<para/>
			/// An extended version of ::cuOccupancyMaxPotentialBlockSize. In
			/// addition to arguments passed to ::cuOccupancyMaxPotentialBlockSize,
			/// ::cuOccupancyMaxPotentialBlockSizeWithFlags also takes a \p Flags
			/// parameter.
			/// 
			/// The \p Flags parameter controls how special cases are handled. The
			/// valid flags are:
			/// - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
			///   ::cuOccupancyMaxPotentialBlockSize;
			/// - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
			///   default behavior on platform where global caching affects
			///   occupancy. On such platforms, the launch configurations that
			///   produces maximal occupancy might not support global
			///   caching. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
			///   guarantees that the the produced launch configuration is global
			///   caching compatible at a potential cost of occupancy. More information
			///   can be found about this feature in the "Unified L1/Texture Cache"
			///   section of the Maxwell tuning guide.
			/// </summary>
			/// <param name="minGridSize">Returned minimum grid size needed to achieve the maximum occupancy</param>
			/// <param name="blockSize">Returned maximum block size that can achieve the maximum occupancy</param>
			/// <param name="func">Kernel for which launch configuration is calculated</param>
			/// <param name="blockSizeToDynamicSMemSize">A function that calculates how much per-block dynamic shared memory \p func uses based on the block size</param>
			/// <param name="dynamicSMemSize">Dynamic shared memory usage intended, in bytes</param>
			/// <param name="blockSizeLimit">The maximum block size \p func is designed to handle</param>
			/// <param name="flags">Options</param>
			/// <returns></returns>
			[DllImport(CUDA_DRIVER_API_DLL_NAME)]
			public static extern CUResult cuOccupancyMaxPotentialBlockSizeWithFlags(ref int minGridSize, ref int blockSize, CUfunction func, del_CUoccupancyB2DSize blockSizeToDynamicSMemSize, SizeT dynamicSMemSize, int blockSizeLimit, CUoccupancy_flags flags);


		}
		#endregion

	}
}
