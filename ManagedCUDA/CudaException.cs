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


using System;
using ManagedCuda.BasicTypes;
using System.Runtime.Serialization;
using System.Runtime.InteropServices;

namespace ManagedCuda
{
    /// <summary>
    /// A CUDA exception is thrown if a CUDA Driver API method call does not return <see cref="CUResult.Success"/>
    /// </summary>
    [Serializable]
    public class CudaException : Exception, ISerializable
    {
        private CUResult _cudaError;
        private string _internalName;
        private string _internalDescripton;

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public CudaException()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="serInfo"></param>
        /// <param name="streamingContext"></param>
        protected CudaException(SerializationInfo serInfo, StreamingContext streamingContext)
            : base(serInfo, streamingContext)
        {
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        public CudaException(CUResult error)
            : base(GetErrorMessageFromCUResult(error))
        {
            this._cudaError = error;
            this._internalDescripton = GetInternalDescriptionFromCUResult(error);
            this._internalName = GetInternalNameFromCUResult(error);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        public CudaException(string message)
            : base(message)
        {

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaException(string message, Exception exception)
            : base(message, exception)
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaException(CUResult error, string message, Exception exception)
            : base(message, exception)
        {
            this._cudaError = error;
            this._internalDescripton = GetInternalDescriptionFromCUResult(error);
            this._internalName = GetInternalNameFromCUResult(error);
        }
        #endregion

        #region Methods
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this.CudaError.ToString();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="info"></param>
        /// <param name="context"></param>
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("CudaError", this._cudaError);
        }
        #endregion

        #region Static methods
        private static string GetErrorMessageFromCUResult(CUResult error)
        {
            string message = string.Empty;

            switch (error)
            {
                case CUResult.Success:
                    message = "No error.";
                    break;
                case CUResult.ErrorInvalidValue:
                    message = "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.";
                    break;
                case CUResult.ErrorOutOfMemory:
                    message = "The API call failed because it was unable to allocate enough memory to perform the requested operation.";
                    break;
                case CUResult.ErrorNotInitialized:
                    message = "The CUDA driver API is not yet initialized. Call cuInit(Flags) before any other driver API call.";
                    break;
                case CUResult.ErrorDeinitialized:
                    message = "This indicates that the CUDA driver is in the process of shutting down.";
                    break;
                case CUResult.ErrorProfilerDisabled:
                    message = "This indicates profiling APIs are called while application is running in visual profiler mode.";
                    break;
                //case CUResult.ErrorProfilerNotInitialized:
                //    message = "This indicates profiling has not been initialized for this context. Call cuProfilerInitialize() to resolve this.";
                //    break;
                //case CUResult.ErrorProfilerAlreadyStarted:
                //    message = "This indicates profiler has already been started and probably cuProfilerStart() is incorrectly called.";
                //    break;
                //case CUResult.ErrorProfilerAlreadyStopped:
                //    message = "This indicates profiler has already been stopped and probably cuProfilerStop() is incorrectly called.";
                //    break;
                case CUResult.ErrorStubLibrary:
                    message = "This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.";
                    break;
                case CUResult.ErrorDeviceUnavailable:
                    message = "This indicates that requested CUDA device is unavailable at the current time. Devices are often unavailable due to use of ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS or ::CU_COMPUTEMODE_PROHIBITED.";
                    break;
                case CUResult.ErrorNoDevice:
                    message = "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.";
                    break;
                case CUResult.ErrorInvalidDevice:
                    message = "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.";
                    break;
                case CUResult.DeviceNotLicensed:
                    message = "This error indicates that the Grid license is not applied.";
                    break;
                case CUResult.ErrorInvalidImage:
                    message = "This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module.";
                    break;
                case CUResult.ErrorInvalidContext:
                    message = "This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.";
                    break;
                //CUResult.ErrorContextAlreadyCurrent is marked obsolet since CUDA version 3.2
                //case CUResult.ErrorContextAlreadyCurrent:
                //    message = "This indicated that the context being supplied as a parameter to the API call was already the active context.";
                //    break;
                case CUResult.ErrorMapFailed:
                    message = "This indicates that a map or register operation has failed.";
                    break;
                case CUResult.ErrorUnmapFailed:
                    message = "This indicates that an unmap or unregister operation has failed.";
                    break;
                case CUResult.ErrorArrayIsMapped:
                    message = "This indicates that the specified array is currently mapped and thus cannot be destroyed.";
                    break;
                case CUResult.ErrorAlreadyMapped:
                    message = "This indicates that the resource is already mapped.";
                    break;
                case CUResult.ErrorNoBinaryForGPU:
                    message = "This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.";
                    break;
                case CUResult.ErrorAlreadyAcquired:
                    message = "This indicates that a resource has already been acquired.";
                    break;
                case CUResult.ErrorNotMapped:
                    message = "This indicates that a resource is not mapped.";
                    break;
                case CUResult.ErrorNotMappedAsArray:
                    message = "This indicates that a mapped resource is not available for access as an array.";
                    break;
                case CUResult.ErrorNotMappedAsPointer:
                    message = "This indicates that a mapped resource is not available for access as a pointer.";
                    break;
                case CUResult.ErrorECCUncorrectable:
                    message = "This indicates that an uncorrectable ECC error was detected during execution.";
                    break;
                case CUResult.ErrorUnsupportedLimit:
                    message = "This indicates that the CUlimit passed to the API call is not supported by the active device.";
                    break;
                case CUResult.ErrorContextAlreadyInUse:
                    message = "This indicates that the ::CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.";
                    break;
                case CUResult.ErrorPeerAccessUnsupported:
                    message = "This indicates that peer access is not supported across the given devices.";
                    break;
                case CUResult.ErrorInvalidPtx:
                    message = "This indicates that a PTX JIT compilation failed.";
                    break;
                case CUResult.ErrorInvalidGraphicsContext:
                    message = "This indicates an error with OpenGL or DirectX context.";
                    break;
                case CUResult.NVLinkUncorrectable:
                    message = "This indicates that an uncorrectable NVLink error was detected during the execution.";
                    break;
                case CUResult.JITCompilerNotFound:
                    message = "This indicates that the PTX JIT compiler library was not found.";
                    break;
                case CUResult.UnsupportedPTXVersion:
                    message = "This indicates that the provided PTX was compiled with an unsupported toolchain.";
                    break;
                case CUResult.JITCompilationDisabled:
                    message = "This indicates that the PTX JIT compilation was disabled.";
                    break;
                case CUResult.UnsupportedExecAffinity:
                    message = "This indicates that the ::CUexecAffinityType passed to the API call is not supported by the active device.";
                    break;
                case CUResult.UnsupportedDeviceSync:
                    message = "This indicates that the code to be compiled by the PTX JIT contains unsupported call to cudaDeviceSynchronize.";
                    break;
                case CUResult.ErrorInvalidSource:
                    message = "This indicates that the device kernel source is invalid. This includes compilation/linker errors encountered in device code or user error.";
                    break;
                case CUResult.ErrorFileNotFound:
                    message = "This indicates that the file specified was not found.";
                    break;
                case CUResult.ErrorSharedObjectSymbolNotFound:
                    message = "This indicates that a link to a shared object failed to resolve.";
                    break;
                case CUResult.ErrorSharedObjectInitFailed:
                    message = "This indicates that initialization of a shared object failed.";
                    break;
                case CUResult.ErrorOperatingSystem:
                    message = "This indicates that an OS call failed.";
                    break;
                case CUResult.ErrorInvalidHandle:
                    message = "This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like CUstream and CUevent.";
                    break;
                case CUResult.ErrorNotFound:
                    message = "This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names.";
                    break;
                case CUResult.ErrorNotReady:
                    message = "This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates completion). Calls that may return this value include cuEventQuery() and cuStreamQuery().";
                    break;
                case CUResult.ErrorIllegalAddress:
                    message = "While executing a kernel, the device encountered a load or store instruction on an invalid memory address.\nThis leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorLaunchOutOfResources:
                    message = "This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.";
                    break;
                case CUResult.ErrorLaunchTimeout:
                    message = "This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorLaunchIncompatibleTexturing:
                    message = "This error indicates a kernel launch that uses an incompatible texturing mode.";
                    break;
                case CUResult.ErrorPeerAccessAlreadyEnabled:
                    message = "This error indicates that a call to ::cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled.";
                    break;
                case CUResult.ErrorPeerAccessNotEnabled:
                    message = "This error indicates that ::cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via ::cuCtxEnablePeerAccess().";
                    break;
                case CUResult.ErrorPrimaryContextActice:
                    message = "This error indicates that the primary context for the specified device has already been initialized.";
                    break;
                case CUResult.ErrorContextIsDestroyed:
                    message = "This error indicates that the context current to the calling thread has been destroyed using ::cuCtxDestroy, or is a primary context which has not yet been initialized.";
                    break;
                case CUResult.ErrorAssert:
                    message = "A device-side assert triggered during kernel execution. This leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorTooManyPeers:
                    message = "This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess().";
                    break;
                case CUResult.ErrorHostMemoryAlreadyRegistered:
                    message = "This error indicates that the memory range passed to cuMemHostRegister() has already been registered.";
                    break;
                case CUResult.ErrorHostMemoryNotRegistered:
                    message = "This error indicates that the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region.";
                    break;
                case CUResult.ErrorHardwareStackError:
                    message = "While executing a kernel, the device encountered a stack error.\nThis can be due to stack corruption or exceeding the stack size limit.\nThis leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorIllegalInstruction:
                    message = "While executing a kernel, the device encountered an illegal instruction.\nThis leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorMisalignedAddress:
                    message = "While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned.\nThis leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorInvalidAddressSpace:
                    message = "While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space.\nThis leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorInvalidPC:
                    message = "While executing a kernel, the device program counter wrapped its address space.\nThis leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorLaunchFailed:
                    message = "An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory.\nThis leaves the process in an inconsistent state and any further CUDA work will return the same error.\nTo continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorCooperativeLaunchTooLarge:
                    message = "This error indicates that the number of blocks launched per grid for a kernel that was launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.";
                    break;
                case CUResult.ErrorNotPermitted:
                    message = "This error indicates that the attempted operation is not permitted.";
                    break;
                case CUResult.ErrorNotSupported:
                    message = "This error indicates that the attempted operation is not supported on the current system or device.";
                    break;
                case CUResult.ErrorSystemNotReady:
                    message = "This error indicates that the system is not yet ready to start any CUDA work.  To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running.";
                    break;
                case CUResult.ErrorSystemDriverMismatch:
                    message = "This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.";
                    break;
                case CUResult.ErrorCompatNotSupportedOnDevice:
                    message = "This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.";
                    break;
                case CUResult.MpsConnectionFailed:
                    message = "This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.";
                    break;
                case CUResult.MpsRpcFailure:
                    message = "This error indicates that the remote procedural call between the MPS server and the MPS client failed.";
                    break;
                case CUResult.MpsServerNotReady:
                    message = "This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.";
                    break;
                case CUResult.MpsMaxClientsReached:
                    message = "This error indicates that the hardware resources required to create MPS client have been exhausted.";
                    break;
                case CUResult.MpsMaxConnectionsReached:
                    message = "This error indicates the the hardware resources required to support device connections have been exhausted.";
                    break;
                case CUResult.ErrorMPSClinetTerminated:
                    message = "This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorCDPNotSupported:
                    message = "This error indicates that the module is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.";
                    break;
                case CUResult.ErrorCDPVersionMismatch:
                    message = "This error indicates that a module contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.";
                    break;
                case CUResult.ErrorStreamCaptureUnsupported:
                    message = "This error indicates that the operation is not permitted when the stream is capturing.";
                    break;
                case CUResult.ErrorStreamCaptureInvalidated:
                    message = "This error indicates that the current capture sequence on the stream has been invalidated due to a previous error.";
                    break;
                case CUResult.ErrorStreamCaptureMerge:
                    message = "This error indicates that the operation would have resulted in a merge of two independent capture sequences.";
                    break;
                case CUResult.ErrorStreamCaptureUnmatched:
                    message = "This error indicates that the capture was not initiated in this stream.";
                    break;
                case CUResult.ErrorStreamCaptureUnjoined:
                    message = "This error indicates that the capture sequence contains a fork that was not joined to the primary stream.";
                    break;
                case CUResult.ErrorStreamCaptureIsolation:
                    message = "This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.";
                    break;
                case CUResult.ErrorStreamCaptureImplicit:
                    message = "This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.";
                    break;
                case CUResult.ErrorCapturedEvent:
                    message = "This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream.";
                    break;
                case CUResult.ErrorStreamCaptureWrongThread:
                    message = "A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a different thread.";
                    break;
                case CUResult.ErrorTimeOut:
                    message = "This error indicates that the timeout specified for the wait operation has lapsed.";
                    break;
                case CUResult.ErrorGraphExecUpdateFailure:
                    message = "This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.";
                    break;
                case CUResult.ErrorExternalDevice:
                    message = "This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.";
                    break;
                case CUResult.ErrorInvalidClusterSize:
                    message = "Indicates a kernel launch error due to cluster misconfiguration.";
                    break;
                case CUResult.ErrorUnknown:
                    message = "This indicates that an unknown internal error has occurred.";
                    break;
                default:
                    break;
            }
            return error.ToString() + ": " + message;
        }
        private static string GetInternalNameFromCUResult(CUResult error)
        {
            IntPtr name = new IntPtr();

            DriverAPINativeMethods.ErrorHandling.cuGetErrorName(error, ref name);
            string val = Marshal.PtrToStringAnsi(name);
            return val;
        }

        private static string GetInternalDescriptionFromCUResult(CUResult error)
        {
            IntPtr descr = new IntPtr();

            DriverAPINativeMethods.ErrorHandling.cuGetErrorString(error, ref descr);
            string val = Marshal.PtrToStringAnsi(descr);
            return val;
        }
        #endregion

        #region Properties
        /// <summary>
        /// 
        /// </summary>
        public CUResult CudaError
        {
            get
            {
                return this._cudaError;
            }
            set
            {
                this._cudaError = value;
            }
        }

        /// <summary>
        /// Error name as returned by CUDA driver API
        /// </summary>
        public string CudaInternalErrorName
        {
            get
            {
                return this._internalName;
            }
        }

        /// <summary>
        /// Error description as returned by CUDA driver API
        /// </summary>
        public string CudaInternalErrorDescription
        {
            get
            {
                return this._internalDescripton;
            }
        }
        #endregion
    }
}

