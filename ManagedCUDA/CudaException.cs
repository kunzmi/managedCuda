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
using ManagedCuda.BasicTypes;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.InteropServices;

namespace ManagedCuda
{
	/// <summary>
	/// A CUDA exception is thrown if a CUDA Driver API method call does not return <see cref="CUResult.Success"/>
	/// </summary>
	[Serializable] 
	public class CudaException : Exception, System.Runtime.Serialization.ISerializable
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
				case CUResult.ErrorNoDevice:
					message = "This indicates that no CUDA-capable devices were detected by the installed CUDA driver.";
					break;
				case CUResult.ErrorInvalidDevice:
					message = "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device.";
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
				case CUResult.ErrorInvalidSource:
					message = "This indicates that the device kernel source is invalid.";
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
					message = "While executing a kernel, the device encountered a load or store instruction on an invalid memory address.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
					break;
				case CUResult.ErrorLaunchOutOfResources:
					message = "This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.";
					break;
				case CUResult.ErrorLaunchTimeout:
					message = "This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The context cannot be used (and must be destroyed similar to CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
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
					message = "A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
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
					message = "While executing a kernel, the device encountered a stack error.\nThis can be due to stack corruption or exceeding the stack size limit.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
					break;
				case CUResult.ErrorIllegalInstruction:
					message = "While executing a kernel, the device encountered an illegal instruction.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
					break;
				case CUResult.ErrorMisalignedAddress:
					message = "While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
					break;
				case CUResult.ErrorInvalidAddressSpace:
					message = "While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
					break;
				case CUResult.ErrorInvalidPC:
					message = "While executing a kernel, the device program counter wrapped its address space.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
					break;
				case CUResult.ErrorLaunchFailed:
					message = "An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.";
					break;
				case CUResult.ErrorNotPermitted:
					message = "This error indicates that the attempted operation is not permitted.";
					break;
				case CUResult.ErrorNotSupported:
					message = "This error indicates that the attempted operation is not supported on the current system or device.";
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

 