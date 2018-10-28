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
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaFFT
{

    /// <summary>
    /// CUFFT API function return values 
    /// </summary>
    public enum cufftResult
    {
        /// <summary>
        /// Any CUFFT operation is successful.
        /// </summary>
        Success = 0x0,
        /// <summary>
        /// CUFFT is passed an invalid plan handle.
        /// </summary>
        InvalidPlan = 0x1,
        /// <summary>
        /// CUFFT failed to allocate GPU memory.
        /// </summary>
        AllocFailed = 0x2,
        /// <summary>
        /// The user requests an unsupported type.
        /// </summary>
        InvalidType = 0x3,
        /// <summary>
        /// The user specifies a bad memory pointer.
        /// </summary>
        InvalidValue = 0x4,
        /// <summary>
        /// Used for all internal driver errors.
        /// </summary>
        InternalError = 0x5,
        /// <summary>
        /// CUFFT failed to execute an FFT on the GPU.
        /// </summary>
        ExecFailed = 0x6,
        /// <summary>
        /// The CUFFT library failed to initialize.
        /// </summary>
        SetupFailed = 0x7,
        /// <summary>
        /// The user specifies an unsupported FFT size.
        /// </summary>
        InvalidSize = 0x8,
        /// <summary>
        /// Input or output does not satisfy texture alignment requirements.
        /// </summary>
        UnalignedData = 0x9,
		/// <summary>
		/// 
		/// </summary>
		IncompleteParameterList = 0xA,
		/// <summary>
		/// Plan creation and execution are on different device
		/// </summary>
		InvalidDevice = 0xB,
		/// <summary>
		/// 
		/// </summary>
		ParseError = 0xC,
		/// <summary>
		/// Workspace not initialized
		/// </summary>
		NoWorkspace = 0xD,
		/// <summary>
		/// Not implemented
		/// </summary>
		NotImplemented = 0xE,
		/// <summary>
		/// License error
		/// </summary>
		LicenseError = 0xF,
		/// <summary>
		/// Not supported error
		/// </summary>
		NotSupported = 0x10
    }

    /// <summary>
    /// cufftHandle is a handle type used to store and access CUFFT plans.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cufftHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public uint Handle;

		/// <summary>
		/// Creates only an opaque handle, and allocates small data structures on the host. The
		/// cufftMakePlan*() calls actually do the plan generation. It is recommended that
		/// cufftSet*() calls, such as cufftSetCompatibilityMode(), that may require a plan
		/// to be broken down and re-generated, should be made after cufftCreate() and before
		/// one of the cufftMakePlan*() calls.
		/// </summary>
		/// <returns></returns>
		public static cufftHandle Create()
		{
			cufftHandle handle = new cufftHandle();

			cufftResult res = CudaFFTNativeMethods.cufftCreate(ref handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftCreate", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);

			return handle;
		}


		/// <summary>
		/// SetWorkArea() overrides the work area pointer associated with a plan.
		/// If the work area was auto-allocated, CUFFT frees the auto-allocated space. The
		/// cufftExecute*() calls assume that the work area pointer is valid and that it points to
		/// a contiguous region in device memory that does not overlap with any other work area. If
		/// this is not the case, results are indeterminate.
		/// </summary>
		/// <param name="workArea"></param>
		public void SetWorkArea(CUdeviceptr workArea)
		{
			cufftResult res = CudaFFTNativeMethods.cufftSetWorkArea(this, workArea);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetWorkArea", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}

		/// <summary>
		/// SetAutoAllocation() indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated. CUFFT default behavior is to allocate
		/// the work area at plan generation time. If cufftSetAutoAllocation() has been called
		/// with autoAllocate set to "false" prior to one of the cufftMakePlan*() calls, CUFFT
		/// does not allocate the work area. This is the preferred sequence for callers wishing to
		/// manage work area allocation.
		/// </summary>
		/// <param name="autoAllocate"></param>
		public void SetAutoAllocation(bool autoAllocate)
		{
			int auto = 0;
			if (autoAllocate) auto = 1;
			cufftResult res = CudaFFTNativeMethods.cufftSetAutoAllocation(this, auto);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetAutoAllocation", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}


		/// <summary>
		/// Associates a CUDA stream with a CUFFT plan. All kernel launches
		/// made during plan execution are now done through the associated
		/// stream, enabling overlap with activity in other streams (for example,
		/// data copying). The association remains until the plan is destroyed or
		/// the stream is changed with another call to SetStream().
		/// </summary>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		public void SetStream(CUstream stream)
		{
			cufftResult res = CudaFFTNativeMethods.cufftSetStream(this, stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetStream", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}

		///// <summary>
		///// configures the layout of CUFFT output in FFTW‐compatible modes.
		///// When FFTW compatibility is desired, it can be configured for padding
		///// only, for asymmetric complex inputs only, or to be fully compatible.
		///// </summary>
		///// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		//public void SetCompatibilityMode(Compatibility mode)
		//{
		//	cufftResult res = CudaFFTNativeMethods.cufftSetCompatibilityMode(this, mode);
		//	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetCompatibilityMode", res));
		//	if (res != cufftResult.Success)
		//		throw new CudaFFTException(res);
		//}
    }

    /// <summary>
    /// CUFFT transform directions 
    /// </summary>
    public enum TransformDirection
    {
        /// <summary>
        /// 
        /// </summary>
        Forward = -1,
        /// <summary>
        /// 
        /// </summary>
        Inverse = 1
    }

    /// <summary>
    /// CUFFT supports the following transform types 
    /// </summary>
    public enum cufftType
    {
        /// <summary>
        /// Real to Complex (interleaved)
        /// </summary>
        R2C = 0x2a,
        /// <summary>
        /// Complex (interleaved) to Real
        /// </summary>
        C2R = 0x2c,
        /// <summary>
        /// Complex to Complex, interleaved
        /// </summary>
        C2C = 0x29,
        /// <summary>
        /// Double to Double-Complex
        /// </summary>
        D2Z = 0x6a,
        /// <summary>
        /// Double-Complex to Double
        /// </summary>
        Z2D = 0x6c,
        /// <summary>
        /// Double-Complex to Double-Complex
        /// </summary>
        Z2Z = 0x69
    }

    ///// <summary>
    ///// Certain R2C and C2R transforms go much more slowly when FFTW memory
    ///// layout and behaviour is required. The default is "best performance",
    ///// which means not-compatible-with-fftw. Use the <see cref="CudaFFTNativeMethods.cufftSetCompatibilityMode"/>
    ///// API to enable exact FFTW-like behaviour.
    ///// </summary>
    //[Flags]
    //public enum Compatibility
    //{
    //    /// <summary>
    //    /// Default value
    //    /// </summary>
    //    FFTWPadding = 0x01
    //}
}
