//	Copyright (c) 2015, Michael Kunz. All rights reserved.
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
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaSparse;


namespace ManagedCuda.CudaSolve
{
	#region common
	/// <summary>
	/// This is a status type returned by the library functions and it can have the following values.
	/// </summary>
	public enum cusolverStatus
	{
		/// <summary>
		/// The operation completed successfully
		/// </summary>
		Success=0,
		/// <summary>
		/// The cuSolver library was not initialized. This is usually caused by the
		/// lack of a prior call, an error in the CUDA Runtime API called by the
		/// cuSolver routine, or an error in the hardware setup.<para/>
		/// To correct: call cusolverCreate() prior to the function call; and
		/// check that the hardware, an appropriate version of the driver, and the
		/// cuSolver library are correctly installed.
		/// </summary>
		NotInititialized=1,
		/// <summary>
		/// Resource allocation failed inside the cuSolver library. This is usually
		/// caused by a cudaMalloc() failure.<para/>
		/// To correct: prior to the function call, deallocate previously allocated
		/// memory as much as possible.
		/// </summary>
		AllocFailed=2,
		/// <summary>
		/// An unsupported value or parameter was passed to the function (a
		/// negative vector size, for example).<para/>
		/// To correct: ensure that all the parameters being passed have valid
		/// values.
		/// </summary>
		InvalidValue=3,
		/// <summary>
		/// The function requires a feature absent from the device architecture;
		/// usually caused by the lack of support for atomic operations or double
		/// precision.<para/>
		/// To correct: compile and run the application on a device with compute
		/// capability 2.0 or above.
		/// </summary>
		ArchMismatch = 4,
		/// <summary>
		/// 
		/// </summary>
		MappingError = 5,
		/// <summary>
		/// The GPU program failed to execute. This is often caused by a launch
		/// failure of the kernel on the GPU, which can be caused by multiple
		/// reasons.<para/>
		/// To correct: check that the hardware, an appropriate version of the
		/// driver, and the cuSolver library are correctly installed.
		/// </summary>
		ExecutionFailed = 6,
		/// <summary>
		/// An internal cuSolver operation failed. This error is usually caused by a
		/// cudaMemcpyAsync() failure.<para/>
		/// To correct: check that the hardware, an appropriate version of the
		/// driver, and the cuSolver library are correctly installed. Also, check
		/// that the memory passed as a parameter to the routine is not being
		/// deallocated prior to the routine’s completion.
		/// </summary>
		InternalError = 7,
		/// <summary>
		/// The matrix type is not supported by this function. This is usually caused
		/// by passing an invalid matrix descriptor to the function.<para/>
		/// To correct: check that the fields in descrA were set correctly.
		/// </summary>
		MatrixTypeNotSupported = 8,
		/// <summary>
		/// 
		/// </summary>
		NotSupported = 9,
		/// <summary>
		/// 
		/// </summary>
		ZeroPivot = 10,
		/// <summary>
		/// 
		/// </summary>
		InvalidLicense=11
	}
	#endregion

	#region Dense
	/// <summary>
	/// This is a pointer type to an opaque cuSolverDN context
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusolverDnHandle
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	#endregion

	#region Sparse
	/// <summary>
	/// This is a pointer type to an opaque cuSolverSP context
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusolverSpHandle
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	/// <summary>
	/// This is a pointer type to an opaque csrqrInfo
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct csrqrInfo
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}
	#endregion

	#region Refactorization
	/// <summary>
	/// This is a pointer type to an opaque cuSolverRF context
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusolverRfHandle
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}


	/// <summary>
	/// The ResetValuesFastMode is an enum that indicates the mode used for
	/// the cusolverRfResetValues() routine. The fast mode requires extra memory and is
	/// recommended only if very fast calls to cusolverRfResetValues() are needed.
	/// </summary>
	public enum ResetValuesFastMode
	{
		/// <summary>
		/// default
		/// </summary>
		Off = 0, 
		/// <summary/>  
		On = 1
	}

	/// <summary>
	/// The MatrixFormat is an enum that indicates the input/output
	/// matrix format assumed by the cusolverRfSetup(), cusolverRfSetupHost(),
	/// cusolverRfResetValues(), cusolveRfExtractBundledFactorsHost() and
	/// cusolverRfExtractSplitFactorsHost() routines.
	/// </summary>
	public enum MatrixFormat
	{
		/// <summary>
		/// default
		/// </summary>
		Csr = 0,   
		/// <summary/>
		Csc = 1
	}

	/// <summary>
	/// The UnitDiagonal is an enum that indicates whether
	/// and where the unit diagonal is stored in the input/output triangular
	/// factors in the cusolverRfSetup(), cusolverRfSetupHost() and
	/// cusolverRfExtractSplitFactorsHost() routines.
	/// </summary>
	public enum UnitDiagonal
	{
		/// <summary>
		/// unit diagonal is stored in lower triangular factor. (default)
		/// </summary>
		StoredL = 0,
		/// <summary>
		/// unit diagonal is stored in upper triangular factor.
		/// </summary>
		StoredU = 1,
		/// <summary>
		/// unit diagonal is assumed in lower triangular factor.
		/// </summary>
		AssumedL = 2,
		/// <summary>
		/// unit diagonal is assumed in upper triangular factor.
		/// </summary>
		AssumedU = 3
	}

	/// <summary>
	/// The Factorization is an enum that indicates which (internal)
	/// algorithm is used for refactorization in the cusolverRfRefactor() routine.
	/// </summary>
	public enum Factorization
	{
		/// <summary>
		/// algorithm 0. (default)
		/// </summary>
		Alg0 = 0,
		/// <summary>
		/// algorithm 1.
		/// </summary>
		Alg1 = 1,
		/// <summary>
		/// algorithm 2. Domino-based scheme.
		/// </summary>
		Alg2 = 2,
	}

	/// <summary>
	/// The TriangularSolve is an enum that indicates which (internal)
	/// algorithm is used for triangular solve in the cusolverRfSolve() routine.
	/// </summary>
	public enum TriangularSolve
	{
		/// <summary>
		/// algorithm 0.
		/// </summary>
		Alg0 = 0,
		/// <summary>
		/// algorithm 1. (default)
		/// </summary>
		Alg1 = 1,
		/// <summary>
		/// algorithm 2. Domino-based scheme.
		/// </summary>
		Alg2 = 2,
		/// <summary>
		/// algorithm 3. Domino-based scheme.
		/// </summary>
		Alg3 = 3
	}

	/// <summary>
	/// The cusolverRfNumericBoostReport_t is an enum that indicates whether
	/// numeric boosting (of the pivot) was used during the cusolverRfRefactor() and
	/// cusolverRfSolve() routines. The numeric boosting is disabled by default.
	/// </summary>
	public enum NumericBoostReport
	{
		/// <summary>
		/// default
		/// </summary>
		NotUsed = 0, 
		/// <summary/>
		Used = 1
	}
	#endregion
}
