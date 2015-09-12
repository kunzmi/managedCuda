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
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace ManagedCuda.CudaSparse
{
	#region Enums
	/// <summary>
	/// This is a status type returned by the library functions and it can have the following values.
	/// </summary>
	public enum cusparseStatus
	{
		/// <summary>
		/// The operation completed successfully.
		/// </summary>
		Success = 0,
		/// <summary>
		/// "The CUSPARSE library was not initialized. This is usually caused by the lack of a prior 
		/// cusparseCreate() call, an error in the CUDA Runtime API called by the CUSPARSE routine, or an 
		/// error in the hardware setup. To correct: call cusparseCreate() prior to the function call; and
		///  check that the hardware, an appropriate version of the driver, and the CUSPARSE library are 
		/// correctly installed.
		/// </summary>
		NotInitialized = 1,
		/// <summary>
		///  "Resource allocation failed inside the CUSPARSE library. This is usually caused by a 
		/// cudaMalloc() failure. To correct: prior to the function call, deallocate previously allocated
		/// memory as much as possible.
		/// </summary>
		AllocFailed = 2,
		/// <summary>
		/// "An unsupported value or parameter was passed to the function (a negative vector size, 
		/// for example). To correct: ensure that all the parameters being passed have valid values.
		/// </summary>
		InvalidValue = 3,
		/// <summary>
		/// "The function requires a feature absent from the device architecture; usually caused by 
		/// the lack of support for atomic operations or double precision. To correct: compile and run the
		///  application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic 
		/// operations and 1.3 for double precision.
		/// </summary>
		ArchMismatch = 4,
		/// <summary>
		/// "An access to GPU memory space failed, which is usually caused by a failure to bind a texture. 
		/// To correct: prior to the function call, unbind any previously bound textures.
		/// </summary>
		MappingError = 5,
		/// <summary>
		/// "The GPU program failed to execute. This is often caused by a launch failure of the kernel on 
		/// the GPU, which can be caused by multiple reasons. To correct: check that the hardware, an appropriate
		///  version of the driver, and the CUSPARSE library are correctly installed.
		/// </summary>
		ExecutionFailed = 6,
		/// <summary>
		/// "An internal CUSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() 
		/// failure. To correct: check that the hardware, an appropriate version of the driver, and the CUSPARSE
		///  library are correctly installed. Also, check that the memory passed as a parameter to the routine 
		/// is not being deallocated prior to the routine’s completion.
		/// </summary>
		InternalError = 7,
		/// <summary>
		/// "The matrix type is not supported by this function. This is usually caused by passing an invalid 
		/// matrix descriptor to the function. To correct: check that the fields in cusparseMatDescr_t descrA were 
		/// set correctly.
		/// </summary>
		MatrixTypeNotSupported = 8,
		/// <summary>
		///
		/// </summary>
		ZeroPivot = 9
	}

	/// <summary>
	/// This type indicates whether the scalar values are passed by reference on the host or device.
	/// It is important to point out that if several scalar values are passed by reference in the
	/// function call, all of them will conform to the same single pointer mode. The pointer mode
	/// can be set and retrieved using <see cref="CudaSparseContext.SetPointerMode"/> and
	/// <see cref="CudaSparseContext.GetPointerMode()"/> routines, respectively.
	/// </summary>
	public enum cusparsePointerMode
	{
		/// <summary>
		/// Use host pointers.
		/// </summary>
		Host = 0,
		/// <summary>
		/// Use device pointers.
		/// </summary>
		Device = 1
	}

	/// <summary>
	/// This type indicates whether the operation is performed only on indices or on data and indices.
	/// </summary>
	public enum cusparseAction
	{
		/// <summary>
		/// the operation is performed only on indices.
		/// </summary>
		Symbolic = 0,
		/// <summary>
		/// the operation is performed on data and indices.
		/// </summary>
		Numeric = 1
	}

	/// <summary>
	/// This type indicates the type of matrix stored in sparse storage. Notice that for symmetric,
	/// Hermitian and triangular matrices only their lower or upper part is assumed to be stored.
	/// </summary>
	public enum cusparseMatrixType
	{
		/// <summary>
		/// the matrix is general.
		/// </summary>
		General = 0,
		/// <summary>
		/// the matrix is symmetric.
		/// </summary>
		Symmetric = 1,
		/// <summary>
		/// the matrix is Hermitian.
		/// </summary>
		Hermitian = 2,
		/// <summary>
		/// the matrix is triangular.
		/// </summary>
		Triangular = 3
	}

	/// <summary>
	/// This type indicates if the lower or upper part of a matrix is stored in sparse storage.
	/// </summary>
	public enum cusparseFillMode
	{
		/// <summary>
		/// the lower triangular part is stored.
		/// </summary>
		Lower = 0,
		/// <summary>
		/// the upper triangular part is stored.
		/// </summary>
		Upper = 1
	}

	/// <summary>
	/// This type indicates if the matrix diagonal entries are unity. The diagonal elements are
	/// always assumed to be present, but if CUSPARSE_DIAG_TYPE_UNIT is passed to an API
	/// routine, then the routine will assume that all diagonal entries are unity and will not read
	/// or modify those entries. Note that in this case the routine assumes the diagonal entries are
	/// equal to one, regardless of what those entries are actuall set to in memory.
	/// </summary>
	public enum cusparseDiagType
	{
		/// <summary>
		/// the matrix diagonal has non-unit elements.
		/// </summary>
		NonUnit = 0,
		/// <summary>
		/// the matrix diagonal has unit elements.
		/// </summary>
		Unit = 1
	}

	/// <summary>
	/// This type indicates if the base of the matrix indices is zero or one.
	/// </summary>
	public enum cusparseIndexBase
	{
		/// <summary>
		/// the base index is zero.
		/// </summary>
		Zero = 0,
		/// <summary>
		/// the base index is one.
		/// </summary>
		One = 1
	}

	/// <summary>
	/// This type indicates which operations need to be performed with the sparse matrix.
	/// </summary>
	public enum cusparseOperation
	{
		/// <summary>
		/// the non-transpose operation is selected.
		/// </summary>
		NonTranspose = 0,
		/// <summary>
		/// the transpose operation is selected.
		/// </summary>
		Transpose = 1,
		/// <summary>
		/// the conjugate transpose operation is selected.
		/// </summary>
		ConjugateTranspose = 2
	}

	/// <summary>
	/// This type indicates whether the elements of a dense matrix should be parsed by rows or by
	/// columns (assuming column-major storage in memory of the dense matrix).
	/// </summary>
	public enum cusparseDirection
	{
		/// <summary>
		/// the matrix should be parsed by rows.
		/// </summary>
		Row = 0,
		/// <summary>
		/// the matrix should be parsed by columns.
		/// </summary>
		Column = 1
	}

	/// <summary>
	/// This type indicates how to perform the partitioning of the matrix into regular (ELL) and
	/// irregular (COO) parts of the HYB format.<para/>
	/// The partitioning is performed during the conversion of the matrix from a dense or sparse
	/// format into the HYB format and is governed by the following rules. When
	/// CUSPARSE_HYB_PARTITION_AUTO is selected, the CUSPARSE library automatically decides
	/// how much data to put into the regular and irregular parts of the HYB format. When
	/// CUSPARSE_HYB_PARTITION_USER is selected, the width of the regular part of the HYB
	/// format should be specified by the caller. When CUSPARSE_HYB_PARTITION_MAX is selected,
	/// the width of the regular part of the HYB format equals to the maximum number of
	/// non-zero elements per row, in other words, the entire matrix is stored in the regular part of
	/// the HYB format.<para/>
	/// The default is to let the library automatically decide how to split the data.
	/// </summary>
	public enum cusparseHybPartition
	{
		/// <summary>
		/// the automatic partitioning is selected (default).
		/// </summary>
		Auto = 0,
		/// <summary>
		/// the user specified treshold is used.
		/// </summary>
		User = 1,
		/// <summary>
		/// the data is stored in ELL format.
		/// </summary>
		Max = 2
	}

	/// <summary>
	/// used in csrsv2, csric02, and csrilu02
	/// </summary>
	public enum cusparseSolvePolicy
	{
		/// <summary>
		/// no level information is generated, only reports structural zero.
		/// </summary>
		NoLevel = 0,
		/// <summary>
		/// 
		/// </summary>
		UseLevel = 1
	}

	/// <summary>
	/// 
	/// </summary>
	public enum cusparseSideMode
	{
		/// <summary>
		/// 
		/// </summary>
		Left = 0,
		/// <summary>
		/// 
		/// </summary>
		Right = 1
	}

	#endregion

	#region structs (opaque handles)
	/// <summary>
	/// Opaque structure holding CUSPARSE library context
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusparseContext
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the matrix descriptor
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusparseMatDescr
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusparseSolveAnalysisInfo
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct csrsv2Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct bsrsv2Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct csric02Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct bsric02Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct bsrsm2Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct csrilu02Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sparse triangular solve information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct bsrilu02Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the hybrid (HYB) storage information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusparseHybMat
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding sparse gemm information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct csrgemm2Info
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the sorting information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct csru2csrInfo
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the coloring information
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cusparseColorInfo
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}
	#endregion
}
