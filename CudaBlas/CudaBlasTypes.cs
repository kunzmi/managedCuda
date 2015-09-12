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

namespace ManagedCuda.CudaBlas
{
	/// <summary>
	/// CUBLAS status type returns
	/// </summary>
	public enum CublasStatus
	{
		/// <summary>
		/// 
		/// </summary>
		Success = 0,
		/// <summary>
		/// 
		/// </summary>
		NotInitialized = 1,
		/// <summary>
		/// 
		/// </summary>
		AllocFailed = 3,
		/// <summary>
		/// 
		/// </summary>
		InvalidValue = 7,
		/// <summary>
		/// 
		/// </summary>
		ArchMismatch = 8,
		/// <summary>
		/// 
		/// </summary>
		MappingError = 11,
		/// <summary>
		/// 
		/// </summary>
		ExecutionFailed = 13,
		/// <summary>
		/// 
		/// </summary>
		InternalError  = 14,
		/// <summary>
		/// 
		/// </summary>
		NotSupported = 15,
		/// <summary>
		/// 
		/// </summary>
		LicenseError = 16
	}

	/// <summary>
	/// The FillMode type indicates which part (lower or upper) of the dense matrix was
	/// filled and consequently should be used by the function. Its values correspond to Fortran
	/// characters ‘L’ or ‘l’ (lower) and ‘U’ or ‘u’ (upper) that are often used as parameters to
	/// legacy BLAS implementations.
	/// </summary>
	public enum FillMode
	{
		/// <summary>
		/// the lower part of the matrix is filled
		/// </summary>
		Lower=0,
		/// <summary>
		/// the upper part of the matrix is filled
		/// </summary>
		Upper=1
	}
	
	/// <summary>
	/// The DiagType type indicates whether the main diagonal of the dense matrix is
	/// unity and consequently should not be touched or modified by the function. Its values
	/// correspond to Fortran characters ‘N’ or ‘n’ (non-unit) and ‘U’ or ‘u’ (unit) that are
	/// often used as parameters to legacy BLAS implementations.
	/// </summary>
	public enum DiagType
	{
		/// <summary>
		/// the matrix diagonal has non-unit elements
		/// </summary>
		NonUnit=0, 
		/// <summary>
		/// the matrix diagonal has unit elements
		/// </summary>
		Unit=1
	}
	
	/// <summary>
	/// The SideMode type indicates whether the dense matrix is on the left or right side
	/// in the matrix equation solved by a particular function. Its values correspond to Fortran
	/// characters ‘L’ or ‘l’ (left) and ‘R’ or ‘r’ (right) that are often used as parameters to
	/// legacy BLAS implementations.
	/// </summary>
	public enum SideMode
	{
		/// <summary>
		/// the matrix is on the left side in the equation
		/// </summary>
		Left=0, 
		/// <summary>
		/// the matrix is on the right side in the equation
		/// </summary>
		Right=1
	}
	
	/// <summary>
	/// The Operation type indicates which operation needs to be performed with the
	/// dense matrix. Its values correspond to Fortran characters ‘N’ or ‘n’ (non-transpose), ‘T’
	/// or ‘t’ (transpose) and ‘C’ or ‘c’ (conjugate transpose) that are often used as parameters
	/// to legacy BLAS implementations
	/// </summary>
	public enum Operation
	{
		/// <summary>
		/// the non-transpose operation is selected
		/// </summary>
		NonTranspose=0, 
		/// <summary>
		/// the transpose operation is selected
		/// </summary>
		Transpose=1, 
		/// <summary>
		/// the conjugate transpose operation is selected
		/// </summary>
		ConjugateTranspose=2
	}
	
	/// <summary>
	/// The PointerMode type indicates whether the scalar values are passed by
	/// reference on the host or device. It is important to point out that if several scalar values are
	/// present in the function call, all of them must conform to the same single pointer mode.
	/// The pointer mode can be set and retrieved using cublasSetPointerMode() and
	/// cublasGetPointerMode() routines, respectively.
	/// </summary>
	public enum PointerMode
	{
		/// <summary>
		/// the scalars are passed by reference on the host
		/// </summary>
		Host=0, 
		/// <summary>
		/// the scalars are passed by reference on the device
		/// </summary>
		Device=1
	}
	
	/// <summary>
	/// The type indicates whether cuBLAS routines which has an alternate implementation
	/// using atomics can be used. The atomics mode can be set and queried using and routines,
	/// respectively.
	/// </summary>
	public enum AtomicsMode
	{
		/// <summary>
		/// the usage of atomics is not allowed
		/// </summary>
		NotAllowed=0, 
		/// <summary>
		/// the usage of atomics is allowed
		/// </summary>
		Allowed=1
	}

	/// <summary>
	/// The cublasDataType_t type is an enumerant to specify the data precision. It is used
	/// when the data reference does not carry the type itself (e.g void *)
	/// </summary>
	public enum DataType
	{
		/// <summary>
		/// the data type is 32-bit floating-point
		/// </summary>
		Float = 0,
		/// <summary>
		/// the data type is 64-bit floating-point
		/// </summary>
		Double = 1,
		/// <summary>
		/// the data type is 16-bit floating-point
		/// </summary>
		Half = 2,
		/// <summary>
		/// the data type is 8-bit signed integer
		/// </summary>
		Int8 = 3
	}

	/// <summary>
	/// Opaque structure holding CUBLAS library context
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CudaBlasHandle
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}
}
