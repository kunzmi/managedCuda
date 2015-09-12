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
using ManagedCuda.VectorTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaSparse;


namespace ManagedCuda.CudaSolve
{
	/// <summary/>
	public static class CudaSolveNativeMethods
	{
#if _x64
		internal const string CUSOLVE_API_DLL_NAME = "cusolver64_75.dll";
#else
		internal const string CUSOLVE_API_DLL_NAME = "cusolver32_75.dll";
#endif
		/// <summary>
		/// The cuSolverDN library was designed to solve dense linear systems of the form Ax=B
		/// </summary>
		public static class Dense
		{
			#region Init
			/// <summary>
			/// This function initializes the cuSolverDN library and creates a handle on the cuSolverDN
			/// context. It must be called before any other cuSolverDN API function is invoked. It
			/// allocates hardware resources necessary for accessing the GPU
			/// </summary>
			/// <param name="handle">the pointer to the handle to the cuSolverDN context.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCreate(ref cusolverDnHandle handle);
			
			/// <summary>
			/// This function releases CPU-side resources used by the cuSolverDN library.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDestroy(cusolverDnHandle handle);
			
			/// <summary>
			/// This function sets the stream to be used by the cuSolverDN library to execute its routines.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="streamId">the stream to be used by the library.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSetStream(cusolverDnHandle handle, CUstream streamId);
			
			/// <summary>
			/// This function sets the stream to be used by the cuSolverDN library to execute its routines.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="sreamId">the stream to be used by the library.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnGetStream(cusolverDnHandle handle, ref CUstream sreamId);
			#endregion

			#region Cholesky factorization and its solver
			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSpotrf_bufferSize(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDpotrf_bufferSize(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCpotrf_bufferSize(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZpotrf_bufferSize(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Workspace</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
			/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSpotrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Workspace</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
			/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDpotrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Workspace</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
			/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCpotrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Workspace</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
			/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZpotrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of columns of matrix X and B.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n). A is either
			/// lower cholesky factor L or upper Cholesky factor U.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="B">array of dimension ldb * nrhs. ldb is not less than max(1,n). As an input, B is right hand side matrix. As an
			/// output, B is the solution matrix.</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo =
			/// -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSpotrs(cusolverDnHandle handle, FillMode uplo, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, CUdeviceptr devInfo);

			/// <summary>
			/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of columns of matrix X and B.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n). A is either
			/// lower cholesky factor L or upper Cholesky factor U.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="B">array of dimension ldb * nrhs. ldb is not less than max(1,n). As an input, B is right hand side matrix. As an
			/// output, B is the solution matrix.</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo =
			/// -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDpotrs(cusolverDnHandle handle, FillMode uplo, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, CUdeviceptr devInfo);

			/// <summary>
			/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of columns of matrix X and B.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n). A is either
			/// lower cholesky factor L or upper Cholesky factor U.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="B">array of dimension ldb * nrhs. ldb is not less than max(1,n). As an input, B is right hand side matrix. As an
			/// output, B is the solution matrix.</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo =
			/// -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCpotrs(cusolverDnHandle handle, FillMode uplo, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, CUdeviceptr devInfo);

			/// <summary>
			/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of columns of matrix X and B.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n). A is either
			/// lower cholesky factor L or upper Cholesky factor U.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="B">array of dimension ldb * nrhs. ldb is not less than max(1,n). As an input, B is right hand side matrix. As an
			/// output, B is the solution matrix.</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo =
			/// -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZpotrs(cusolverDnHandle handle, FillMode uplo, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, CUdeviceptr devInfo);
			#endregion

			#region LU Factorization
			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgetrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgetrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgetrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of Workspace</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgetrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is
			/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgetrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, CUdeviceptr devIpiv, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is
			/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgetrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, CUdeviceptr devIpiv, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is
			/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgetrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, CUdeviceptr devIpiv, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the LU factorization of a m×n matrix P*A=L*U
			/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
			/// unit diagonal, and U is an upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is
			/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgetrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr Workspace, CUdeviceptr devIpiv, CUdeviceptr devInfo);
			#endregion

			#region Row pivoting
			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnClaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);
			#endregion

			#region LU solve
			/// <summary>
			/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of right-hand sides.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
			/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgetrs(cusolverDnHandle handle, Operation trans, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr devIpiv, CUdeviceptr B, int ldb, CUdeviceptr devInfo);

			/// <summary>
			/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of right-hand sides.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
			/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgetrs(cusolverDnHandle handle, Operation trans, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr devIpiv, CUdeviceptr B, int ldb, CUdeviceptr devInfo);

			/// <summary>
			/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of right-hand sides.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
			/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgetrs(cusolverDnHandle handle, Operation trans, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr devIpiv, CUdeviceptr B, int ldb, CUdeviceptr devInfo);

			/// <summary>
			/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nrhs">number of right-hand sides.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
			/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
			/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgetrs(cusolverDnHandle handle, Operation trans, int n, int nrhs, CUdeviceptr A, int lda, CUdeviceptr devIpiv, CUdeviceptr B, int ldb, CUdeviceptr devInfo);
			#endregion

			#region QR factorization
			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="TAU">array of dimension at least min(m,n).</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgeqrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr TAU, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="TAU">array of dimension at least min(m,n).</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgeqrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr TAU, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="TAU">array of dimension at least min(m,n).</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgeqrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr TAU, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="TAU">array of dimension at least min(m,n).</param>
			/// <param name="Workspace">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgeqrf(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr TAU, CUdeviceptr Workspace, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function overwrites m×n matrix C by 
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="side">indicates if matrix Q is on the left or right of C.</param>
			/// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="k">number of elementary relfections.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="tau">array of dimension at least min(m,n). The vector tau is from geqrf,
			/// so tau(i) is the scalar of i-th elementary reflection vector.</param>
			/// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
			/// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc >= max(1,m).</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working array work.</param>
			/// <param name="devInfo">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSormqr(cusolverDnHandle handle, SideMode side, Operation trans, int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, CUdeviceptr work, int lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function overwrites m×n matrix C by 
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="side">indicates if matrix Q is on the left or right of C.</param>
			/// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="k">number of elementary relfections.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="tau">array of dimension at least min(m,n). The vector tau is from geqrf,
			/// so tau(i) is the scalar of i-th elementary reflection vector.</param>
			/// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
			/// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc >= max(1,m).</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working array work.</param>
			/// <param name="devInfo">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDormqr(cusolverDnHandle handle, SideMode side, Operation trans, int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, CUdeviceptr work, int lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function overwrites m×n matrix C by 
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="side">indicates if matrix Q is on the left or right of C.</param>
			/// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="k">number of elementary relfections.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="tau">array of dimension at least min(m,n). The vector tau is from geqrf,
			/// so tau(i) is the scalar of i-th elementary reflection vector.</param>
			/// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
			/// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc >= max(1,m).</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working array work.</param>
			/// <param name="devInfo">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCunmqr(cusolverDnHandle handle, SideMode side, Operation trans, int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, CUdeviceptr work, int lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function overwrites m×n matrix C by 
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="side">indicates if matrix Q is on the left or right of C.</param>
			/// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="k">number of elementary relfections.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="tau">array of dimension at least min(m,n). The vector tau is from geqrf,
			/// so tau(i) is the scalar of i-th elementary reflection vector.</param>
			/// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
			/// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc >= max(1,m).</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working array work.</param>
			/// <param name="devInfo">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZunmqr(cusolverDnHandle handle, SideMode side, Operation trans, int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, CUdeviceptr work, int lwork, CUdeviceptr devInfo);
			#endregion

			#region QR factorization workspace query
			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgeqrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgeqrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgeqrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the QR factorization of a m×n matrix A=Q*R
			/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working array Workspace.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgeqrf_bufferSize(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, ref int Lwork);
			#endregion

			#region bidiagonal
			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="D">array of dimension min(m,n). The diagonal elements of the bidiagonal
			/// matrix B: D(i) = A(i,i).</param>
			/// <param name="E">array of dimension min(m,n). The off-diagonal elements of the bidiagonal
			/// matrix B: if m&gt;=n, E(i) = A(i,i+1) for i = 1,2,...,n-1; if m&lt;n, E(i) = A(i+1,i) for i = 1,2,...,m-1.</param>
			/// <param name="TAUQ">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix Q.</param>
			/// <param name="TAUP">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix P.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgebrd(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr TAUQ, CUdeviceptr TAUP, CUdeviceptr Work, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="D">array of dimension min(m,n). The diagonal elements of the bidiagonal
			/// matrix B: D(i) = A(i,i).</param>
			/// <param name="E">array of dimension min(m,n). The off-diagonal elements of the bidiagonal
			/// matrix B: if m&gt;=n, E(i) = A(i,i+1) for i = 1,2,...,n-1; if m&lt;n, E(i) = A(i+1,i) for i = 1,2,...,m-1.</param>
			/// <param name="TAUQ">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix Q.</param>
			/// <param name="TAUP">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix P.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgebrd(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr TAUQ, CUdeviceptr TAUP, CUdeviceptr Work, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="D">array of dimension min(m,n). The diagonal elements of the bidiagonal
			/// matrix B: D(i) = A(i,i).</param>
			/// <param name="E">array of dimension min(m,n). The off-diagonal elements of the bidiagonal
			/// matrix B: if m&gt;=n, E(i) = A(i,i+1) for i = 1,2,...,n-1; if m&lt;n, E(i) = A(i+1,i) for i = 1,2,...,m-1.</param>
			/// <param name="TAUQ">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix Q.</param>
			/// <param name="TAUP">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix P.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgebrd(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr TAUQ, CUdeviceptr TAUP, CUdeviceptr Work, int Lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="D">array of dimension min(m,n). The diagonal elements of the bidiagonal
			/// matrix B: D(i) = A(i,i).</param>
			/// <param name="E">array of dimension min(m,n). The off-diagonal elements of the bidiagonal
			/// matrix B: if m&gt;=n, E(i) = A(i,i+1) for i = 1,2,...,n-1; if m&lt;n, E(i) = A(i+1,i) for i = 1,2,...,m-1.</param>
			/// <param name="TAUQ">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix Q.</param>
			/// <param name="TAUP">array of dimension min(m,n). The scalar factors of the elementary reflectors
			/// which represent the orthogonal matrix P.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgebrd(cusolverDnHandle handle, int m, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr TAUQ, CUdeviceptr TAUP, CUdeviceptr Work, int Lwork, CUdeviceptr devInfo);


			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSsytrd(cusolverDnHandle handle, char uplo, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr tau, CUdeviceptr Work, int Lwork, CUdeviceptr info);

			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDsytrd(cusolverDnHandle handle, char uplo, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr tau, CUdeviceptr Work, int Lwork, CUdeviceptr info);
			#endregion

			#region bidiagonal factorization workspace query
			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgebrd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);

			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgebrd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);

			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgebrd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);

			/// <summary>
			/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
			/// an orthogonal transformation: Q^H*A*P=B
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gebrd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgebrd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);
			#endregion

			#region singular value decomposition, A = U * Sigma * V^H
			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgesvd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);

			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgesvd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);

			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgesvd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);

			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgesvd_bufferSize(cusolverDnHandle handle, int m, int n, ref int Lwork);

			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="jobu">specifies options for computing all or part of the matrix U: = 'A': all m columns of
			/// U are returned in array U: = 'S': the first min(m,n) columns of U (the left singular
			/// vectors) are returned in the array U; = 'O': the first min(m,n) columns of U (the left singular vectors) are overwritten on
			/// the array A; = 'N': no columns of U (no left singular vectors) are computed.</param>
			/// <param name="jobvt">specifies options for computing all or part of the matrix V**T: = 'A': all N rows
			/// of V**T are returned in the array VT; = 'S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the
			/// array VT; = 'O': the first min(m,n) rows of V**T (the right singular vectors) are overwritten on the array A; = 'N': no rows
			/// of V**T (no right singular vectors) are computed.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m). On exit,
			/// the contents of A are destroyed.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="S">array of dimension min(m,n). The singular values of A, sorted so that S(i)
			/// &gt;= S(i+1).</param>
			/// <param name="U">array of dimension ldu * m with ldu is not less than max(1,m). U contains
			/// the m×m unitary matrix U.</param>
			/// <param name="ldu">leading dimension of two-dimensional array used to store matrix U.</param>
			/// <param name="VT">array of dimension ldvt * n with ldvt is not less than max(1,n). VT
			/// contains the n×n unitary matrix V**T.</param>
			/// <param name="ldvt">leading dimension of two-dimensional array used to store matrix Vt.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			/// <param name="rwork"></param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the ith
			/// parameter is wrong. if devInfo &gt; 0, devInfo indicates how many superdiagonals of an intermediate
			/// bidiagonal form B did not converge to zero.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSgesvd(cusolverDnHandle handle, [MarshalAs(UnmanagedType.I1)] char jobu, [MarshalAs(UnmanagedType.I1)] char jobvt, int m, int n, CUdeviceptr A, int lda, CUdeviceptr S, CUdeviceptr U, int ldu, CUdeviceptr VT, int ldvt, CUdeviceptr Work, int Lwork, CUdeviceptr rwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="jobu">specifies options for computing all or part of the matrix U: = 'A': all m columns of
			/// U are returned in array U: = 'S': the first min(m,n) columns of U (the left singular
			/// vectors) are returned in the array U; = 'O': the first min(m,n) columns of U (the left singular vectors) are overwritten on
			/// the array A; = 'N': no columns of U (no left singular vectors) are computed.</param>
			/// <param name="jobvt">specifies options for computing all or part of the matrix V**T: = 'A': all N rows
			/// of V**T are returned in the array VT; = 'S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the
			/// array VT; = 'O': the first min(m,n) rows of V**T (the right singular vectors) are overwritten on the array A; = 'N': no rows
			/// of V**T (no right singular vectors) are computed.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m). On exit,
			/// the contents of A are destroyed.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="S">array of dimension min(m,n). The singular values of A, sorted so that S(i)
			/// &gt;= S(i+1).</param>
			/// <param name="U">array of dimension ldu * m with ldu is not less than max(1,m). U contains
			/// the m×m unitary matrix U.</param>
			/// <param name="ldu">leading dimension of two-dimensional array used to store matrix U.</param>
			/// <param name="VT">array of dimension ldvt * n with ldvt is not less than max(1,n). VT
			/// contains the n×n unitary matrix V**T.</param>
			/// <param name="ldvt">leading dimension of two-dimensional array used to store matrix Vt.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			/// <param name="rwork"></param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the ith
			/// parameter is wrong. if devInfo &gt; 0, devInfo indicates how many superdiagonals of an intermediate
			/// bidiagonal form B did not converge to zero.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDgesvd(cusolverDnHandle handle, [MarshalAs(UnmanagedType.I1)] char jobu, [MarshalAs(UnmanagedType.I1)] char jobvt, int m, int n, CUdeviceptr A, int lda, CUdeviceptr S, CUdeviceptr U, int ldu, CUdeviceptr VT, int ldvt, CUdeviceptr Work, int Lwork, CUdeviceptr rwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="jobu">specifies options for computing all or part of the matrix U: = 'A': all m columns of
			/// U are returned in array U: = 'S': the first min(m,n) columns of U (the left singular
			/// vectors) are returned in the array U; = 'O': the first min(m,n) columns of U (the left singular vectors) are overwritten on
			/// the array A; = 'N': no columns of U (no left singular vectors) are computed.</param>
			/// <param name="jobvt">specifies options for computing all or part of the matrix V**T: = 'A': all N rows
			/// of V**T are returned in the array VT; = 'S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the
			/// array VT; = 'O': the first min(m,n) rows of V**T (the right singular vectors) are overwritten on the array A; = 'N': no rows
			/// of V**T (no right singular vectors) are computed.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m). On exit,
			/// the contents of A are destroyed.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="S">array of dimension min(m,n). The singular values of A, sorted so that S(i)
			/// &gt;= S(i+1).</param>
			/// <param name="U">array of dimension ldu * m with ldu is not less than max(1,m). U contains
			/// the m×m unitary matrix U.</param>
			/// <param name="ldu">leading dimension of two-dimensional array used to store matrix U.</param>
			/// <param name="VT">array of dimension ldvt * n with ldvt is not less than max(1,n). VT
			/// contains the n×n unitary matrix V**T.</param>
			/// <param name="ldvt">leading dimension of two-dimensional array used to store matrix Vt.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			/// <param name="rwork"></param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the ith
			/// parameter is wrong. if devInfo &gt; 0, devInfo indicates how many superdiagonals of an intermediate
			/// bidiagonal form B did not converge to zero.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCgesvd(cusolverDnHandle handle, [MarshalAs(UnmanagedType.I1)] char jobu, [MarshalAs(UnmanagedType.I1)] char jobvt, int m, int n, CUdeviceptr A, int lda, CUdeviceptr S, CUdeviceptr U, int ldu, CUdeviceptr VT, int ldvt, CUdeviceptr Work, int Lwork, CUdeviceptr rwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
			/// corresponding the left and/or right singular vectors.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="jobu">specifies options for computing all or part of the matrix U: = 'A': all m columns of
			/// U are returned in array U: = 'S': the first min(m,n) columns of U (the left singular
			/// vectors) are returned in the array U; = 'O': the first min(m,n) columns of U (the left singular vectors) are overwritten on
			/// the array A; = 'N': no columns of U (no left singular vectors) are computed.</param>
			/// <param name="jobvt">specifies options for computing all or part of the matrix V**T: = 'A': all N rows
			/// of V**T are returned in the array VT; = 'S': the first min(m,n) rows of V**T (the right singular vectors) are returned in the
			/// array VT; = 'O': the first min(m,n) rows of V**T (the right singular vectors) are overwritten on the array A; = 'N': no rows
			/// of V**T (no right singular vectors) are computed.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,m). On exit,
			/// the contents of A are destroyed.</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="S">array of dimension min(m,n). The singular values of A, sorted so that S(i)
			/// &gt;= S(i+1).</param>
			/// <param name="U">array of dimension ldu * m with ldu is not less than max(1,m). U contains
			/// the m×m unitary matrix U.</param>
			/// <param name="ldu">leading dimension of two-dimensional array used to store matrix U.</param>
			/// <param name="VT">array of dimension ldvt * n with ldvt is not less than max(1,n). VT
			/// contains the n×n unitary matrix V**T.</param>
			/// <param name="ldvt">leading dimension of two-dimensional array used to store matrix Vt.</param>
			/// <param name="Work">working space, array of size Lwork.</param>
			/// <param name="Lwork">size of Work, returned by gesvd_bufferSize.</param>
			/// <param name="rwork"></param>
			/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the ith
			/// parameter is wrong. if devInfo &gt; 0, devInfo indicates how many superdiagonals of an intermediate
			/// bidiagonal form B did not converge to zero.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZgesvd(cusolverDnHandle handle, [MarshalAs(UnmanagedType.I1)] char jobu, [MarshalAs(UnmanagedType.I1)] char jobvt, int m, int n, CUdeviceptr A, int lda, CUdeviceptr S, CUdeviceptr U, int ldu, CUdeviceptr VT, int ldvt, CUdeviceptr Work, int Lwork, CUdeviceptr rwork, CUdeviceptr devInfo);
			#endregion

			#region LDLT,UDUT factorization
			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working space work.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
			/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSsytrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr ipiv, CUdeviceptr work, int lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working space work.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
			/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDsytrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr ipiv, CUdeviceptr work, int lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working space work.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
			/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCsytrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr ipiv, CUdeviceptr work, int lwork, CUdeviceptr devInfo);

			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
			/// <param name="work">working space, array of size lwork.</param>
			/// <param name="lwork">size of working space work.</param>
			/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
			/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZsytrf(cusolverDnHandle handle, FillMode uplo, int n, CUdeviceptr A, int lda, CUdeviceptr ipiv, CUdeviceptr work, int lwork, CUdeviceptr devInfo);
			#endregion

			#region SYTRF factorization workspace query
			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working space work.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnSsytrf_bufferSize(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working space work.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnDsytrf_bufferSize(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working space work.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnCsytrf_bufferSize(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, ref int Lwork);

			/// <summary>
			/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
			/// </summary>
			/// <param name="handle">handle to the cuSolverDN library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
			/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
			/// <param name="Lwork">size of working space work.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverDnZsytrf_bufferSize(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, ref int Lwork);
			#endregion
		}

		/// <summary>
		/// The cuSolverSP library was mainly designed to a solve sparse linear system AxB and the least-squares problem
		/// x = argmin||A*z-b||
		/// </summary>
		public static class Sparse
		{
			#region Init
			/// <summary>
			/// This function initializes the cuSolverSP library and creates a handle on the cuSolver
			/// context. It must be called before any other cuSolverSP API function is invoked. It
			/// allocates hardware resources necessary for accessing the GPU.
			/// </summary>
			/// <param name="handle">the pointer to the handle to the cuSolverSP context.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCreate(ref cusolverSpHandle handle);

			/// <summary>
			/// This function releases CPU-side resources used by the cuSolverSP library.
			/// </summary>
			/// <param name="handle">the handle to the cuSolverSP context.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDestroy(cusolverSpHandle handle);

			/// <summary>
			/// This function sets the stream to be used by the cuSolverSP library to execute its routines.
			/// </summary>
			/// <param name="handle">the handle to the cuSolverSP context.</param>
			/// <param name="streamId">the stream to be used by the library.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpSetStream(cusolverSpHandle handle, CUstream streamId);

			/// <summary>
			/// This function gets the stream to be used by the cuSolverSP library to execute its routines.
			/// </summary>
			/// <param name="handle">the handle to the cuSolverSP context.</param>
			/// <param name="streamId">the stream to be used by the library.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpGetStream(cusolverSpHandle handle, ref CUstream streamId);
			#endregion

			/// <summary>
			/// This function checks if A has symmetric pattern or not. The output parameter issym
			/// reports 1 if A is symmetric; otherwise, it reports 0.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix
			/// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrRowPtrA">integer array of m elements that contains the start of every row.</param>
			/// <param name="csrEndPtrA">integer array of m elements that contains the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of matrix A.</param>
			/// <param name="issym">1 if A is symmetric; 0 otherwise.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpXcsrissymHost(cusolverSpHandle handle, int m, int nnzA, cusparseMatDescr descrA, int[] csrRowPtrA, int[] csrEndPtrA, int[] csrColIndA, ref int issym);

			#region linear solver based on LU factorization
			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzA (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size n.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no ordering if reorder=0. Otherwise, symrcm is used to reduce zero fill-in.</param>
			/// <param name="x">solution vector of size n, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrlsvluHost(cusolverSpHandle handle, int n, int nnzA, cusparseMatDescr descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float[] b, float tol, int reorder, float[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzA (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size n.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no ordering if reorder=0. Otherwise, symrcm is used to reduce zero fill-in.</param>
			/// <param name="x">solution vector of size n, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrlsvluHost(cusolverSpHandle handle, int n, int nnzA, cusparseMatDescr descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double[] b, double tol, int reorder, double[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzA (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size n.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no ordering if reorder=0. Otherwise, symrcm is used to reduce zero fill-in.</param>
			/// <param name="x">solution vector of size n, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrlsvluHost(cusolverSpHandle handle, int n, int nnzA, cusparseMatDescr descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex[] b, float tol, int reorder, cuFloatComplex[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzA (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size n.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no ordering if reorder=0. Otherwise, symrcm is used to reduce zero fill-in.</param>
			/// <param name="x">solution vector of size n, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrlsvluHost(cusolverSpHandle handle, int n, int nnzA, cusparseMatDescr descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex[] b, double tol, int reorder, cuDoubleComplex[] x, ref int singularity);

			#endregion

			#region linear solver based on QR factorization
			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrlsvqr(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, float tol, int reorder, CUdeviceptr x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrlsvqr(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, double tol, int reorder, CUdeviceptr x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrlsvqr(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, float tol, int reorder, CUdeviceptr x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrlsvqr(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, double tol, int reorder, CUdeviceptr x, ref int singularity);


			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrlsvqrHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float[] b, float tol, int reorder, float[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrlsvqrHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double[] b, double tol, int reorder, double[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrlsvqrHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex[] b, float tol, int reorder, cuFloatComplex[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide if singular or not.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is invertible. Otherwise, first index j such that R(j,j)≈0</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrlsvqrHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex[] b, double tol, int reorder, cuDoubleComplex[] x, ref int singularity);

			#endregion

			#region linear solver based on Cholesky factorization
			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrlsvcholHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, float[] csrVal, int[] csrRowPtr, int[] csrColInd, float[] b, float tol, int reorder, float[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrlsvcholHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, double[] csrVal, int[] csrRowPtr, int[] csrColInd, double[] b, double tol, int reorder, double[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrlsvcholHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuFloatComplex[] csrVal, int[] csrRowPtr, int[] csrColInd, cuFloatComplex[] b, float tol, int reorder, cuFloatComplex[] x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrlsvcholHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuDoubleComplex[] csrVal, int[] csrRowPtr, int[] csrColInd, cuDoubleComplex[] b, double tol, int reorder, cuDoubleComplex[] x, ref int singularity);


			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrlsvchol(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr b, float tol, int reorder, CUdeviceptr x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrlsvchol(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr b, double tol, int reorder, CUdeviceptr x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrlsvchol(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr b, float tol, int reorder, CUdeviceptr x, ref int singularity);

			/// <summary>
			/// This function solves the linear system A*x=b
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtr">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide singularity.</param>
			/// <param name="reorder">no effect.</param>
			/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
			/// <param name="singularity">-1 if A is symmetric postive definite.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrlsvchol(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr b, double tol, int reorder, CUdeviceptr x, ref int singularity);

			#endregion

			#region least square solver based on QR factorization
			/// <summary>
			/// This function solves the following least-square problem x = argmin||A*z-b||
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide rank of A.</param>
			/// <param name="rankA">numerical rank of A.</param>
			/// <param name="x">solution vector of size n, x=pinv(A)*b.</param>
			/// <param name="p">a vector of size n, which represents the permuation matrix P satisfying A*P^T=Q*R.</param>
			/// <param name="min_norm">||A*x-b||, x=pinv(A)*b.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrlsqvqrHost(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float[] b, float tol, ref int rankA, float[] x, int[] p, ref float min_norm);

			/// <summary>
			/// This function solves the following least-square problem x = argmin||A*z-b||
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide rank of A.</param>
			/// <param name="rankA">numerical rank of A.</param>
			/// <param name="x">solution vector of size n, x=pinv(A)*b.</param>
			/// <param name="p">a vector of size n, which represents the permuation matrix P satisfying A*P^T=Q*R.</param>
			/// <param name="min_norm">||A*x-b||, x=pinv(A)*b.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrlsqvqrHost(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double[] b, double tol, ref int rankA, double[] x, int[] p, ref double min_norm);

			/// <summary>
			/// This function solves the following least-square problem x = argmin||A*z-b||
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide rank of A.</param>
			/// <param name="rankA">numerical rank of A.</param>
			/// <param name="x">solution vector of size n, x=pinv(A)*b.</param>
			/// <param name="p">a vector of size n, which represents the permuation matrix P satisfying A*P^T=Q*R.</param>
			/// <param name="min_norm">||A*x-b||, x=pinv(A)*b.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrlsqvqrHost(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex[] b, float tol, ref int rankA, cuFloatComplex[] x, int[] p, ref float min_norm);

			/// <summary>
			/// This function solves the following least-square problem x = argmin||A*z-b||
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="b">right hand side vector of size m.</param>
			/// <param name="tol">tolerance to decide rank of A.</param>
			/// <param name="rankA">numerical rank of A.</param>
			/// <param name="x">solution vector of size n, x=pinv(A)*b.</param>
			/// <param name="p">a vector of size n, which represents the permuation matrix P satisfying A*P^T=Q*R.</param>
			/// <param name="min_norm">||A*x-b||, x=pinv(A)*b.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrlsqvqrHost(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex[] b, double tol, ref int rankA, cuDoubleComplex[] x, int[] p, ref double min_norm);

			#endregion

			#region eigenvalue solver based on shift inverse
			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsreigvsiHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float mu0, float[] x0, int maxite, float tol, ref float mu, float[] x);

			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsreigvsiHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double mu0, double[] x0, int maxite, double tol, ref double mu, double[] x);

			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsreigvsiHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex mu0, cuFloatComplex[] x0, int maxite, float tol, ref cuFloatComplex mu, cuFloatComplex[] x);

			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsreigvsiHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex mu0, cuDoubleComplex[] x0, int maxite, double tol, ref cuDoubleComplex mu, cuDoubleComplex[] x);

			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsreigvsi(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, float mu0, CUdeviceptr x0, int maxite, float tol, ref float mu, CUdeviceptr x);

			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsreigvsi(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, double mu0, CUdeviceptr x0, int maxite, double tol, ref double mu, CUdeviceptr x);

			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsreigvsi(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cuFloatComplex mu0, CUdeviceptr x0, int maxite, float tol, ref cuFloatComplex mu, CUdeviceptr x);

			/// <summary>
			/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows and columns of matrix A.</param>
			/// <param name="nnz">number of nonzeros of matrix A.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="mu0">initial guess of eigenvalue.</param>
			/// <param name="x0">initial guess of eigenvector, a vecotr of size m.</param>
			/// <param name="maxite">maximum iterations in shift-inverse method.</param>
			/// <param name="tol">tolerance for convergence.</param>
			/// <param name="mu">approximated eigenvalue nearest mu0 under tolerance.</param>
			/// <param name="x">approximated eigenvector of size m.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsreigvsi(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cuDoubleComplex mu0, CUdeviceptr x0, int maxite, double tol, ref cuDoubleComplex mu, CUdeviceptr x);

			#endregion

			#region enclosed eigenvalues
			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsreigsHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex left_bottom_corner, cuFloatComplex right_upper_corner, ref int num_eigs);

			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsreigsHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex left_bottom_corner, cuDoubleComplex right_upper_corner, ref int num_eigs);

			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsreigsHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex left_bottom_corner, cuFloatComplex right_upper_corner, ref int num_eigs);

			/// <summary/>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsreigsHost(cusolverSpHandle handle, int m, int nnz, cusparseMatDescr descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex left_bottom_corner, cuDoubleComplex right_upper_corner, ref int num_eigs);

			#endregion

			#region CPU symrcm
			/// <summary>
			/// This function implements Symmetric Reverse Cuthill-McKee permutation. It returns a
			/// permutation vector p such that A(p,p) would concentrate nonzeros to diagonal. This is
			/// equivalent to symrcm in MATLAB, however the result may not be the same because of
			/// different heuristics in the pseudoperipheral finder.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="p">permutation vector of size n.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpXcsrsymrcmHost(cusolverSpHandle handle, int n, int nnzA, cusparseMatDescr descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p);
			
			#endregion
			
			#region CPU symmdq
			/// <summary>
			/// Symmetric minimum degree algorithm based on quotient graph.<para/>
			/// This function implements Symmetric Minimum Degree Algorithm based on Quotient
			/// Graph. It returns a permutation vector p such that A(p,p) would have less zero fill-in
			/// during Cholesky factorization.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="p">permutation vector of size n.</param>
			/// <returns></returns>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpXcsrsymmdqHost(
				cusolverSpHandle handle,
				int n,
				int nnzA,
				cusparseMatDescr descrA,
				int[] csrRowPtrA,
				int[] csrColIndA,
				int[] p);

			
			/// <summary>
			/// Symmetric Approximate minimum degree algorithm based on quotient graph.<para/>
			/// This function implements Symmetric Approximate Minimum Degree Algorithm based
			/// on Quotient Graph. It returns a permutation vector p such that A(p,p) would have less
			/// zero fill-in during Cholesky factorization.
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="n">number of rows and columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="p">permutation vector of size n.</param>
			/// <returns></returns>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpXcsrsymamdHost(
				cusolverSpHandle handle,
				int n,
				int nnzA,
				cusparseMatDescr descrA,
				int[] csrRowPtrA,
				int[] csrColIndA,
				int[] p);

			#endregion

			#region CPU permuation
			/// <summary>
			/// Given a left permutation vector p which corresponds to permutation matrix P and a
			/// right permutation vector q which corresponds to permutation matrix Q, this function
			/// computes permutation of matrix A by B = P*A*Q^T
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="p">left permutation vector of size m.</param>
			/// <param name="q">right permutation vector of size n.</param>
			/// <param name="bufferSizeInBytes">number of bytes of the buffer.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpXcsrperm_bufferSizeHost(cusolverSpHandle handle, int m, int n, int nnzA, cusparseMatDescr descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p, int[] q, ref SizeT bufferSizeInBytes);

			/// <summary>
			/// Given a left permutation vector p which corresponds to permutation matrix P and a
			/// right permutation vector q which corresponds to permutation matrix Q, this function
			/// computes permutation of matrix A by B = P*A*Q^T
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of matrix A.</param>
			/// <param name="n">number of columns of matrix A.</param>
			/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
			/// <param name="p">left permutation vector of size m.</param>
			/// <param name="q">right permutation vector of size n.</param>
			/// <param name="map">integer array of nnzA indices. If the user wants to
			/// get relationship between A and B, map must be set 0:1:(nnzA-1).</param>
			/// <param name="pBuffer">buffer allocated by the user, the size is returned by csrperm_bufferSize().</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpXcsrpermHost(cusolverSpHandle handle, int m, int n, int nnzA, cusparseMatDescr descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p, int[] q, int[] map, byte[] pBuffer);

			
			#endregion

			#region Low-level API: Batched QR
			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="info">opaque structure for QR factorization.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCreateCsrqrInfo(ref csrqrInfo info);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="info">opaque structure for QR factorization.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDestroyCsrqrInfo(csrqrInfo info);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnzA">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrRowPtrA">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpXcsrqrAnalysisBatched(cusolverSpHandle handle, int m, int n, int nnzA, cusparseMatDescr descrA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, csrqrInfo info);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtr">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="internalDataInBytes">number of bytes of the internal data.</param>
			/// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtr">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="internalDataInBytes">number of bytes of the internal data.</param>
			/// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtr">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="internalDataInBytes">number of bytes of the internal data.</param>
			/// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrVal">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtr">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="internalDataInBytes">number of bytes of the internal data.</param>
			/// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtrA">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
			/// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="pBuffer">buffer allocated by the user, the size is returned
			/// by cusolverSpXcsrqrBufferInfoBatched().</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpScsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtrA">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
			/// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="pBuffer">buffer allocated by the user, the size is returned
			/// by cusolverSpXcsrqrBufferInfoBatched().</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpDcsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtrA">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
			/// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="pBuffer">buffer allocated by the user, the size is returned
			/// by cusolverSpXcsrqrBufferInfoBatched().</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpCcsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);

			/// <summary>
			/// The batched sparse QR factorization is used to solve either a set of least-squares
			/// problems or a set of linear systems
			/// </summary>
			/// <param name="handle">handle to the cuSolverSP library context.</param>
			/// <param name="m">number of rows of each matrix Aj.</param>
			/// <param name="n">number of columns of each matrix Aj.</param>
			/// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
			/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
			/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
			/// <param name="csrValA">array of nnzA*batchSize nonzero 
			/// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
			/// <param name="csrRowPtrA">integer array of m+1 elements that contains the
			/// start of every row and the end of the last row plus one.</param>
			/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
			/// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
			/// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
			/// <param name="batchSize">number of systems to be solved.</param>
			/// <param name="info">opaque structure for QR factorization.</param>
			/// <param name="pBuffer">buffer allocated by the user, the size is returned
			/// by cusolverSpXcsrqrBufferInfoBatched().</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverSpZcsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);
			#endregion

		}

		/// <summary>
		/// The cuSolverRF library was designed to accelerate solution of sets of linear systems by
		/// fast re-factorization when given new coefficients in the same sparsity pattern
		/// A_i x_i = f_i
		/// </summary>
		public static class Refactorization
		{
			#region Init
			/// <summary>
			/// This routine initializes the cuSolverRF library. It allocates required resources and must be called prior to any other cuSolverRF library routine.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfCreate(ref cusolverRfHandle handle);
			/// <summary>
			/// This routine shuts down the cuSolverRF library. It releases acquired resources and must be called after all the cuSolverRF library routines.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfDestroy(cusolverRfHandle handle);

			/// <summary>
			/// This routine gets the matrix format used in the cusolverRfSetup(),
			/// cusolverRfSetupHost(), cusolverRfResetValues(), cusolverRfExtractBundledFactorsHost() and cusolverRfExtractSplitFactorsHost() routines.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="format">the enumerated matrix format type.</param>
			/// <param name="diag">the enumerated unit diagonal type.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfGetMatrixFormat(cusolverRfHandle handle, ref MatrixFormat format, ref UnitDiagonal diag);

			/// <summary>
			/// This routine sets the matrix format used in the cusolverRfSetup(),
			/// cusolverRfSetupHost(), cusolverRfResetValues(), cusolverRfExtractBundledFactorsHost() and cusolverRfExtractSplitFactorsHost() routines.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="format">the enumerated matrix format type.</param>
			/// <param name="diag">the enumerated unit diagonal type.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfSetMatrixFormat(cusolverRfHandle handle, MatrixFormat format, UnitDiagonal diag);

			/// <summary>
			/// This routine sets the numeric values used for checking for "zero" pivot and for boosting
			/// it in the cusolverRfRefactor() and cusolverRfSolve() routines. It may be called 
			/// multiple times prior to cusolverRfRefactor() and cusolverRfSolve() routines.
			/// The numeric boosting will be used only if boost &gt; 0.0.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="zero">the value below which zero pivot is flagged.</param>
			/// <param name="boost">the value which is substituted for zero pivot (if the later is flagged).</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfSetNumericProperties(cusolverRfHandle handle, double zero, double boost);

			/// <summary>
			/// This routine gets the numeric values used for checking for "zero" pivot and for boosting
			/// it in the cusolverRfRefactor() and cusolverRfSolve() routines. It may be called 
			/// multiple times prior to cusolverRfRefactor() and cusolverRfSolve() routines.
			/// The numeric boosting will be used only if boost &gt; 0.0.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="zero">the value below which zero pivot is flagged.</param>
			/// <param name="boost">the value which is substituted for zero pivot (if the later is flagged).</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfGetNumericProperties(cusolverRfHandle handle, ref double zero, ref double boost);

			/// <summary>
			/// This routine gets the report whether numeric boosting was used in the
			/// cusolverRfRefactor() and cusolverRfSolve() routines.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="report">the enumerated boosting report type.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfGetNumericBoostReport(cusolverRfHandle handle, ref NumericBoostReport report);

			/// <summary>
			/// This routine sets the algorithm used for the refactorization in cusolverRfRefactor()
			/// and the triangular solve in cusolverRfSolve(). It may be called once prior to
			/// cusolverRfAnalyze() routine.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="factAlg">the enumerated algorithm type.</param>
			/// <param name="solveAlg">the enumerated algorithm type.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfSetAlgs(cusolverRfHandle handle, Factorization factAlg, TriangularSolve solveAlg);

			/// <summary>
			/// This routine gets the algorithm used for the refactorization in cusolverRfRefactor()
			/// and the triangular solve in cusolverRfSolve(). It may be called once prior to
			/// cusolverRfAnalyze() routine.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="factAlg">the enumerated algorithm type.</param>
			/// <param name="solveAlg">the enumerated algorithm type.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfGetAlgs(cusolverRfHandle handle, ref Factorization factAlg, ref TriangularSolve solveAlg);

			/// <summary>
			/// This routine gets the mode used in the cusolverRfResetValues routine.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="fastMode">the enumerated mode type.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfGetResetValuesFastMode(cusolverRfHandle handle, ref ResetValuesFastMode fastMode);

			/// <summary>
			/// This routine sets the mode used in the cusolverRfResetValues routine.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="fastMode">the enumerated mode type.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfSetResetValuesFastMode(cusolverRfHandle handle, ResetValuesFastMode fastMode);
			#endregion



			#region Non-Batched Routines

			#region setup of internal structures from host or device memory


			/// <summary>This routine assembles the internal data structures of the cuSolverRF library. It is often
			/// the first routine to be called after the call to the cusolverRfCreate() routine.
			/// </summary>
			/// <param name="n">the number of rows (and columns) of matrix A.</param>
			/// <param name="nnzA">the number of non-zero elements of matrix A.</param>
			/// <param name="h_csrRowPtrA">the array of offsets corresponding to 
			/// the start of each row in the arrays h_csrColIndA and h_csrValA. This
			/// array has also an extra entry at the end that stores the number of non-zero
			/// elements in the matrix. The array size is n+1.</param>
			/// <param name="h_csrColIndA">the array of column indices corresponding
			/// to the non-zero elements in the matrix. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzA.</param>
			/// <param name="h_csrValA">the array of values corresponding to the
			/// non-zero elements in the matrix. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzA.</param>
			/// <param name="nnzL">the number of non-zero elements of matrix L.</param>
			/// <param name="h_csrRowPtrL">the array of offsets corresponding to
			/// the start of each row in the arrays h_csrColIndL and h_csrValL. This
			/// array has also an extra entry at the end that stores the number of non-zero
			/// elements in the matrix L. The array size is n+1.</param>
			/// <param name="h_csrColIndL">the array of column indices corresponding
			/// to the non-zero elements in the matrix L. It is assumed that this array is sorted by
			/// row and by column within each row. The array size is nnzL.</param>
			/// <param name="h_csrValL">the array of values corresponding to the
			/// non-zero elements in the matrix L. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzL.</param>
			/// <param name="nnzU">the number of non-zero elements of matrix U.</param>
			/// <param name="h_csrRowPtrU">the array of offsets corresponding to
			/// the start of each row in the arrays h_csrColIndU and h_csrValU. This
			/// array has also an extra entry at the end that stores the number of non-zero elements in the matrix U. The array size is n+1.</param>
			/// <param name="h_csrColIndU">the array of column indices corresponding 
			/// to the non-zero elements in the matrix U. It is assumed that this array is sorted by row and by column within each row. The array size is nnzU.</param>
			/// <param name="h_csrValU">the array of values corresponding to the non-zero elements in the matrix U. It is
			/// assumed that this array is sorted by row and by column within each row. The array size is nnzU.</param>
			/// <param name="h_P">the left permutation (often associated with pivoting). The array size in n.</param>
			/// <param name="h_Q">the right permutation (often associated with reordering). The array size in n.</param>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfSetupHost(int n, int nnzA, int[] h_csrRowPtrA, int[] h_csrColIndA, double[] h_csrValA, int nnzL, int[] h_csrRowPtrL, int[] h_csrColIndL, double[] h_csrValL, int nnzU, int[] h_csrRowPtrU, int[] h_csrColIndU, double[] h_csrValU, int[] h_P, int[] h_Q, cusolverRfHandle handle);

			/// <summary>This routine assembles the internal data structures of the cuSolverRF library. It is often
			/// the first routine to be called after the call to the cusolverRfCreate() routine.
			/// </summary>
			/// <param name="n">the number of rows (and columns) of matrix A.</param>
			/// <param name="nnzA">the number of non-zero elements of matrix A.</param>
			/// <param name="csrRowPtrA">the array of offsets corresponding to 
			/// the start of each row in the arrays h_csrColIndA and h_csrValA. This
			/// array has also an extra entry at the end that stores the number of non-zero
			/// elements in the matrix. The array size is n+1.</param>
			/// <param name="csrColIndA">the array of column indices corresponding
			/// to the non-zero elements in the matrix. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzA.</param>
			/// <param name="csrValA">the array of values corresponding to the
			/// non-zero elements in the matrix. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzA.</param>
			/// <param name="nnzL">the number of non-zero elements of matrix L.</param>
			/// <param name="csrRowPtrL">the array of offsets corresponding to
			/// the start of each row in the arrays h_csrColIndL and h_csrValL. This
			/// array has also an extra entry at the end that stores the number of non-zero
			/// elements in the matrix L. The array size is n+1.</param>
			/// <param name="csrColIndL">the array of column indices corresponding
			/// to the non-zero elements in the matrix L. It is assumed that this array is sorted by
			/// row and by column within each row. The array size is nnzL.</param>
			/// <param name="csrValL">the array of values corresponding to the
			/// non-zero elements in the matrix L. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzL.</param>
			/// <param name="nnzU">the number of non-zero elements of matrix U.</param>
			/// <param name="csrRowPtrU">the array of offsets corresponding to
			/// the start of each row in the arrays h_csrColIndU and h_csrValU. This
			/// array has also an extra entry at the end that stores the number of non-zero elements in the matrix U. The array size is n+1.</param>
			/// <param name="csrColIndU">the array of column indices corresponding 
			/// to the non-zero elements in the matrix U. It is assumed that this array is sorted by row and by column within each row. The array size is nnzU.</param>
			/// <param name="csrValU">the array of values corresponding to the non-zero elements in the matrix U. It is
			/// assumed that this array is sorted by row and by column within each row. The array size is nnzU.</param>
			/// <param name="P">the left permutation (often associated with pivoting). The array size in n.</param>
			/// <param name="Q">the right permutation (often associated with reordering). The array size in n.</param>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfSetupDevice(int n, int nnzA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr csrValA, int nnzL,
				CUdeviceptr csrRowPtrL, CUdeviceptr csrColIndL, CUdeviceptr csrValL, int nnzU, CUdeviceptr csrRowPtrU, CUdeviceptr csrColIndU, CUdeviceptr csrValU,
				CUdeviceptr P, CUdeviceptr Q, cusolverRfHandle handle);

			#endregion

			/// <summary>
			/// This routine updates internal data structures with the values of the new coefficient
			/// matrix. It is assumed that the arrays csrRowPtrA, csrColIndA, P and Q have not
			/// changed since the last call to the cusolverRfSetup[Host] routine. This assumption
			/// reflects the fact that the sparsity pattern of coefficient matrices as well as reordering to
			/// minimize fill-in and pivoting remain the same in the set of linear systems
			/// </summary>
			/// <param name="n">the number of rows (and columns) of matrix A.</param>
			/// <param name="nnzA">the number of non-zero elements of matrix A.</param>
			/// <param name="csrRowPtrA">the array of offsets corresponding to the start of each row in the arrays
			/// csrColIndA and csrValA. This array has also an extra entry at the end that stores the number of non-zero elements in the
			/// matrix. The array size is n+1.</param>
			/// <param name="csrColIndA">the array of column indices corresponding to the non-zero elements in the matrix. It
			/// is assumed that this array is sorted by row and by column within each row. The array size is nnzA.</param>
			/// <param name="csrValA">the array of values corresponding to the non-zero elements in the matrix. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzA.</param>
			/// <param name="P">the left permutation (often associated with pivoting). The array size in n.</param>
			/// <param name="Q">the right permutation (often associated with reordering). The array size in n.</param>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfResetValues(int n, int nnzA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr csrValA,
				CUdeviceptr P, CUdeviceptr Q, cusolverRfHandle handle);


			/// <summary>
			/// This routine performs the appropriate analysis of parallelism available in the LU refactorization depending upon the algorithm chosen by the user.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfAnalyze(cusolverRfHandle handle);

			/// <summary>
			/// This routine performs the LU re-factorization
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfRefactor(cusolverRfHandle handle);


			/// <summary>
			/// This routine allows direct access to the lower L and upper U triangular factors stored in
			/// the cuSolverRF library handle. The factors are compressed into a single matrix M=(LI)+
			/// U, where the unitary diagonal of L is not stored. It is assumed that a prior call to the
			/// cusolverRfRefactor() was done in order to generate these triangular factors.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="nnzM">the number of non-zero elements of matrix M.</param>
			/// <param name="Mp">the array of offsets corresponding to the start of each row in the arrays Mi and Mx.
			/// This array has also an extra entry at the end that stores the number of non-zero elements in the matrix $M$. The array size is n+1.</param>
			/// <param name="Mi">the array of column indices corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
			/// <param name="Mx">the array of values corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfAccessBundledFactorsDevice(cusolverRfHandle handle, ref int nnzM, ref CUdeviceptr Mp, ref CUdeviceptr Mi, ref CUdeviceptr Mx);

			/// <summary>
			/// This routine allows direct access to the lower L and upper U triangular factors stored in
			/// the cuSolverRF library handle. The factors are compressed into a single matrix M=(LI)+
			/// U, where the unitary diagonal of L is not stored. It is assumed that a prior call to the
			/// cusolverRfRefactor() was done in order to generate these triangular factors.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="h_nnzM">the number of non-zero elements of matrix M.</param>
			/// <param name="h_Mp">the array of offsets corresponding to the start of each row in the arrays Mi and Mx.
			/// This array has also an extra entry at the end that stores the number of non-zero elements in the matrix $M$. The array size is n+1.</param>
			/// <param name="h_Mi">the array of column indices corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
			/// <param name="h_Mx">the array of values corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfExtractBundledFactorsHost(cusolverRfHandle handle, ref int h_nnzM, ref IntPtr h_Mp, ref IntPtr h_Mi, ref IntPtr h_Mx);

			/// <summary>
			/// This routine extracts lower (L) and upper (U) triangular factors from the
			/// cuSolverRF library handle into the host memory. It is assumed that a prior call to the
			/// cusolverRfRefactor() was done in order to generate these triangular factors.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="h_nnzL">the number of non-zero elements of matrix L.</param>
			/// <param name="h_csrRowPtrL">the array of offsets corresponding to the start of each row in the arrays h_Li and
			/// h_Lx. This array has also an extra entry at the end that stores the number of nonzero elements in the matrix L. The array size is n+1.</param>
			/// <param name="h_csrColIndL">the array of column indices corresponding to the non-zero elements in the matrix L. It is assumed that this array is sorted by
			/// row and by column within each row. The array size is h_nnzL.</param>
			/// <param name="h_csrValL">the array of values corresponding to the non-zero elements in the matrix L. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is h_nnzL.</param>
			/// <param name="h_nnzU">the number of non-zero elements of matrix U.</param>
			/// <param name="h_csrRowPtrU">the array of offsets corresponding to the start of each row in the arrays h_Ui and h_Ux. This array has also an extra entry
			/// at the end that stores the number of nonzero elements in the matrix U. The array size is n+1.</param>
			/// <param name="h_csrColIndU">the array of column indices corresponding to the non-zero elements in the matrix U. It is assumed that this array is sorted by
			/// row and by column within each row. The array size is h_nnzU.</param>
			/// <param name="h_csrValU">the array of values corresponding to the non-zero elements in the matrix U. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is h_nnzU.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfExtractSplitFactorsHost(cusolverRfHandle handle, ref int h_nnzL, ref IntPtr h_csrRowPtrL, ref IntPtr h_csrColIndL, ref IntPtr h_csrValL, ref int h_nnzU, ref IntPtr h_csrRowPtrU, ref IntPtr h_csrColIndU, ref IntPtr h_csrValU);

			/// <summary>
			/// This routine performs the forward and backward solve with the lower and upper
			/// triangular factors resulting from the LU re-factorization
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="P">the left permutation (often associated with pivoting). The array size in n.</param>
			/// <param name="Q">the right permutation (often associated with reordering). The array size in n.</param>
			/// <param name="nrhs">the number right-hand-sides to be solved.</param>
			/// <param name="Temp">the dense matrix that contains temporary workspace (of size ldt*nrhs).</param>
			/// <param name="ldt">the leading dimension of dense matrix Temp (ldt &gt;= n).</param>
			/// <param name="XF">the dense matrix that contains the righthand-sides F and solutions X (of size ldxf*nrhs).</param>
			/// <param name="ldxf">the leading dimension of dense matrix XF (ldxf &gt;= n).</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfSolve(cusolverRfHandle handle, CUdeviceptr P, CUdeviceptr Q, int nrhs, double[] Temp, int ldt, double[] XF, int ldxf);

			#endregion

			#region Batched Routines


			/// <summary>
			/// This routine assembles the internal data structures of the cuSolverRF library for batched
			/// operation. It is called after the call to the cusolverRfCreate() routine, and before any
			/// other batched routines.
			/// </summary>
			/// <param name="batchSize">the number of matrices in the batched mode.</param>
			/// <param name="n">the number of rows (and columns) of matrix A.</param>
			/// <param name="nnzA">the number of non-zero elements of matrix A.</param>
			/// <param name="h_csrRowPtrA">the array of offsets corresponding to 
			/// the start of each row in the arrays h_csrColIndA and h_csrValA. This array has also an extra entry at the
			/// end that stores the number of non-zero elements in the matrix. The array size is n+1.</param>
			/// <param name="h_csrColIndA">the array of column indices corresponding 
			/// to the non-zero elements in the matrix. It is assumed that this array is sorted by row and by column within each row. The array size is nnzA.</param>
			/// <param name="h_csrValA_array">array of pointers of size batchSize, each pointer points to the array of values corresponding to the non-zero elements in the matrix.</param>
			/// <param name="nnzL">the number of non-zero elements of matrix L.</param>
			/// <param name="h_csrRowPtrL">the array of offsets corresponding to the start of each row in the arrays h_csrColIndL and h_csrValL. This
			/// array has also an extra entry at the end that stores the number of non-zero elements in the matrix L. The array size is n+1.</param>
			/// <param name="h_csrColIndL">the array of column indices corresponding to the non-zero elements in the matrix L. It is assumed that this array is sorted by
			/// row and by column within each row. The array size is nnzL.</param>
			/// <param name="h_csrValL">the array of values corresponding to the non-zero elements in the matrix L. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzL.</param>
			/// <param name="nnzU">the number of non-zero elements of matrix U.</param>
			/// <param name="h_csrRowPtrU">the array of offsets corresponding to the start of each row in the arrays h_csrColIndU and h_csrValU. This
			/// array has also an extra entry at the end that stores the number of non-zero elements in the matrix U. The array size is n+1.</param>
			/// <param name="h_csrColIndU">the array of column indices corresponding to the non-zero elements in the matrix U. It is assumed that this array is sorted by
			/// row and by column within each row. The array size is nnzU.</param> 
			/// <param name="h_csrValU">the array of values corresponding to the non-zero elements in the matrix U. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzU.</param>
			/// <param name="h_P">the left permutation (often associated with pivoting). The array size in n.</param>
			/// <param name="h_Q">the right permutation (often associated with reordering). The array size in n.</param>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfBatchSetupHost(int batchSize ,int n, int nnzA, int[] h_csrRowPtrA, int[] h_csrColIndA, IntPtr[] h_csrValA_array, int nnzL, 
				int[] h_csrRowPtrL, int[] h_csrColIndL, double[] h_csrValL, int nnzU, int[] h_csrRowPtrU, int[] h_csrColIndU, double[] h_csrValU, int[] h_P, int[] h_Q, cusolverRfHandle handle);

			/// <summary>
			/// This routine updates internal data structures with the values of the new coefficient
			/// matrix. It is assumed that the arrays csrRowPtrA, csrColIndA, P and Q have not 
			/// changed since the last call to the cusolverRfbatch_setup_host routine.
			/// </summary>
			/// <param name="batchSize">the number of matrices in batched mode.</param>
			/// <param name="n">the number of rows (and columns) of matrix A.</param>
			/// <param name="nnzA">the number of non-zero elements of matrix A.</param>
			/// <param name="csrRowPtrA">the array of offsets corresponding to the start of each row in the arrays csrColIndA and csrValA. 
			/// This array has also an extra entry at the end that stores the number of non-zero elements in the matrix. The array size is n+1.</param>
			/// <param name="csrColIndA">the array of column indices corresponding to the non-zero elements in the matrix. It is assumed that this array is sorted by row
			/// and by column within each row. The array size is nnzA.</param>
			/// <param name="csrValA_array">array of pointers of size batchSize, each pointer points to the array of values corresponding to the non-zero elements in the matrix.</param>
			/// <param name="P">the left permutation (often associated with pivoting). The array size in n.</param>
			/// <param name="Q">the right permutation (often associated with reordering). The array size in n.</param>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfBatchResetValues(int batchSize, int n, int nnzA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr[] csrValA_array, 
				CUdeviceptr P, CUdeviceptr Q, cusolverRfHandle handle);
 
			/// <summary>
			/// This routine performs the appropriate analysis of parallelism available in the batched LU re-factorization.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfBatchAnalyze(cusolverRfHandle handle);

			/// <summary>
			/// This routine performs the LU re-factorization
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfBatchRefactor(cusolverRfHandle handle);

			/// <summary>
			/// To solve A_j * x_j = b_j, first we reform the equation by M_j * Q * x_j = P * b_j. Then do refactorization by
			/// cusolverRfBatch_Refactor(). Further cusolverRfBatch_Solve() takes over the remaining steps.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="P">the left permutation (often associated with pivoting). The array size in n.</param>
			/// <param name="Q">the right permutation (often associated with reordering). The array size in n.</param>
			/// <param name="nrhs">the number right-hand-sides to be solved.</param>
			/// <param name="Temp">the dense matrix that contains temporary workspace (of size ldt*nrhs).</param>
			/// <param name="ldt">the leading dimension of dense matrix Temp (ldt &gt;= n).</param>
			/// <param name="XF_array">array of pointers of size batchSize, each pointer points to the dense matrix that contains the right-hand-sides F and solutions X (of size ldxf*nrhs).</param>
			/// <param name="ldxf">the leading dimension of dense matrix XF (ldxf &gt;= n).</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfBatchSolve(cusolverRfHandle handle, CUdeviceptr P, CUdeviceptr Q, int nrhs, double[] Temp, 
				int ldt, IntPtr[] XF_array, int ldxf);

			/// <summary>
			/// The user can query which matrix failed LU refactorization by checking
			/// corresponding value in position array. The input parameter position is an integer array of size batchSize.
			/// </summary>
			/// <param name="handle">the pointer to the cuSolverRF library handle.</param>
			/// <param name="position">integer array of size batchSize. The value of position(j) reports singularity
			/// of matrix Aj, -1 if no structural / numerical zero, k &gt;= 0 if Aj(k,k) is either structural zero or numerical zero.</param>
			[DllImport(CUSOLVE_API_DLL_NAME)]
			public static extern cusolverStatus cusolverRfBatchZeroPivot(cusolverRfHandle handle, int[] position);

			#endregion

		}
	}
}
