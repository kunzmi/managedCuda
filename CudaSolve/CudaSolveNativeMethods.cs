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
using System.Runtime.CompilerServices;
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
        internal const string CUSOLVE_API_DLL_NAME = "cusolver64_11.dll";

#if (NETCOREAPP)
        internal const string CUSOLVE_API_DLL_NAME_LINUX = "cusolver";

        static CudaSolveNativeMethods()
        {
            NativeLibrary.SetDllImportResolver(typeof(CudaSolveNativeMethods).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            IntPtr libHandle = IntPtr.Zero;

            if (libraryName == CUSOLVE_API_DLL_NAME)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    NativeLibrary.TryLoad(CUSOLVE_API_DLL_NAME_LINUX, assembly, DllImportSearchPath.SafeDirectories, out libHandle);
                }
            }
            //On Windows, use the default library name
            return libHandle;
        }

        [MethodImpl(MethodImplOptions.NoOptimization)]
        internal static void Init()
        {
            //Need that to have the constructor called before any library call.
        }
#endif
        /// <summary>
        /// The cuSolverDN library was designed to solve dense linear systems of the form Ax=B
        /// </summary>
        public static class Dense
        {
#if (NETCOREAPP)
            static Dense()
            {
                CudaSolveNativeMethods.Init();
            }
#endif
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
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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



            /// <summary>
            /// This function computes the QR factorization of a m×n matrix A=Q*R
            /// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">indicates if matrix Q is on the left or right of C.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix A.</param>
            /// <param name="n">number of columns of matrix A.</param>
            /// <param name="k">number of elementary relfections.</param>
            /// <param name="A">array of dimension lda * k with lda is not less than max(1,m).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
            /// <param name="tau">array of dimension at least min(m,n).</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array Workspace.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSormqr_bufferSize(cusolverDnHandle handle, SideMode side, Operation trans,
                int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, ref int lwork);

            /// <summary>
            /// This function computes the QR factorization of a m×n matrix A=Q*R
            /// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">indicates if matrix Q is on the left or right of C.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix A.</param>
            /// <param name="n">number of columns of matrix A.</param>
            /// <param name="k">number of elementary relfections.</param>
            /// <param name="A">array of dimension lda * k with lda is not less than max(1,m).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
            /// <param name="tau">array of dimension at least min(m,n).</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array Workspace.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDormqr_bufferSize(cusolverDnHandle handle, SideMode side, Operation trans,
                int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, ref int lwork);

            /// <summary>
            /// This function computes the QR factorization of a m×n matrix A=Q*R
            /// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">indicates if matrix Q is on the left or right of C.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix A.</param>
            /// <param name="n">number of columns of matrix A.</param>
            /// <param name="k">number of elementary relfections.</param>
            /// <param name="A">array of dimension lda * k with lda is not less than max(1,m).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
            /// <param name="tau">array of dimension at least min(m,n).</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array Workspace.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCunmqr_bufferSize(cusolverDnHandle handle, SideMode side, Operation trans,
                int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, ref int lwork);

            /// <summary>
            /// This function computes the QR factorization of a m×n matrix A=Q*R
            /// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">indicates if matrix Q is on the left or right of C.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix A.</param>
            /// <param name="n">number of columns of matrix A.</param>
            /// <param name="k">number of elementary relfections.</param>
            /// <param name="A">array of dimension lda * k with lda is not less than max(1,m).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
            /// <param name="tau">array of dimension at least min(m,n).</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C.</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array Workspace.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZunmqr_bufferSize(cusolverDnHandle handle, SideMode side, Operation trans,
                int m, int n, int k, CUdeviceptr A, int lda, CUdeviceptr tau, CUdeviceptr C, int ldc, ref int lwork);
            #endregion


            #region generate unitary matrix Q from QR factorization
            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSorgqr_bufferSize(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDorgqr_bufferSize(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCungqr_bufferSize(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZungqr_bufferSize(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="work">working space, rray of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the orgqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSorgqr(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="work">working space, rray of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the orgqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDorgqr(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="work">working space, rray of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the orgqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCungqr(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generate unitary matrix Q from QR factorization
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="m">number of rows of matrix Q. m >= 0;</param>
            /// <param name="n">number of columns of matrix Q. m >= n >= 0;</param>
            /// <param name="k">number of elementary relfections whose product defines the matrix Q. n >= k >= 0;</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,m).
            /// i-th column of A contains elementary reflection vector.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m).</param>
            /// <param name="tau">array of dimension k. tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="work">working space, rray of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the orgqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZungqr(
                cusolverDnHandle handle,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);
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

            #endregion

            #region tridiagonal factorization
            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsytrd_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr d,
                CUdeviceptr e,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsytrd_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr d,
                CUdeviceptr e,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsytrd(cusolverDnHandle handle, char uplo, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr tau, CUdeviceptr Work, int Lwork, CUdeviceptr info);

            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsytrd(cusolverDnHandle handle, char uplo, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr tau, CUdeviceptr Work, int Lwork, CUdeviceptr info);


            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChetrd_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr d,
                CUdeviceptr e,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhetrd_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr d,
                CUdeviceptr e,
                CUdeviceptr tau,
                ref int lwork);


            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChetrd(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr d,
                CUdeviceptr e,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary/>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhetrd(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr d,
                CUdeviceptr e,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

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



            #region generates one of the unitary matrices Q or P**T determined by GEBRD
            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSorgbr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDorgbr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCungbr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZungbr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            /// <param name="work">working space, array of size lwork.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSorgbr(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            /// <param name="work">working space, array of size lwork.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDorgbr(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            /// <param name="work">working space, array of size lwork.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCungbr(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generates one of the unitary matrices Q or P**T determined by GEBRD
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">if side = CUBLAS_SIDE_LEFT, generate Q. if side = CUBLAS_SIDE_RIGHT, generate P**T.</param>
            /// <param name="m">number of rows of matrix Q or P**T.</param>
            /// <param name="n">if side = CUBLAS_SIDE_LEFT, m>= n>= min(m,k). if side = CUBLAS_SIDE_RIGHT, n>= m>= min(n,k).</param>
            /// <param name="k">if side = CUBLAS_SIDE_LEFT, the number of columns in the original mby-
            /// k matrix reduced by gebrd. if side = CUBLAS_SIDE_RIGHT, the number of rows in the original k-by-n matrix reduced by gebrd.</param>
            /// <param name="A">array of dimension lda * n On entry, the vectors which define the
            /// elementary reflectors, as returned by gebrd. On exit, the m-by-n matrix Q or P**T.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,m);</param>
            /// <param name="tau">array of dimension min(m,k) if side is CUBLAS_SIDE_LEFT; of dimension min(n,k) if side is
            /// CUBLAS_SIDE_RIGHT; tau(i) must contain the scalar factor of the elementary reflector H(i) or G(i), which determines Q
            /// or P**T, as returned by gebrd in its array argument TAUQ or TAUP.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            /// <param name="work">working space, array of size lwork.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZungbr(
                cusolverDnHandle handle,
                SideMode side,
                int m,
                int n,
                int k,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);
            #endregion

            #region generate unitary Q comes from sytrd
            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSorgtr_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDorgtr_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCungtr_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZungtr_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                ref int lwork);

            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="info">if info = 0, the orgtr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSorgtr(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="info">if info = 0, the orgtr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDorgtr(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="info">if info = 0, the orgtr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCungtr(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// generate unitary Q comes from sytrd
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="uplo">uplo = CUBLAS_FILL_MODE_LOWER: Lower triangle of A contains elementary
            /// reflectors from sytrd. uplo = CUBLAS_FILL_MODE_UPPER: Upper triangle of A contains elementary
            /// reflectors from sytrd.</param>
            /// <param name="n">number of rows (columns) of matrix Q.</param>
            /// <param name="A">array of dimension lda * n On entry, matrix A from sytrd contains the elementary reflectors. On exit, matrix A
            /// contains the n-by-n orthogonal matrix Q.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda >= max(1,n).</param>
            /// <param name="tau">array of dimension (n-1) tau(i) is the scalar of i-th elementary reflection vector.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="info">if info = 0, the orgtr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZungtr(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);
            #endregion

            #region compute op(Q)*C or C*op(Q) where Q comes from sytrd
            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSormtr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                ref int lwork);

            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDormtr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                ref int lwork);

            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCunmtr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                ref int lwork);

            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="lwork">size of working array work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZunmtr_bufferSize(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                ref int lwork);

            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSormtr(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDormtr(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCunmtr(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function overwrites m×n matrix C by
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="side">side = CUBLAS_SIDE_LEFT, apply Q or Q**T from the Left; 
            /// side = CUBLAS_SIDE_RIGHT, apply Q or Q**T from the Right.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="trans">operation op(Q) that is non- or (conj.) transpose.</param>
            /// <param name="m">number of rows of matrix C.</param>
            /// <param name="n">number of columns of matrix C.</param>
            /// <param name="A">array of dimension lda * m if side = CUBLAS_SIDE_LEFT; lda * n if
            /// side = CUBLAS_SIDE_RIGHT. The matrix A from sytrd contains the elementary reflectors.</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. if side is
            /// CUBLAS_SIDE_LEFT, lda &gt;= max(1,m); if side is CUBLAS_SIDE_RIGHT, lda &gt;= max(1,n).</param>
            /// <param name="tau">array of dimension (m-1) if side is CUBLAS_SIDE_LEFT; of dimension
            /// (n-1) if side is CUBLAS_SIDE_RIGHT; The vector tau is from sytrd, so tau(i) 
            /// is the scalar of i-th elementary reflection vector.</param>
            /// <param name="C">array of size ldc * n. On exit, C is overwritten by op(Q)*C or C*op(Q).</param>
            /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of working array work.</param>
            /// <param name="info">if info = 0, the ormqr is successful. if info = -i, the i-th parameter is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZunmtr(
                cusolverDnHandle handle,
                SideMode side,
                FillMode uplo,
                Operation trans,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr tau,
                CUdeviceptr C,
                int ldc,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);
            #endregion

            #region standard symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer
            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevd(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevd(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevd(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix A.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevd(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            #endregion

            #region generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer

            /// <summary>
            /// The helper functions below can calculate the sizes needed for pre-allocated buffer.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsygvd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork);

            /// <summary>
            /// The helper functions below can calculate the sizes needed for pre-allocated buffer.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsygvd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork);

            /// <summary>
            /// The helper functions below can calculate the sizes needed for pre-allocated buffer.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChegvd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork);

            /// <summary>
            /// The helper functions below can calculate the sizes needed for pre-allocated buffer.
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="lwork">size of work</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhegvd_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork);


            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix-pair (A,B).
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsygvd(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix-pair (A,B).
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsygvd(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix-pair (A,B).
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChegvd(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            /// <summary>
            /// This function computes eigenvalues and eigenvectors of a symmetric (Hermitian) n×n matrix-pair (A,B).
            /// </summary>
            /// <param name="handle">handle to the cuSolverDN library context.</param>
            /// <param name="itype">Specifies the problem type to be solved.</param>
            /// <param name="jobz">specifies options to either compute eigenvalue only or compute eigen-pair.</param>
            /// <param name="uplo">specifies which part of A and B are stored.</param>
            /// <param name="n">number of rows (or columns) of matrix A and B.</param>
            /// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
            /// <param name="lda">leading dimension of two-dimensional array used to store matrix A. lda is not less than max(1,n).</param>
            /// <param name="B">array of dimension ldb * n.</param>
            /// <param name="ldb">leading dimension of two-dimensional array used to store matrix B. ldb is not less than max(1,n).</param>
            /// <param name="W">a real array of dimension n. The eigenvalue values of A, sorted so that W(i) &gt;= W(i+1).</param>
            /// <param name="work">working space, array of size lwork.</param>
            /// <param name="lwork">size of work, returned by sygvd_bufferSize.</param>
            /// <param name="info">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th
            /// parameter is wrong. if devInfo = i (&gt;0), devInfo indicates either potrf or syevd is wrong.</param>
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhegvd(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);
            #endregion


            #region IRS headers 

            // =============================================================================
            // IRS helper function API
            // =============================================================================
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsCreate(ref cusolverDnIRSParams params_ptr);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsDestroy(cusolverDnIRSParams parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetRefinementSolver(
            cusolverDnIRSParams parameters, cusolverIRSRefinement refinement_solver);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetSolverMainPrecision(
            cusolverDnIRSParams parameters, cusolverPrecType solver_main_precision);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetSolverLowestPrecision(
            cusolverDnIRSParams parameters, cusolverPrecType solver_lowest_precision);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetSolverPrecisions(
            cusolverDnIRSParams parameters, cusolverPrecType solver_main_precision, cusolverPrecType solver_lowest_precision);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetTol(cusolverDnIRSParams parameters, double val);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams parameters, double val);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams parameters, int maxiters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams parameters, int maxiters_inner);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams parameters, ref int maxiters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams parameters);

            #endregion
            #region cusolverDnIRSInfos prototypes

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSInfosDestroy(
                    cusolverDnIRSInfos infos);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSInfosCreate(
                    ref cusolverDnIRSInfos infos_ptr);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSInfosGetNiters(
                        cusolverDnIRSInfos infos,
                        ref int niters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSInfosGetOuterNiters(
                        cusolverDnIRSInfos infos,
                        ref int outer_niters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSInfosGetMaxIters(
                        cusolverDnIRSInfos infos,
                        ref int maxiters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSInfosRequestResidual(
                    cusolverDnIRSInfos infos);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSInfosGetResidualHistory(
                        cusolverDnIRSInfos infos,
                        ref IntPtr residual_history);


            #endregion
            #region IRS functions API

            /*******************************************************************************//*
			 * [ZZ, ZC, ZK, CC, CK, DD, DS, DH, SS, SH,]gesv users API Prototypes
			 * */
            /*******************************************************************************/
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZZgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZCgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZKgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZEgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZYgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCCgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCEgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCKgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);



            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCYgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDDgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDSgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDHgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDBgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDXgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSSgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSHgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSBgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSXgesv(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);
            /*******************************************************************************/

            #endregion
            #region [ZZ,ZC, ZK, CC, CK, DD, DS, DH, SS, SH,]gesv_bufferSize users API Prototypes



            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZZgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZCgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZKgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZEgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZYgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCCgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCKgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);



            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCEgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCYgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDDgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDSgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDHgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDBgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDXgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSSgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSHgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSBgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSXgesv_bufferSize(
                    cusolverDnHandle handle,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dipiv,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);
            /*******************************************************************************/


            #endregion

            #region gels users API
            /*******************************************************************************//*
			 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels 
			 * users API Prototypes */
            /*******************************************************************************/
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZZgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZCgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZKgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZEgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZYgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCCgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCKgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCEgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCYgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDDgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDSgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDHgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDBgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDXgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSSgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSHgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSBgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSXgels(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int iter,
                    CUdeviceptr d_info);
            /*******************************************************************************/

            /*******************************************************************************//*
			 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels_bufferSize 
			 * API prototypes */
            /*******************************************************************************/
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZZgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZCgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZKgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZEgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZYgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCCgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCKgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCEgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCYgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDDgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDSgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDHgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDBgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDXgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSSgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSHgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSBgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSXgels_bufferSize(
                    cusolverDnHandle handle,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, ref SizeT lwork_bytes);
            #endregion
            #region expert users API for IRS Prototypes

            /*******************************************************************************//*
			 * 
			 * */
            /*******************************************************************************/
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSXgesv(
                    cusolverDnHandle handle,
                    cusolverDnIRSParams gesv_irs_parameters,
                    cusolverDnIRSInfos gesv_irs_infos,
                    int n, int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int niters,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSXgesv_bufferSize(
                    cusolverDnHandle handle,
                    cusolverDnIRSParams parameters,
                    int n, int nrhs,
                    ref SizeT lwork_bytes);




            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSXgels(
                    cusolverDnHandle handle,
                    cusolverDnIRSParams gels_irs_params,
                    cusolverDnIRSInfos gels_irs_infos,
                    int m,
                    int n,
                    int nrhs,
                    CUdeviceptr dA, int ldda,
                    CUdeviceptr dB, int lddb,
                    CUdeviceptr dX, int lddx,
                    CUdeviceptr dWorkspace, SizeT lwork_bytes,
                    ref int niters,
                    CUdeviceptr d_info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnIRSXgels_bufferSize(
                    cusolverDnHandle handle,
                    cusolverDnIRSParams parameters,
                    int m,
                    int n,
                    int nrhs,
                    ref SizeT lwork_bytes);


            #endregion
            #region batched Cholesky factorization and its solver

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSpotrfBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr Aarray,
                int lda,
                CUdeviceptr infoArray,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDpotrfBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr Aarray,
                int lda,
                CUdeviceptr infoArray,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCpotrfBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr Aarray,
                int lda,
                CUdeviceptr infoArray,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZpotrfBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr Aarray,
                int lda,
                CUdeviceptr infoArray,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSpotrsBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                int nrhs, /* only support rhs = 1*/
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr d_info,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDpotrsBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                int nrhs, /* only support rhs = 1*/
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr d_info,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCpotrsBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                int nrhs, /* only support rhs = 1*/
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr d_info,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZpotrsBatched(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                int nrhs, /* only support rhs = 1*/
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr d_info,
                int batchSize);

            #endregion
            #region s.p.d. matrix inversion (POTRI) and auxiliary routines (TRTRI and LAUUM) 


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSpotri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDpotri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCpotri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZpotri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSpotri(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDpotri(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCpotri(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZpotri(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnStrtri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDtrtri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCtrtri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZtrtri_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnStrtri(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDtrtri(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCtrtri(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZtrtri(
                cusolverDnHandle handle,
                FillMode uplo,
                DiagType diag,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            #endregion
            #region lauum, auxiliar routine for s.p.d matrix inversion

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSlauum_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDlauum_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnClauum_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZlauum_bufferSize(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSlauum(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDlauum(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnClauum(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZlauum(
                cusolverDnHandle handle,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr devInfo);







            #endregion

            #region Symmetric indefinite solve (SYTRS)

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsytrs_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsytrs_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCsytrs_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZsytrs_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsytrs(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsytrs(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCsytrs(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZsytrs(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    int nrhs,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr B,
                    int ldb,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);

            #endregion
            #region Symmetric indefinite inversion (sytri)

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsytri_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsytri_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCsytri_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZsytri_bufferSize(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsytri(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsytri(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCsytri(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZsytri(
                    cusolverDnHandle handle,
                    FillMode uplo,
                    int n,
                    CUdeviceptr A,
                    int lda,
                    CUdeviceptr ipiv,
                    CUdeviceptr work,
                    int lwork,
                    CUdeviceptr info);






            #endregion
            #region standard selective symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevdx(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevdx(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevdx(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevdx(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            #endregion
            #region selective generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsygvdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsygvdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChegvdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhegvdx_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                ref int lwork);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsygvdx(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsygvdx(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChegvdx(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                float vl,
                float vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhegvdx(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                double vl,
                double vu,
                int il,
                int iu,
                ref int meig,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info);







            #endregion

            #region 

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCreateSyevjInfo(
                ref syevjInfo info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDestroySyevjInfo(
                syevjInfo info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevjSetTolerance(
                syevjInfo info,
                double tolerance);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevjSetMaxSweeps(
                syevjInfo info,
                int max_sweeps);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevjSetSortEig(
                syevjInfo info,
                int sort_eig);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevjGetResidual(
                cusolverDnHandle handle,
                syevjInfo info,
                ref double residual);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevjGetSweeps(
                cusolverDnHandle handle,
                syevjInfo info,
                ref int executed_sweeps);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters,
                int batchSize
                );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters,
                int batchSize
                );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters,
                int batchSize
                );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters,
                int batchSize
                );


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters,
                int batchSize
                );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters,
                int batchSize
                );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters,
                int batchSize
                );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters,
                int batchSize
                );


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsyevj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsyevj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCheevj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZheevj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsygvj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsygvj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChegvj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhegvj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                ref int lwork,
                syevjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSsygvj(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDsygvj(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnChegvj(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZhegvj(
                cusolverDnHandle handle,
                cusolverEigType itype,
                cusolverEigMode jobz,
                FillMode uplo,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr B,
                int ldb,
                CUdeviceptr W,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                syevjInfo parameters);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCreateGesvdjInfo(
                ref gesvdjInfo info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDestroyGesvdjInfo(
                gesvdjInfo info);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdjSetTolerance(
                gesvdjInfo info,
                double tolerance);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdjSetMaxSweeps(
                gesvdjInfo info,
                int max_sweeps);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdjSetSortEig(
                gesvdjInfo info,
                int sort_svd);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdjGetResidual(
                cusolverDnHandle handle,
                gesvdjInfo info,
                ref double residual);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdjGetSweeps(
                cusolverDnHandle handle,
                gesvdjInfo info,
                ref int executed_sweeps);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSgesvdjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDgesvdjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCgesvdjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZgesvdjBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSgesvdjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDgesvdjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCgesvdjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZgesvdjBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters,
                int batchSize);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSgesvdj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDgesvdj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCgesvdj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZgesvdj_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                ref int lwork,
                gesvdjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSgesvdj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDgesvdj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCgesvdj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZgesvdj(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int econ,
                int m,
                int n,
                CUdeviceptr A,
                int lda,
                CUdeviceptr S,
                CUdeviceptr U,
                int ldu,
                CUdeviceptr V,
                int ldv,
                CUdeviceptr work,
                int lwork,
                CUdeviceptr info,
                gesvdjInfo parameters);

            #endregion
            #region batched approximate SVD


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSgesvdaStridedBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                ref int lwork,
                int batchSize
                );


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDgesvdaStridedBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                ref int lwork,
                int batchSize
                );


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCgesvdaStridedBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                ref int lwork,
                int batchSize
                );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZgesvdaStridedBatched_bufferSize(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                ref int lwork,
                int batchSize
                );


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSgesvdaStridedBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                CUdeviceptr d_work,
                int lwork,
                CUdeviceptr d_info,
                double[] h_R_nrmF,
                int batchSize);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDgesvdaStridedBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                CUdeviceptr d_work,
                int lwork,
                CUdeviceptr d_info,
                double[] h_R_nrmF,
                int batchSize);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCgesvdaStridedBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                CUdeviceptr d_work,
                int lwork,
                CUdeviceptr d_info,
                double[] h_R_nrmF,
                int batchSize);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnZgesvdaStridedBatched(
                cusolverDnHandle handle,
                cusolverEigMode jobz,
                int rank,
                int m,
                int n,
                CUdeviceptr d_A,
                int lda,
                long strideA,
                CUdeviceptr d_S,
                long strideS,
                CUdeviceptr d_U,
                int ldu,
                long strideU,
                CUdeviceptr d_V,
                int ldv,
                long strideV,
                CUdeviceptr d_work,
                int lwork,
                CUdeviceptr d_info,
                double[] h_R_nrmF,
                int batchSize);

            #endregion

            #region 64-bit API for POTRF

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnCreateParams(
                ref cusolverDnParams parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnDestroyParams(
                cusolverDnParams parameters);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnSetAdvOptions(
                cusolverDnParams parameters,
                cusolverDnFunction function,
                cusolverAlgMode algo);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnPotrf_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,

                CUdeviceptr A,
                long lda,
                cudaDataType computeType,
                ref SizeT workspaceInBytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnPotrf(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType computeType,
                CUdeviceptr pBuffer,
                SizeT workspaceInBytes,
                CUdeviceptr info);

            #endregion

            #region 64-bit API for POTRS 
            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnPotrs(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                FillMode uplo,
                long n,
                long nrhs,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeB,

                CUdeviceptr B,
                long ldb,
                CUdeviceptr info);

            #endregion

            #region 64-bit API for GEQRF 
            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnGeqrf_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                long m,
                long n,
                cudaDataType dataTypeA,

                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeTau,

                CUdeviceptr tau,
                cudaDataType computeType,
                ref SizeT workspaceInBytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnGeqrf(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeTau,
                CUdeviceptr tau,
                cudaDataType computeType,
                CUdeviceptr pBuffer,
                SizeT workspaceInBytes,
                CUdeviceptr info);

            #endregion

            #region 64-bit API for GETRF
            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnGetrf_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType computeType,
                ref SizeT workspaceInBytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnGetrf(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                CUdeviceptr ipiv,
                cudaDataType computeType,
                CUdeviceptr pBuffer,
                SizeT workspaceInBytes,
                CUdeviceptr info);

            #endregion

            #region 64-bit API for GETRS
            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnGetrs(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                Operation trans,
                long n,
                long nrhs,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                CUdeviceptr ipiv,
                cudaDataType dataTypeB,
                CUdeviceptr B,
                long ldb,
                CUdeviceptr info);

            #endregion

            #region 64-bit API for SYEVD
            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnSyevd_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                ref SizeT workspaceInBytes);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnSyevd(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                CUdeviceptr pBuffer,
                SizeT workspaceInBytes,
                CUdeviceptr info);

            #endregion

            #region 64-bit API for SYEVDX 
            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnSyevdx_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                IntPtr vl,
                IntPtr vu,
                long il,
                long iu,
                ref long h_meig,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                ref SizeT workspaceInBytes);


            [DllImport(CUSOLVE_API_DLL_NAME)]
            [Obsolete("Deprecated in Cuda version 11.1")]
            public static extern cusolverStatus cusolverDnSyevdx(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                IntPtr vl,
                IntPtr vu,
                long il,
                long iu,
                ref long meig64,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                CUdeviceptr pBuffer,
                SizeT workspaceInBytes,
                CUdeviceptr info);
            #endregion

            #region new 64-bit API
            #region 64-bit API for POTRF

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXpotrf_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType computeType,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXpotrf(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType computeType,
                CUdeviceptr bufferOnDevice,
                SizeT workspaceInBytesOnDevice,
                byte[] bufferOnHost,
                SizeT workspaceInBytesOnHost,
                CUdeviceptr info);

            /* 64-bit API for POTRS */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXpotrs(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                FillMode uplo,
                long n,
                long nrhs,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeB,
                CUdeviceptr B,
                long ldb,
                CUdeviceptr info);

            /* 64-bit API for GEQRF */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgeqrf_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeTau,
                CUdeviceptr tau,
                cudaDataType computeType,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgeqrf(
            cusolverDnHandle handle,
            cusolverDnParams parameters,
            long m,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType dataTypeTau,
            CUdeviceptr tau,
            cudaDataType computeType,
            CUdeviceptr bufferOnDevice,
            SizeT workspaceInBytesOnDevice,
            byte[] bufferOnHost,
            SizeT workspaceInBytesOnHost,
            CUdeviceptr info);

            /* 64-bit API for GETRF */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgetrf_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType computeType,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgetrf(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                CUdeviceptr ipiv,
                cudaDataType computeType,
                CUdeviceptr bufferOnDevice,
                SizeT workspaceInBytesOnDevice,
                byte[] bufferOnHost,
                SizeT workspaceInBytesOnHost,
                CUdeviceptr info);

            /* 64-bit API for GETRS */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgetrs(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                Operation trans,
                long n,
                long nrhs,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                CUdeviceptr ipiv,
                cudaDataType dataTypeB,
                CUdeviceptr B,
                long ldb,
                CUdeviceptr info);

            /* 64-bit API for SYEVD */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevd_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevd(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                CUdeviceptr bufferOnDevice,
                SizeT workspaceInBytesOnDevice,
                byte[] bufferOnHost,
                SizeT workspaceInBytesOnHost,
                CUdeviceptr info);

            /* 64-bit API for SYEVDX */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevdx_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                IntPtr vl,
                IntPtr vu,
                long il,
                long iu,
                ref long h_meig,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXsyevdx(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                cusolverEigRange range,
                FillMode uplo,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                IntPtr vl,
                IntPtr vu,
                long il,
                long iu,
                ref long meig64,
                cudaDataType dataTypeW,
                CUdeviceptr W,
                cudaDataType computeType,
                CUdeviceptr bufferOnDevice,
                SizeT workspaceInBytesOnDevice,
                byte[] bufferOnHost,
                SizeT workspaceInBytesOnHost,
                CUdeviceptr info);

            /* 64-bit API for GESVD */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvd_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                sbyte jobu,
                sbyte jobvt,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeS,
                CUdeviceptr S,
                cudaDataType dataTypeU,
                CUdeviceptr U,
                long ldu,
                cudaDataType dataTypeVT,
                CUdeviceptr VT,
                long ldvt,
                cudaDataType computeType,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvd(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                sbyte jobu,
                sbyte jobvt,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeS,
                CUdeviceptr S,
                cudaDataType dataTypeU,
                CUdeviceptr U,
                long ldu,
                cudaDataType dataTypeVT,
                CUdeviceptr VT,
                long ldvt,
                cudaDataType computeType,
                CUdeviceptr bufferOnDevice,
                SizeT workspaceInBytesOnDevice,
                byte[] bufferOnHost,
                SizeT workspaceInBytesOnHost,
                CUdeviceptr info);

            /* 64-bit API for GESVDP */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdp_bufferSize(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                int econ,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeS,
                CUdeviceptr S,
                cudaDataType dataTypeU,
                CUdeviceptr U,
                long ldu,
                cudaDataType dataTypeV,
                CUdeviceptr V,
                long ldv,
                cudaDataType computeType,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdp(
                cusolverDnHandle handle,
                cusolverDnParams parameters,
                cusolverEigMode jobz,
                int econ,
                long m,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                cudaDataType dataTypeS,
                CUdeviceptr S,
                cudaDataType dataTypeU,
                CUdeviceptr U,
                long ldu,
                cudaDataType dataTypeV,
                CUdeviceptr V,
                long ldv,
                cudaDataType computeType,
                CUdeviceptr bufferOnDevice,
                SizeT workspaceInBytesOnDevice,
                byte[] bufferOnHost,
                SizeT workspaceInBytesOnHost,
                CUdeviceptr d_info,
                ref double h_err_sigma);
            #endregion


            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdr_bufferSize(
                    cusolverDnHandle handle,
                    cusolverDnParams parameters,
                    sbyte jobu,
                    sbyte jobv,
                    long m,
                    long n,
                    long k,
                    long p,
                    long niters,
                    cudaDataType dataTypeA,
                    CUdeviceptr A,
                    long lda,
                    cudaDataType dataTypeSrand,
                    CUdeviceptr Srand,
                    cudaDataType dataTypeUrand,
                    CUdeviceptr Urand,
                    long ldUrand,
                    cudaDataType dataTypeVrand,
                    CUdeviceptr Vrand,
                    long ldVrand,
                    cudaDataType computeType,
                    ref SizeT workspaceInBytesOnDevice,
                    ref SizeT workspaceInBytesOnHost
                    );

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverDnXgesvdr(
                    cusolverDnHandle handle,
                    cusolverDnParams parameters,
                    sbyte jobu,
                    sbyte jobv,
                    long m,
                    long n,
                    long k,
                    long p,
                    long niters,
                    cudaDataType dataTypeA,
                    CUdeviceptr A,
                    long lda,
                    cudaDataType dataTypeSrand,
                    CUdeviceptr Srand,
                    cudaDataType dataTypeUrand,
                    CUdeviceptr Urand,
                    long ldUrand,
                    cudaDataType dataTypeVrand,
                    CUdeviceptr Vrand,
                    long ldVrand,
                    cudaDataType computeType,
                    CUdeviceptr bufferOnDevice,
                    SizeT workspaceInBytesOnDevice,
                    byte[] bufferOnHost,
                    SizeT workspaceInBytesOnHost,
                    CUdeviceptr d_info
                    );

            #endregion
        }

        /// <summary>
        /// The cuSolverSP library was mainly designed to a solve sparse linear system AxB and the least-squares problem
        /// x = argmin||A*z-b||
        /// </summary>
        public static class Sparse
        {
#if (NETCOREAPP)
            static Sparse()
            {
                CudaSolveNativeMethods.Init();
            }
#endif
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
            /// type is CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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


            #region CPU metis
            /* --------- CPU metis 
			 *   symmetric reordering 
			 */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverSpXcsrmetisndHost(
                cusolverSpHandle handle,
                int n,
                int nnzA,
                cusparseMatDescr descrA,
                int[] csrRowPtrA,
                int[] csrColIndA,
                long[] options,
                int[] p);


            /* --------- CPU zfd
			 *  Zero free diagonal reordering
			 */
            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverSpScsrzfdHost(
                cusolverSpHandle handle,
                int n,
                int nnz,
                cusparseMatDescr descrA,
                float[] csrValA,
                int[] csrRowPtrA,
                int[] csrColIndA,
                int[] P,
                ref int numnz);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverSpDcsrzfdHost(
                cusolverSpHandle handle,
                int n,
                int nnz,
                cusparseMatDescr descrA,
                double[] csrValA,
                int[] csrRowPtrA,
                int[] csrColIndA,
                int[] P,
                ref int numnz);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverSpCcsrzfdHost(
                cusolverSpHandle handle,
                int n,
                int nnz,
                cusparseMatDescr descrA,
                cuFloatComplex[] csrValA,
                int[] csrRowPtrA,
                int[] csrColIndA,
                int[] P,
                ref int numnz);

            [DllImport(CUSOLVE_API_DLL_NAME)]
            public static extern cusolverStatus cusolverSpZcsrzfdHost(
                cusolverSpHandle handle,
                int n,
                int nnz,
                cusparseMatDescr descrA,
                cuDoubleComplex[] csrValA,
                int[] csrRowPtrA,
                int[] csrColIndA,
                int[] P,
                ref int numnz);

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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            /// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
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
            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="info">opaque structure for QR factorization.</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpCreateCsrqrInfo(ref csrqrInfo info);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="info">opaque structure for QR factorization.</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpDestroyCsrqrInfo(csrqrInfo info);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnzA">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrRowPtrA">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpXcsrqrAnalysisBatched(cusolverSpHandle handle, int m, int n, int nnzA, cusparseMatDescr descrA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, csrqrInfo info);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrVal">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtr">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="internalDataInBytes">number of bytes of the internal data.</param>
            ///// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpScsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrVal">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtr">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="internalDataInBytes">number of bytes of the internal data.</param>
            ///// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpDcsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrVal">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtr">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="internalDataInBytes">number of bytes of the internal data.</param>
            ///// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpCcsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrVal">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtr">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColInd">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="internalDataInBytes">number of bytes of the internal data.</param>
            ///// <param name="workspaceInBytes">number of bytes of the buffer in numerical factorization.</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpZcsrqrBufferInfoBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, int batchSize, csrqrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrValA">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtrA">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
            ///// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="pBuffer">buffer allocated by the user, the size is returned
            ///// by cusolverSpXcsrqrBufferInfoBatched().</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpScsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrValA">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtrA">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
            ///// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="pBuffer">buffer allocated by the user, the size is returned
            ///// by cusolverSpXcsrqrBufferInfoBatched().</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpDcsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrValA">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtrA">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
            ///// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="pBuffer">buffer allocated by the user, the size is returned
            ///// by cusolverSpXcsrqrBufferInfoBatched().</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpCcsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);

            ///// <summary>
            ///// The batched sparse QR factorization is used to solve either a set of least-squares
            ///// problems or a set of linear systems
            ///// </summary>
            ///// <param name="handle">handle to the cuSolverSP library context.</param>
            ///// <param name="m">number of rows of each matrix Aj.</param>
            ///// <param name="n">number of columns of each matrix Aj.</param>
            ///// <param name="nnz">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
            ///// <param name="descrA">the descriptor of matrix A. The supported matrix type is
            ///// CUSPARSE_MATRIXYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
            ///// <param name="csrValA">array of nnzA*batchSize nonzero 
            ///// elements of matrices A0, A1, .... All matrices are aggregated one after another.</param>
            ///// <param name="csrRowPtrA">integer array of m+1 elements that contains the
            ///// start of every row and the end of the last row plus one.</param>
            ///// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
            ///// <param name="b">array of m*batchSize of right-hand-side vectors b0, b1, .... All vectors are aggregated one after another.</param>
            ///// <param name="x">array of m*batchSize of solution vectors x0, x1, .... All vectors are aggregated one after another.</param>
            ///// <param name="batchSize">number of systems to be solved.</param>
            ///// <param name="info">opaque structure for QR factorization.</param>
            ///// <param name="pBuffer">buffer allocated by the user, the size is returned
            ///// by cusolverSpXcsrqrBufferInfoBatched().</param>
            //[DllImport(CUSOLVE_API_DLL_NAME)]
            //public static extern cusolverStatus cusolverSpZcsrqrsvBatched(cusolverSpHandle handle, int m, int n, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr b, CUdeviceptr x, int batchSize, csrqrInfo info, CUdeviceptr pBuffer);






            // /*
            // * "diag" is a device array of size N.
            // * cusolverSp<t>csrcholDiag returns diag(L) to "diag" where A(P,P) = L*L**T
            // * "diag" can estimate det(A) because det(A(P,P)) = det(A) = det(L)^2 if A = L*L**T.
            // * 
            // * cusolverSp<t>csrcholDiag must be called after cusolverSp<t>csrcholFactor.
            // * otherwise "diag" is wrong.
            // */
            //cusolverStatus cusolverSpScsrcholDiag(
            //	cusolverSpHandle handle,
            //	csrcholInfo info,
            //	CUdeviceptr diag);

            //cusolverStatus cusolverSpDcsrcholDiag(
            //	cusolverSpHandle handle,
            //	csrcholInfo info,
            //	CUdeviceptr diag);

            //cusolverStatus cusolverSpCcsrcholDiag(
            //	cusolverSpHandle handle,
            //	csrcholInfo info,
            //	CUdeviceptr diag);

            //cusolverStatus cusolverSpZcsrcholDiag(
            //	cusolverSpHandle handle,
            //	csrcholInfo info,
            //	CUdeviceptr diag);
            #endregion

        }

        /// <summary>
        /// The cuSolverRF library was designed to accelerate solution of sets of linear systems by
        /// fast re-factorization when given new coefficients in the same sparsity pattern
        /// A_i x_i = f_i
        /// </summary>
        public static class Refactorization
        {
#if (NETCOREAPP)
            static Refactorization()
            {
                CudaSolveNativeMethods.Init();
            }
#endif
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
            public static extern cusolverStatus cusolverRfBatchSetupHost(int batchSize, int n, int nnzA, int[] h_csrRowPtrA, int[] h_csrColIndA, IntPtr[] h_csrValA_array, int nnzL,
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
