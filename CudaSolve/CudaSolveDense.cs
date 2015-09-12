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
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.CudaBlas;

namespace ManagedCuda.CudaSolve
{
	/// <summary>
	/// CudaSolveDense: The cuSolverDN library was designed to solve dense linear systems of the form Ax=B
	/// </summary>
	public class CudaSolveDense : IDisposable
	{
		bool disposed;
		cusolverStatus res;
		cusolverDnHandle _handle;

        #region Constructor
		/// <summary>
		/// Create new dense solve instance
		/// </summary>
        public CudaSolveDense()
        {
			_handle = new cusolverDnHandle();
			res = CudaSolveNativeMethods.Dense.cusolverDnCreate(ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDestroy", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

		/// <summary>
		/// Create new dense solve instance using stream stream
		/// </summary>
		/// <param name="stream"></param>
        public CudaSolveDense(CudaStream stream)
			: this()
        {
			SetStream(stream);
        }

        /// <summary>
        /// For dispose
        /// </summary>
		~CudaSolveDense()
        {
            Dispose(false);
        }
        #endregion

        #region Dispose
        /// <summary>
        /// Dispose
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// For IDisposable
        /// </summary>
        /// <param name="fDisposing"></param>
        protected virtual void Dispose(bool fDisposing)
        {
            if (fDisposing && !disposed)
            {
				res = CudaSolveNativeMethods.Dense.cusolverDnDestroy(_handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

		#region Stream

		/// <summary>
		/// This function sets the stream to be used by the cuSolverDN library to execute its routines.
		/// </summary>
		/// <param name="stream">the stream to be used by the library.</param>
		public void SetStream(CudaStream stream)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSetStream(_handle, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSetStream", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function gets the stream to be used by the cuSolverDN library to execute its routines.
		/// </summary>
		public CudaStream GetStream()
		{
			CUstream stream = new CUstream();
			res = CudaSolveNativeMethods.Dense.cusolverDnGetStream(_handle, ref stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnGetStream", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);

			return new CudaStream(stream);
		}
		#endregion

		#region Cholesky factorization and its solver
		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int PotrfBufferSize(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnSpotrf_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSpotrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int PotrfBufferSize(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnDpotrf_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDpotrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int PotrfBufferSize(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnCpotrf_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCpotrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int PotrfBufferSize(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnZpotrf_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZpotrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}

		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of Workspace</param>
		/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
		/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
		public void Potrf(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSpotrf(_handle, uplo, n, A.DevicePointer, lda, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSpotrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of Workspace</param>
		/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
		/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
		public void Potrf(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDpotrf(_handle, uplo, n, A.DevicePointer, lda, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDpotrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of Workspace</param>
		/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
		/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
		public void Potrf(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCpotrf(_handle, uplo, n, A.DevicePointer, lda, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCpotrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of Workspace</param>
		/// <param name="devInfo">if devInfo = 0, the Cholesky factorization is successful. if devInfo
		/// = -i, the i-th parameter is wrong. if devInfo = i, the leading minor of order i is not positive definite.</param>
		public void Potrf(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZpotrf(_handle, uplo, n, A.DevicePointer, lda, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZpotrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}








		/// <summary>
		/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
		/// </summary>
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
		public void Potrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSpotrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSpotrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
		/// </summary>
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
		public void Potrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDpotrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDpotrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
		/// </summary>
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
		public void Potrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCpotrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCpotrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function solves a system of linear equations A*X=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
		/// </summary>
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
		public void Potrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZpotrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZpotrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}




		#endregion

		#region LU Factorization
		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GetrfBufferSize(int m, int n, CudaDeviceVariable<float> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnSgetrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgetrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GetrfBufferSize(int m, int n, CudaDeviceVariable<double> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnDgetrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgetrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GetrfBufferSize(int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnCgetrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgetrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GetrfBufferSize(int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnZgetrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgetrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}




		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is
		/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
		public void Getrf(int m, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> Workspace, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSgetrf(_handle, m, n, A.DevicePointer, lda, Workspace.DevicePointer, devIpiv.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgetrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is
		/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
		public void Getrf(int m, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> Workspace, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDgetrf(_handle, m, n, A.DevicePointer, lda, Workspace.DevicePointer, devIpiv.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgetrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is
		/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
		public void Getrf(int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> Workspace, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCgetrf(_handle, m, n, A.DevicePointer, lda, Workspace.DevicePointer, devIpiv.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgetrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function computes the LU factorization of a m×n matrix P*A=L*U
		/// where A is a m×n matrix, P is a permutation matrix, L is a lower triangular matrix with
		/// unit diagonal, and U is an upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="devIpiv">array of size at least min(m,n), containing pivot indices.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is
		/// successful. if devInfo = -i, the i-th parameter is wrong. if devInfo = i, the U(i,i) = 0.</param>
		public void Getrf(int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> Workspace, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZgetrf(_handle, m, n, A.DevicePointer, lda, Workspace.DevicePointer, devIpiv.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgetrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


		#endregion

		#region Row pivoting
		///// <summary/>
		//[DllImport(CUSOLVE_API_DLL_NAME)]
		//public static extern cusolverStatus cusolverDnSlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

		///// <summary/>
		//[DllImport(CUSOLVE_API_DLL_NAME)]
		//public static extern cusolverStatus cusolverDnDlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

		///// <summary/>
		//[DllImport(CUSOLVE_API_DLL_NAME)]
		//public static extern cusolverStatus cusolverDnClaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

		///// <summary/>
		//[DllImport(CUSOLVE_API_DLL_NAME)]
		//public static extern cusolverStatus cusolverDnZlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);
		#endregion

		#region LU solve
		/// <summary>
		/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="nrhs">number of right-hand sides.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
		/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
		public void Getrs(Operation trans, int n, int nrhs, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSgetrs(_handle, trans, n, nrhs, A.DevicePointer, lda, devIpiv.DevicePointer, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgetrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="nrhs">number of right-hand sides.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
		/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
		public void Getrs(Operation trans, int n, int nrhs, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDgetrs(_handle, trans, n, nrhs, A.DevicePointer, lda, devIpiv.DevicePointer, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgetrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="nrhs">number of right-hand sides.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
		/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
		public void Getrs(Operation trans, int n, int nrhs, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCgetrs(_handle, trans, n, nrhs, A.DevicePointer, lda, devIpiv.DevicePointer, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgetrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves a linear system of multiple right-hand sides op(A)*X=B.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="nrhs">number of right-hand sides.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="devIpiv">array of size at least n, containing pivot indices.</param>
		/// <param name="B">array of dimension ldb * nrhs with ldb is not less than max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="devInfo">if devInfo = 0, the operation is successful. if devInfo = -i, the i-th parameter is wrong.</param>
		public void Getrs(Operation trans, int n, int nrhs, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<int> devIpiv, CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZgetrs(_handle, trans, n, nrhs, A.DevicePointer, lda, devIpiv.DevicePointer, B.DevicePointer, ldb, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgetrs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		#endregion

		#region QR factorization

		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="TAU">array of dimension at least min(m,n).</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of working array Workspace.</param>
		/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
		public void Geqrf(int m, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> TAU, CudaDeviceVariable<float> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSgeqrf(_handle, m, n, A.DevicePointer, lda, TAU.DevicePointer, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgeqrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="TAU">array of dimension at least min(m,n).</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of working array Workspace.</param>
		/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
		public void Geqrf(int m, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> TAU, CudaDeviceVariable<double> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDgeqrf(_handle, m, n, A.DevicePointer, lda, TAU.DevicePointer, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgeqrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="TAU">array of dimension at least min(m,n).</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of working array Workspace.</param>
		/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
		public void Geqrf(int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> TAU, CudaDeviceVariable<cuFloatComplex> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCgeqrf(_handle, m, n, A.DevicePointer, lda, TAU.DevicePointer, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgeqrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="TAU">array of dimension at least min(m,n).</param>
		/// <param name="Workspace">working space, array of size Lwork.</param>
		/// <param name="Lwork">size of working array Workspace.</param>
		/// <param name="devInfo">if info = 0, the LU factorization is successful. if info = -i, the i-th parameter is wrong.</param>
		public void Geqrf(int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> TAU, CudaDeviceVariable<cuDoubleComplex> Workspace, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZgeqrf(_handle, m, n, A.DevicePointer, lda, TAU.DevicePointer, Workspace.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgeqrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}



		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
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
		public void Ormqr(SideMode side, Operation trans, int m, int n, int k, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> tau, CudaDeviceVariable<float> C, int ldc, CudaDeviceVariable<float> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSormqr(_handle, side, trans, m, n, k, A.DevicePointer, lda, tau.DevicePointer, C.DevicePointer, ldc, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSormqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
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
		public void Ormqr(SideMode side, Operation trans, int m, int n, int k, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> tau, CudaDeviceVariable<double> C, int ldc, CudaDeviceVariable<double> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDormqr(_handle, side, trans, m, n, k, A.DevicePointer, lda, tau.DevicePointer, C.DevicePointer, ldc, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDormqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
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
		public void Unmqr(SideMode side, Operation trans, int m, int n, int k, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> tau, CudaDeviceVariable<cuFloatComplex> C, int ldc, CudaDeviceVariable<cuFloatComplex> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCunmqr(_handle, side, trans, m, n, k, A.DevicePointer, lda, tau.DevicePointer, C.DevicePointer, ldc, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCunmqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
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
		public void Unmqr(SideMode side, Operation trans, int m, int n, int k, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> tau, CudaDeviceVariable<cuDoubleComplex> C, int ldc, CudaDeviceVariable<cuDoubleComplex> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZunmqr(_handle, side, trans, m, n, k, A.DevicePointer, lda, tau.DevicePointer, C.DevicePointer, ldc, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZunmqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		#endregion

		#region QR factorization workspace query
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GeqrfBufferSize(int m, int n, CudaDeviceVariable<float> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnSgeqrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgeqrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GeqrfBufferSize(int m, int n, CudaDeviceVariable<double> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnDgeqrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgeqrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GeqrfBufferSize(int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnCgeqrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgeqrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the QR factorization of a m×n matrix A=Q*R
		/// where A is a m×n matrix, Q is a m×n matrix, and R is a n×n upper triangular matrix.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,m).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GeqrfBufferSize(int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnZgeqrf_bufferSize(_handle, m, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgeqrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		#endregion

		#region bidiagonal
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
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
		public void Gebrd(int m, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> D, CudaDeviceVariable<float> E, CudaDeviceVariable<float> TAUQ, CudaDeviceVariable<float> TAUP, CudaDeviceVariable<float> Work, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSgebrd(_handle, m, n, A.DevicePointer, lda, D.DevicePointer, E.DevicePointer, TAUQ.DevicePointer, TAUP.DevicePointer, Work.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgebrd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
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
		public void Gebrd(int m, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> D, CudaDeviceVariable<double> E, CudaDeviceVariable<double> TAUQ, CudaDeviceVariable<double> TAUP, CudaDeviceVariable<double> Work, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDgebrd(_handle, m, n, A.DevicePointer, lda, D.DevicePointer, E.DevicePointer, TAUQ.DevicePointer, TAUP.DevicePointer, Work.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgebrd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
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
		public void Gebrd(int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<float> D, CudaDeviceVariable<float> E, CudaDeviceVariable<cuFloatComplex> TAUQ, CudaDeviceVariable<cuFloatComplex> TAUP, CudaDeviceVariable<cuFloatComplex> Work, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCgebrd(_handle, m, n, A.DevicePointer, lda, D.DevicePointer, E.DevicePointer, TAUQ.DevicePointer, TAUP.DevicePointer, Work.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgebrd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
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
		public void Gebrd(int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<double> D, CudaDeviceVariable<double> E, CudaDeviceVariable<cuDoubleComplex> TAUQ, CudaDeviceVariable<cuDoubleComplex> TAUP, CudaDeviceVariable<cuDoubleComplex> Work, int Lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZgebrd(_handle, m, n, A.DevicePointer, lda, D.DevicePointer, E.DevicePointer, TAUQ.DevicePointer, TAUP.DevicePointer, Work.DevicePointer, Lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgebrd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		///// <summary/>
		//[DllImport(CUSOLVE_API_DLL_NAME)]
		//public static extern cusolverStatus cusolverDnSsytrd(cusolverDnHandle handle, char uplo, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr tau, CUdeviceptr Work, int Lwork, CUdeviceptr info);

		///// <summary/>
		//[DllImport(CUSOLVE_API_DLL_NAME)]
		//public static extern cusolverStatus cusolverDnDsytrd(cusolverDnHandle handle, char uplo, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr tau, CUdeviceptr Work, int Lwork, CUdeviceptr info);
		#endregion

		#region bidiagonal factorization workspace query
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GebrdBufferSizeFloat(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnSgebrd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgebrd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GebrdBufferSizeDouble(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnDgebrd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgebrd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GebrdBufferSizeFloatComplex(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnCgebrd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgebrd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function reduces a general real m×n matrix A to upper or lower bidiagonal form B by
		/// an orthogonal transformation: Q^H*A*P=B
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GebrdBufferSizeDoubleComplex(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnZgebrd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgebrd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		#endregion

		#region singular value decomposition, A = U * Sigma * V^H
		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GesvdBufferSizeFloat(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnSgesvd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GesvdBufferSizeDouble(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnDgesvd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GesvdBufferSizeFloatComplex(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnCgesvd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgesvd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int GesvdBufferSizeDoubleComplex(int m, int n)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnZgesvd_bufferSize(_handle, m, n, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvd_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}


		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
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
		public void Gesvd(char jobu, char jobvt, int m, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> S, CudaDeviceVariable<float> U, int ldu, CudaDeviceVariable<float> VT, int ldvt, CudaDeviceVariable<float> Work, int Lwork, CudaDeviceVariable<float> rwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSgesvd(_handle, jobu, jobvt, m, n, A.DevicePointer, lda, S.DevicePointer, U.DevicePointer, ldu, VT.DevicePointer, ldvt, Work.DevicePointer, Lwork, rwork.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
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
		public void Gesvd(char jobu, char jobvt, int m, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> S, CudaDeviceVariable<double> U, int ldu, CudaDeviceVariable<double> VT, int ldvt, CudaDeviceVariable<double> Work, int Lwork, CudaDeviceVariable<double> rwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDgesvd(_handle, jobu, jobvt, m, n, A.DevicePointer, lda, S.DevicePointer, U.DevicePointer, ldu, VT.DevicePointer, ldvt, Work.DevicePointer, Lwork, rwork.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
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
		public void Gesvd(char jobu, char jobvt, int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<float> S, CudaDeviceVariable<cuFloatComplex> U, int ldu, CudaDeviceVariable<cuFloatComplex> VT, int ldvt, CudaDeviceVariable<cuFloatComplex> Work, int Lwork, CudaDeviceVariable<float> rwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCgesvd(_handle, jobu, jobvt, m, n, A.DevicePointer, lda, S.DevicePointer, U.DevicePointer, ldu, VT.DevicePointer, ldvt, Work.DevicePointer, Lwork, rwork.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgesvd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the singular value decomposition (SVD) of a m×n matrix A and
		/// corresponding the left and/or right singular vectors.
		/// </summary>
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
		public void Gesvd(char jobu, char jobvt, int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<double> S, CudaDeviceVariable<cuDoubleComplex> U, int ldu, CudaDeviceVariable<cuDoubleComplex> VT, int ldvt, CudaDeviceVariable<cuDoubleComplex> Work, int Lwork, CudaDeviceVariable<double> rwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZgesvd(_handle, jobu, jobvt, m, n, A.DevicePointer, lda, S.DevicePointer, U.DevicePointer, ldu, VT.DevicePointer, ldvt, Work.DevicePointer, Lwork, rwork.DevicePointer, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvd", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		#endregion

		#region LDLT,UDUT factorization
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
		/// <param name="work">working space, array of size lwork.</param>
		/// <param name="lwork">size of working space work.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
		/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
		public void Sytrf(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<int> ipiv, CudaDeviceVariable<float> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnSsytrf(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsytrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
		/// <param name="work">working space, array of size lwork.</param>
		/// <param name="lwork">size of working space work.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
		/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
		public void Sytrf(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<int> ipiv, CudaDeviceVariable<double> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnDsytrf(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsytrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
		/// <param name="work">working space, array of size lwork.</param>
		/// <param name="lwork">size of working space work.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
		/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
		public void Sytrf(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<int> ipiv, CudaDeviceVariable<cuFloatComplex> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnCsytrf(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCsytrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="ipiv">array of size at least n, containing pivot indices.</param>
		/// <param name="work">working space, array of size lwork.</param>
		/// <param name="lwork">size of working space work.</param>
		/// <param name="devInfo">if devInfo = 0, the LU factorization is successful. if devInfo = -i, the i-th
		/// parameter is wrong. if devInfo = i, the D(i,i) = 0.</param>
		public void Sytrf(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<int> ipiv, CudaDeviceVariable<cuDoubleComplex> work, int lwork, CudaDeviceVariable<int> devInfo)
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnZsytrf(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, work.DevicePointer, lwork, devInfo.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZsytrf", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		
		#endregion

		#region SYTRF factorization workspace query
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int SytrfBufferSize(int n, CudaDeviceVariable<float> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnSsytrf_bufferSize(_handle, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsytrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int SytrfBufferSize(int n, CudaDeviceVariable<double> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnDsytrf_bufferSize(_handle, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsytrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int SytrfBufferSize(int n, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnCsytrf_bufferSize(_handle, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCsytrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		/// <summary>
		/// This function computes the Bunch-Kaufman factorization of a n×n symmetric indefinite matrix.
		/// </summary>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimension lda * n with lda is not less than max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <returns>size of Workspace</returns>
		public int SytrfBufferSize(int n, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			int Lwork = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnZsytrf_bufferSize(_handle, n, A.DevicePointer, lda, ref Lwork);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZsytrf_bufferSize", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return Lwork;
		}
		#endregion
	}
}
