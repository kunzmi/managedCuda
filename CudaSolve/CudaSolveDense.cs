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
using System.Diagnostics;
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
        private bool disposed;
        private cusolverStatus res;
        private cusolverDnHandle _handle;

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
        //
        //public static extern cusolverStatus cusolverDnSlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

        ///// <summary/>
        //
        //public static extern cusolverStatus cusolverDnDlaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

        ///// <summary/>
        //
        //public static extern cusolverStatus cusolverDnClaswp(cusolverDnHandle handle, int n, CUdeviceptr A, int lda, int k1, int k2, CUdeviceptr devIpiv, int incx);

        ///// <summary/>
        //
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
        /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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
        /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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
        /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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
        /// <param name="ldc">leading dimension of two-dimensional array of matrix C. ldc &gt;= max(1,m).</param>
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
        //
        //public static extern cusolverStatus cusolverDnSsytrd(cusolverDnHandle handle, char uplo, int n, CUdeviceptr A, int lda, CUdeviceptr D, CUdeviceptr E, CUdeviceptr tau, CUdeviceptr Work, int Lwork, CUdeviceptr info);

        ///// <summary/>
        //
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

        #region IRS functions API
        #region [ZZ, ZC, ZK, CC, CK, DD, DS, DH, SS, SH,] gesv_bufferSize users API Prototypes

        public SizeT Gesv_bufferSizeZZ(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZZgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZZgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeZC(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZCgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZCgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeZK(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZKgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZKgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeZE(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZEgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZEgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeZY(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZYgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZYgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeCC(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCCgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCCgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeCK(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCKgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCKgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeCE(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCEgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCEgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeCY(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCYgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCYgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeDD(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDDgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDDgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeDS(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDSgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDSgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeDH(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDHgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDHgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeDB(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDBgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDBgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeDX(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDXgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDXgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeSS(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSSgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSSgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeSH(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSHgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSHgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeSB(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSBgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSBgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public SizeT Gesv_bufferSizeSX(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSXgesv_bufferSize(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSXgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        #endregion

        #region [ZZ, ZC, ZK, CC, CK, DD, DS, DH, SS, SH,]gesv users API Prototypes
        public int ZZgesv(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZZgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZZgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int ZCgesv(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZCgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZCgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int ZKgesv(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZKgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZKgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int ZEgesv(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZEgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZEgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int ZYgesv(int n, int nrhs,
            CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
            CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZYgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZYgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int CCgesv(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCCgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCCgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int CEgesv(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCEgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCEgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int CKgesv(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCKgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCKgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int CYgesv(int n, int nrhs,
            CudaDeviceVariable<cuFloatComplex> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<cuFloatComplex> dB, int lddb,
            CudaDeviceVariable<cuFloatComplex> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCYgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCYgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int DDgesv(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDDgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDDgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int DSgesv(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDSgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDSgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int DHgesv(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDHgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDHgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int DBgesv(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDBgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDBgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int DXgesv(int n, int nrhs,
            CudaDeviceVariable<double> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<double> dB, int lddb,
            CudaDeviceVariable<double> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDXgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDXgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int SSgesv(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSSgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSSgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int SHgesv(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSHgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSHgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int SBgesv(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSBgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSBgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        public int SXgesv(int n, int nrhs,
            CudaDeviceVariable<float> dA, int ldda,
            CudaDeviceVariable<int> dipiv,
            CudaDeviceVariable<float> dB, int lddb,
            CudaDeviceVariable<float> dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSXgesv(_handle, n, nrhs, dA.DevicePointer, ldda,
                dipiv.DevicePointer, dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSXgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        #endregion

        #region [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels users API Prototypes 

        public int ZZgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZZgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZZgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int ZCgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZCgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZCgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int ZKgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZKgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZKgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int ZEgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZEgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZEgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int ZYgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZYgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZYgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int CCgels(
                    int m,
                    int n,
                    int nrhs,
                    CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                    CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                    CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                    CudaDeviceVariable<byte> dWorkspace,
                    CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCCgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCCgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int CKgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCKgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCKgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int CEgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCEgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCEgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int CYgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCYgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCYgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int DDgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDDgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDDgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int DSgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDSgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDSgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int DHgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDHgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDHgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int DBgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDBgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDBgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int DXgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDXgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDXgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int SSgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSSgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSSgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int SHgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSHgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSHgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int SBgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSBgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSBgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public int SXgels(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace,
                CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSXgels(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSXgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }
        /*******************************************************************************/

        /*******************************************************************************//*
		 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels_bufferSize 
		 * API prototypes */
        /*******************************************************************************/
        public SizeT Gels_bufferSizeZZ(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZZgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZZgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeZC(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZCgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZCgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeZK(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZKgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZKgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeZE(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZEgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZEgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeZY(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuDoubleComplex> dA, int ldda,
                CudaDeviceVariable<cuDoubleComplex> dB, int lddb,
                CudaDeviceVariable<cuDoubleComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZYgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZYgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeCC(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCCgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCCgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeCK(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCKgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCKgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeCE(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCEgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCEgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeCY(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<cuFloatComplex> dA, int ldda,
                CudaDeviceVariable<cuFloatComplex> dB, int lddb,
                CudaDeviceVariable<cuFloatComplex> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCYgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCYgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeDD(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDDgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDDgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeDS(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDSgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDSgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeDH(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDHgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDHgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeDB(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDBgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDBgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeDX(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<double> dA, int ldda,
                CudaDeviceVariable<double> dB, int lddb,
                CudaDeviceVariable<double> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDXgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDXgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeSS(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSSgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSSgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeSH(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSHgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSHgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeSB(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSBgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSBgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }

        public SizeT Gels_bufferSizeSX(
                int m,
                int n,
                int nrhs,
                CudaDeviceVariable<float> dA, int ldda,
                CudaDeviceVariable<float> dB, int lddb,
                CudaDeviceVariable<float> dX, int lddx,
                CudaDeviceVariable<byte> dWorkspace)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSXgels_bufferSize(_handle, m, n, nrhs, dA.DevicePointer, ldda,
                dB.DevicePointer, lddb, dX.DevicePointer, lddx, dWorkspace.DevicePointer, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSXgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        /*******************************************************************************/
        #endregion

        #region expert users API for IRS Prototypes
        public SizeT IRSXgesv_bufferSize(IRSParams gesv_irs_params, int n, int nrhs)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnIRSXgesv_bufferSize(_handle, gesv_irs_params.Params, n, nrhs, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSXgesv_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }
        public int IRSXgesv(int n, int nrhs,
            IRSParams gesv_irs_params,
            IRSInfos gesv_irs_infos,
            CUdeviceptr dA, int ldda,
            CUdeviceptr dB, int lddb,
            CUdeviceptr dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnIRSXgesv(_handle, gesv_irs_params.Params, gesv_irs_infos.Infos, n, nrhs, dA, ldda,
                dB, lddb, dX, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSXgesv", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        public SizeT IRSXgesv_bufferSize(IRSParams gesv_irs_params, int m, int n, int nrhs)
        {
            SizeT Lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnIRSXgels_bufferSize(_handle, gesv_irs_params.Params, m, n, nrhs, ref Lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSXgels_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return Lwork;
        }


        public int IRSXgels(int m, int n, int nrhs,
            IRSParams gesv_irs_params,
            IRSInfos gesv_irs_infos,
            CUdeviceptr dA, int ldda,
            CUdeviceptr dB, int lddb,
            CUdeviceptr dX, int lddx,
            CudaDeviceVariable<byte> dWorkspace, CudaDeviceVariable<int> d_info)
        {
            int iter = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnIRSXgels(_handle, gesv_irs_params.Params, gesv_irs_infos.Infos, m, n, nrhs, dA, ldda,
                dB, lddb, dX, lddx, dWorkspace.DevicePointer, dWorkspace.Size, ref iter, d_info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSXgels", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return iter;
        }

        #endregion

        #endregion

        #region batched Cholesky factorization and its solver
        public void SpotrfBatched(FillMode uplo, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSpotrfBatched(_handle, uplo, n, Aarray.DevicePointer, lda, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSpotrfBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void DpotrfBatched(FillMode uplo, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDpotrfBatched(_handle, uplo, n, Aarray.DevicePointer, lda, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDpotrfBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void CpotrfBatched(FillMode uplo, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCpotrfBatched(_handle, uplo, n, Aarray.DevicePointer, lda, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCpotrfBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void ZpotrfBatched(FillMode uplo, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZpotrfBatched(_handle, uplo, n, Aarray.DevicePointer, lda, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZpotrfBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void SpotrsBatched(FillMode uplo, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSpotrsBatched(_handle, uplo, n, nrhs, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSpotrsBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void DpotrsBatched(FillMode uplo, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDpotrsBatched(_handle, uplo, n, nrhs, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDpotrsBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void CpotrsBatched(FillMode uplo, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCpotrsBatched(_handle, uplo, n, nrhs, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCpotrsBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void ZpotrsBatched(FillMode uplo, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb, CudaDeviceVariable<int> infoArray)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZpotrsBatched(_handle, uplo, n, nrhs, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, infoArray.DevicePointer, Aarray.Size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZpotrsBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        #endregion

        #region s.p.d. matrix inversion (POTRI) and auxiliary routines (TRTRI and LAUUM)
        public int potri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSpotri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSpotri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int potri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDpotri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDpotri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int potri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCpotri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCpotri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int potri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZpotri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZpotri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }


        public void potri(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSpotri(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSpotri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void potri(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDpotri(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDpotri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void potri(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCpotri(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCpotri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void potri(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZpotri(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZpotri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }



        public void XsytrsBufferSize(
                FillMode uplo,
                DiagType diag,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXtrtri_bufferSize(_handle, uplo, diag, n, dataTypeA, A, lda, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXtrtri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void Xsytrs(
                FillMode uplo,
                DiagType diag,
                long n,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                CudaDeviceVariable<byte> work, byte[] workHost,
                CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXtrtri(_handle, uplo, diag, n, dataTypeA, A, lda,
                work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXtrtri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        //public int trtri_bufferSize(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<float> A, int lda)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnStrtri_bufferSize(_handle, uplo, diag, n, A.DevicePointer, lda, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnStrtri_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}
        //public int trtri_bufferSize(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<double> A, int lda)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnDtrtri_bufferSize(_handle, uplo, diag, n, A.DevicePointer, lda, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDtrtri_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}
        //public int trtri_bufferSize(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<cuFloatComplex> A, int lda)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnCtrtri_bufferSize(_handle, uplo, diag, n, A.DevicePointer, lda, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCtrtri_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}
        //public int trtri_bufferSize(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnZtrtri_bufferSize(_handle, uplo, diag, n, A.DevicePointer, lda, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZtrtri_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}

        //public void trtri(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> work, CudaDeviceVariable<int> devInfo)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnStrtri(_handle, uplo, diag, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnStrtri", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}
        //public void trtri(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> work, CudaDeviceVariable<int> devInfo)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnDtrtri(_handle, uplo, diag, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDtrtri", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}
        //public void trtri(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> devInfo)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnCtrtri(_handle, uplo, diag, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCtrtri", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}
        //public void trtri(FillMode uplo, DiagType diag, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> devInfo)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnZtrtri(_handle, uplo, diag, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZtrtri", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}

        #endregion

        #region lauum, auxiliar routine for s.p.d matrix inversion

        public int lauum_bufferSize(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSlauum_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSlauum_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int lauum_bufferSize(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDlauum_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDlauum_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int lauum_bufferSize(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnClauum_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnClauum_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int lauum_bufferSize(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZlauum_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZlauum_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }


        public void lauum(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSlauum(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSlauum", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void lauum(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDlauum(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDlauum", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void lauum(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnClauum(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnClauum", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void lauum(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> devInfo)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDlauum(_handle, uplo, n, A.DevicePointer, lda, work.DevicePointer, work.Size, devInfo.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDlauum", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        #endregion

        #region Symmetric indefinite solve (SYTRS)

        public void XsytrsBufferSize(
                FillMode uplo,
                long n,
                long nrhs,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                CudaDeviceVariable<long> ipiv,
                cudaDataType dataTypeB,
                CUdeviceptr B,
                long ldb,
                ref SizeT workspaceInBytesOnDevice,
                ref SizeT workspaceInBytesOnHost)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXsytrs_bufferSize(_handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv.DevicePointer, dataTypeB, B, ldb, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsytrs_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void Xsytrs(
                FillMode uplo,
                long n,
                long nrhs,
                cudaDataType dataTypeA,
                CUdeviceptr A,
                long lda,
                CudaDeviceVariable<long> ipiv,
                cudaDataType dataTypeB,
                CUdeviceptr B,
                long ldb,
                CudaDeviceVariable<byte> work, byte[] workHost,
                CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXsytrs(_handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv.DevicePointer, dataTypeB, B, ldb,
                work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsytrs", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        //public int sytrs_bufferSize(FillMode uplo, int n, int nrhs, CudaDeviceVariable<float> A, int lda,
        //    CudaDeviceVariable<int> ipiv, CudaDeviceVariable<float> B, int ldb)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnSsytrs_bufferSize(_handle, uplo, n, nrhs, A.DevicePointer, lda,
        //        ipiv.DevicePointer, B.DevicePointer, ldb, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsytrs_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}
        //public int sytrs_bufferSize(FillMode uplo, int n, int nrhs, CudaDeviceVariable<double> A, int lda,
        //    CudaDeviceVariable<int> ipiv, CudaDeviceVariable<double> B, int ldb)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnDsytrs_bufferSize(_handle, uplo, n, nrhs, A.DevicePointer, lda,
        //        ipiv.DevicePointer, B.DevicePointer, ldb, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsytrs_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}
        //public int sytrs_bufferSize(FillMode uplo, int n, int nrhs, CudaDeviceVariable<cuFloatComplex> A, int lda,
        //    CudaDeviceVariable<int> ipiv, CudaDeviceVariable<cuFloatComplex> B, int ldb)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnCsytrs_bufferSize(_handle, uplo, n, nrhs, A.DevicePointer, lda,
        //        ipiv.DevicePointer, B.DevicePointer, ldb, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCsytrs_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}
        //public int sytrs_bufferSize(FillMode uplo, int n, int nrhs, CudaDeviceVariable<cuDoubleComplex> A, int lda,
        //    CudaDeviceVariable<int> ipiv, CudaDeviceVariable<cuDoubleComplex> B, int ldb)
        //{
        //    int lwork = 0;
        //    res = CudaSolveNativeMethods.Dense.cusolverDnZsytrs_bufferSize(_handle, uplo, n, nrhs, A.DevicePointer, lda,
        //        ipiv.DevicePointer, B.DevicePointer, ldb, ref lwork);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZsytrs_bufferSize", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //    return lwork;
        //}

        //public void sytrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<int> ipiv,
        //    CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> work, CudaDeviceVariable<int> info)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnSsytrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, ipiv.DevicePointer,
        //        B.DevicePointer, ldb, work.DevicePointer, work.Size, info.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsytrs", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}

        //public void sytrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<int> ipiv,
        //    CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> work, CudaDeviceVariable<int> info)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnDsytrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, ipiv.DevicePointer,
        //        B.DevicePointer, ldb, work.DevicePointer, work.Size, info.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsytrs", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}

        //public void sytrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<int> ipiv,
        //    CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnCsytrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, ipiv.DevicePointer,
        //        B.DevicePointer, ldb, work.DevicePointer, work.Size, info.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCsytrs", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}

        //public void sytrs(FillMode uplo, int n, int nrhs, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<int> ipiv,
        //    CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info)
        //{
        //    res = CudaSolveNativeMethods.Dense.cusolverDnZsytrs(_handle, uplo, n, nrhs, A.DevicePointer, lda, ipiv.DevicePointer,
        //        B.DevicePointer, ldb, work.DevicePointer, work.Size, info.DevicePointer);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZsytrs", res));
        //    if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        //}

        #endregion

        #region Symmetric indefinite inversion (sytri)
        public int sytri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<int> ipiv)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsytri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsytri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sytri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<int> ipiv)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsytri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsytri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sytri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<int> ipiv)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCsytri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCsytri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sytri_bufferSize(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<int> ipiv)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZsytri_bufferSize(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZsytri_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }



        public void sytrs(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<int> ipiv,
            CudaDeviceVariable<float> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSsytri(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer,
                work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsytri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void sytrs(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<int> ipiv,
            CudaDeviceVariable<double> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDsytri(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer,
                work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsytri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void sytrs(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<int> ipiv,
            CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCsytri(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer,
                work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCsytri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void sytrs(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<int> ipiv,
            CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZsytri(_handle, uplo, n, A.DevicePointer, lda, ipiv.DevicePointer,
                work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZsytri", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        #endregion

        #region standard selective symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer
        public int syevdx_bufferSize(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            float vl, float vu, int il, int iu, ref int meig, CudaDeviceVariable<float> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsyevdx_bufferSize(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsyevdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevdx_bufferSize(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            double vl, double vu, int il, int iu, ref int meig, CudaDeviceVariable<double> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsyevdx_bufferSize(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsyevdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevdx_bufferSize(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            float vl, float vu, int il, int iu, ref int meig, CudaDeviceVariable<float> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCheevdx_bufferSize(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCheevdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevdx_bufferSize(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            double vl, double vu, int il, int iu, ref int meig, CudaDeviceVariable<double> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZheevdx_bufferSize(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZheevdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }



        public int syevdx(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            float vl, float vu, int il, int iu, CudaDeviceVariable<float> W, CudaDeviceVariable<float> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsyevdx(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsyevdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }
        public int syevdx(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            double vl, double vu, int il, int iu, CudaDeviceVariable<double> W, CudaDeviceVariable<double> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsyevdx(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsyevdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }
        public int syevdx(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            float vl, float vu, int il, int iu, CudaDeviceVariable<float> W, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCheevdx(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCheevdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }
        public int syevdx(cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            double vl, double vu, int il, int iu, CudaDeviceVariable<double> W, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZheevdx(_handle, jobz, range, uplo, n, A.DevicePointer, lda,
                vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZheevdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }
        #endregion

        #region selective generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer

        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> B, int ldb, float vl, float vu, int il, int iu, ref int meig, CudaDeviceVariable<float> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsygvdx_bufferSize(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsygvdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> B, int ldb, double vl, double vu, int il, int iu, ref int meig, CudaDeviceVariable<double> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsygvdx_bufferSize(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsygvdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<cuFloatComplex> B, int ldb, float vl, float vu, int il, int iu, ref int meig, CudaDeviceVariable<float> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnChegvdx_bufferSize(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnChegvdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<cuDoubleComplex> B, int ldb, double vl, double vu, int il, int iu, ref int meig, CudaDeviceVariable<double> W)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZhegvdx_bufferSize(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, ref lwork);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZhegvdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }


        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> B, int ldb, float vl, float vu, int il, int iu,
            CudaDeviceVariable<float> W, CudaDeviceVariable<float> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsygvdx(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsygvdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }

        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> B, int ldb, double vl, double vu, int il, int iu,
            CudaDeviceVariable<double> W, CudaDeviceVariable<double> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsygvdx(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsygvdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }
        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<cuFloatComplex> B, int ldb, float vl, float vu, int il, int iu,
            CudaDeviceVariable<float> W, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnChegvdx(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnChegvdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }

        public int sygvdx_bufferSize(cusolverEigType itype, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<cuDoubleComplex> B, int ldb, double vl, double vu, int il, int iu,
            CudaDeviceVariable<double> W, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info)
        {
            int meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZhegvdx(_handle, itype, jobz, range, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, vl, vu, il, iu, ref meig, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZhegvdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return meig;
        }
        #endregion

        #region MyRegion

        public double syevjGetResidual(SyevjInfo info)
        {
            double residual = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnXsyevjGetResidual(_handle, info.Info, ref residual);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsyevjGetResidual", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return residual;
        }
        public int syevjGetSweeps(SyevjInfo info)
        {
            int executed_sweeps = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnXsyevjGetSweeps(_handle, info.Info, ref executed_sweeps);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsyevjGetSweeps", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return executed_sweeps;
        }


        public int syevjBatched_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> W, SyevjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsyevjBatched_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsyevjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevjBatched_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> W, SyevjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsyevjBatched_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsyevjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevjBatched_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> W, SyevjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCheevjBatched_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCheevjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevjBatched_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> W, SyevjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZheevjBatched_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZheevjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }

        public void syevjBatched(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> W, CudaDeviceVariable<float> work, CudaDeviceVariable<int> info, SyevjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSsyevjBatched(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsyevjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void syevjBatched(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> W, CudaDeviceVariable<double> work, CudaDeviceVariable<int> info, SyevjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDsyevjBatched(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsyevjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void syevjBatched(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> W, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info, SyevjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCheevjBatched(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCheevjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void syevjBatched(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> W, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info, SyevjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZheevjBatched(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZheevjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }



        public int syevj_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsyevj_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsyevj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevj_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsyevj_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsyevj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevj_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCheevj_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCheevj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int syevj_bufferSize(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZheevj_bufferSize(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZheevj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }



        public void syevj(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> W, CudaDeviceVariable<float> work, CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSsyevj(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsyevj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void syevj(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> W, CudaDeviceVariable<double> work, CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDsyevj(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsyevj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void syevj(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> W, CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCheevj(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCheevj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void syevj(cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> W, CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZheevj(_handle, jobz, uplo, n, A.DevicePointer, lda,
                W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZheevj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public int sygvj_bufferSize(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSsygvj_bufferSize(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsygvj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sygvj_bufferSize(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDsygvj_bufferSize(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsygvj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sygvj_bufferSize(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<float> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnChegvj_bufferSize(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnChegvj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int sygvj_bufferSize(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<double> W, SyevjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZhegvj_bufferSize(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZhegvj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }


        public void sygvj(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> W, CudaDeviceVariable<float> work,
            CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSsygvj(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSsygvj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void sygvj(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> W, CudaDeviceVariable<double> work,
            CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDsygvj(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDsygvj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void sygvj(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<float> W, CudaDeviceVariable<cuFloatComplex> work,
            CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnChegvj(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnChegvj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void sygvj(cusolverEigType itype, cusolverEigMode jobz, FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<double> W, CudaDeviceVariable<cuDoubleComplex> work,
            CudaDeviceVariable<int> info, SyevjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZhegvj(_handle, itype, jobz, uplo, n, A.DevicePointer, lda,
                B.DevicePointer, ldb, W.DevicePointer, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZhegvj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        #endregion

        #region MyRegion


        public double gesvdjGetResidual(GesvdjInfo info)
        {
            double residual = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvdjGetResidual(_handle, info.Info, ref residual);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvdjGetResidual", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return residual;
        }
        public int gesvdjGetSweeps(GesvdjInfo info)
        {
            int executed_sweeps = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvdjGetSweeps(_handle, info.Info, ref executed_sweeps);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvdjGetSweeps", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return executed_sweeps;
        }



        public int gesvdjBatched_bufferSize(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<float> U, int ldu, CudaDeviceVariable<float> V, int ldv,
            GesvdjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSgesvdjBatched_bufferSize(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvdjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdjBatched_bufferSize(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<double> U, int ldu, CudaDeviceVariable<double> V, int ldv,
            GesvdjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDgesvdjBatched_bufferSize(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvdjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdjBatched_bufferSize(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<cuFloatComplex> U, int ldu, CudaDeviceVariable<cuFloatComplex> V, int ldv,
            GesvdjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCgesvdjBatched_bufferSize(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgesvdjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdjBatched_bufferSize(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<cuDoubleComplex> U, int ldu, CudaDeviceVariable<cuDoubleComplex> V, int ldv,
            GesvdjInfo parameters, int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZgesvdjBatched_bufferSize(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvdjBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }

        public void gesvdjBatched(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<float> U, int ldu, CudaDeviceVariable<float> V, int ldv,
            CudaDeviceVariable<float> work, CudaDeviceVariable<int> info, GesvdjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSgesvdjBatched(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvdjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void gesvdjBatched(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<double> U, int ldu, CudaDeviceVariable<double> V, int ldv,
            CudaDeviceVariable<double> work, CudaDeviceVariable<int> info, GesvdjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDgesvdjBatched(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvdjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        public void gesvdjBatched(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<cuFloatComplex> U, int ldu, CudaDeviceVariable<cuFloatComplex> V, int ldv,
            CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info, GesvdjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCgesvdjBatched(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvdjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void gesvdjBatched(cusolverEigMode jobz, int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<cuDoubleComplex> U, int ldu, CudaDeviceVariable<cuDoubleComplex> V, int ldv,
            CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info, GesvdjInfo parameters, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZgesvdjBatched(_handle, jobz, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvdjBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }



        public int gesvdj_bufferSize(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<float> U, int ldu, CudaDeviceVariable<float> V, int ldv,
            GesvdjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSgesvdj_bufferSize(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvdj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdj_bufferSize(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<double> U, int ldu, CudaDeviceVariable<double> V, int ldv,
            GesvdjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDgesvdj_bufferSize(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvdj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdj_bufferSize(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<cuFloatComplex> U, int ldu, CudaDeviceVariable<cuFloatComplex> V, int ldv,
            GesvdjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCgesvdj_bufferSize(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgesvdj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdj_bufferSize(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<cuDoubleComplex> U, int ldu, CudaDeviceVariable<cuDoubleComplex> V, int ldv,
            GesvdjInfo parameters)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZgesvdj_bufferSize(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, ref lwork, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvdj_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }

        public void gesvdj(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<float> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<float> U, int ldu, CudaDeviceVariable<float> V, int ldv,
            CudaDeviceVariable<float> work, CudaDeviceVariable<int> info, GesvdjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSgesvdj(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvdj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void gesvdj(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<double> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<double> U, int ldu, CudaDeviceVariable<double> V, int ldv,
            CudaDeviceVariable<double> work, CudaDeviceVariable<int> info, GesvdjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDgesvdj(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvdj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void gesvdj(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
            CudaDeviceVariable<float> S, CudaDeviceVariable<cuFloatComplex> U, int ldu, CudaDeviceVariable<cuFloatComplex> V, int ldv,
            CudaDeviceVariable<cuFloatComplex> work, CudaDeviceVariable<int> info, GesvdjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCgesvdj(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgesvdj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void gesvdj(cusolverEigMode jobz, int econ, int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
            CudaDeviceVariable<double> S, CudaDeviceVariable<cuDoubleComplex> U, int ldu, CudaDeviceVariable<cuDoubleComplex> V, int ldv,
            CudaDeviceVariable<cuDoubleComplex> work, CudaDeviceVariable<int> info, GesvdjInfo parameters)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZgesvdj(_handle, jobz, econ, m, n, A.DevicePointer, lda,
                S.DevicePointer, U.DevicePointer, ldu, V.DevicePointer, ldv, work.DevicePointer, work.Size, info.DevicePointer, parameters.Info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvdj", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        #endregion

        #region batched approximate SVD

        public int gesvdaStridedBatched_bufferSize(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<float> d_A, int lda, long strideA,
            CudaDeviceVariable<float> d_S, long strideS,
            CudaDeviceVariable<float> d_U, int ldu, long strideU,
            CudaDeviceVariable<float> d_V, int ldv, long strideV,
            int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSgesvdaStridedBatched_bufferSize(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV, ref lwork, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvdaStridedBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdaStridedBatched_bufferSize(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<double> d_A, int lda, long strideA,
            CudaDeviceVariable<double> d_S, long strideS,
            CudaDeviceVariable<double> d_U, int ldu, long strideU,
            CudaDeviceVariable<double> d_V, int ldv, long strideV,
            int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnDgesvdaStridedBatched_bufferSize(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV, ref lwork, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvdaStridedBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdaStridedBatched_bufferSize(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<cuFloatComplex> d_A, int lda, long strideA,
            CudaDeviceVariable<float> d_S, long strideS,
            CudaDeviceVariable<cuFloatComplex> d_U, int ldu, long strideU,
            CudaDeviceVariable<cuFloatComplex> d_V, int ldv, long strideV,
            int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnCgesvdaStridedBatched_bufferSize(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV, ref lwork, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgesvdaStridedBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }
        public int gesvdaStridedBatched_bufferSize(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<cuDoubleComplex> d_A, int lda, long strideA,
            CudaDeviceVariable<double> d_S, long strideS,
            CudaDeviceVariable<cuDoubleComplex> d_U, int ldu, long strideU,
            CudaDeviceVariable<cuDoubleComplex> d_V, int ldv, long strideV,
            int batchSize)
        {
            int lwork = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnZgesvdaStridedBatched_bufferSize(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV, ref lwork, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvdaStridedBatched_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return lwork;
        }



        public void gesvdaStridedBatched(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<float> d_A, int lda, long strideA,
            CudaDeviceVariable<float> d_S, long strideS,
            CudaDeviceVariable<float> d_U, int ldu, long strideU,
            CudaDeviceVariable<float> d_V, int ldv, long strideV,
            CudaDeviceVariable<float> d_work,
            CudaDeviceVariable<int> d_info, double[] h_R_nrmF, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSgesvdaStridedBatched(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV,
                d_work.DevicePointer, d_work.Size, d_info.DevicePointer, h_R_nrmF, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSgesvdaStridedBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void gesvdaStridedBatched(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<double> d_A, int lda, long strideA,
            CudaDeviceVariable<double> d_S, long strideS,
            CudaDeviceVariable<double> d_U, int ldu, long strideU,
            CudaDeviceVariable<double> d_V, int ldv, long strideV,
            CudaDeviceVariable<double> d_work,
            CudaDeviceVariable<int> d_info, double[] h_R_nrmF, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnDgesvdaStridedBatched(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV,
                d_work.DevicePointer, d_work.Size, d_info.DevicePointer, h_R_nrmF, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnDgesvdaStridedBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void gesvdaStridedBatched(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<cuFloatComplex> d_A, int lda, long strideA,
            CudaDeviceVariable<float> d_S, long strideS,
            CudaDeviceVariable<cuFloatComplex> d_U, int ldu, long strideU,
            CudaDeviceVariable<cuFloatComplex> d_V, int ldv, long strideV,
            CudaDeviceVariable<cuFloatComplex> d_work,
            CudaDeviceVariable<int> d_info, double[] h_R_nrmF, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnCgesvdaStridedBatched(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV,
                d_work.DevicePointer, d_work.Size, d_info.DevicePointer, h_R_nrmF, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnCgesvdaStridedBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        public void gesvdaStridedBatched(cusolverEigMode jobz, int rank, int m, int n,
            CudaDeviceVariable<cuDoubleComplex> d_A, int lda, long strideA,
            CudaDeviceVariable<double> d_S, long strideS,
            CudaDeviceVariable<cuDoubleComplex> d_U, int ldu, long strideU,
            CudaDeviceVariable<cuDoubleComplex> d_V, int ldv, long strideV,
            CudaDeviceVariable<cuDoubleComplex> d_work,
            CudaDeviceVariable<int> d_info, double[] h_R_nrmF, int batchSize)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnZgesvdaStridedBatched(_handle, jobz, rank, m, n, d_A.DevicePointer, lda, strideA,
                d_S.DevicePointer, strideS, d_U.DevicePointer, ldu, strideU, d_V.DevicePointer, ldv, strideV,
                d_work.DevicePointer, d_work.Size, d_info.DevicePointer, h_R_nrmF, batchSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnZgesvdaStridedBatched", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        #endregion

        #region 64Bit
        [Obsolete("Deprecated in Cuda version 11.1")]
        public SizeT Potrf_bufferSize(Params parameters,
            FillMode uplo,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType computeType)
        {
            SizeT workspaceInBytes = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnPotrf_bufferSize(_handle, parameters.ParamsHandle, uplo, n, dataTypeA, A, lda, computeType, ref workspaceInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnPotrf_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return workspaceInBytes;
        }
        [Obsolete("Deprecated in Cuda version 11.1")]
        public SizeT Geqrf_bufferSize(Params parameters,
            long m,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType dataTypeTau,
            CUdeviceptr tau,
            cudaDataType computeType)
        {
            SizeT workspaceInBytes = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnGeqrf_bufferSize(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, ref workspaceInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnGeqrf_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return workspaceInBytes;
        }
        [Obsolete("Deprecated in Cuda version 11.1")]
        public SizeT Getrf_bufferSize(Params parameters,
            long m,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType computeType)
        {
            SizeT workspaceInBytes = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnGetrf_bufferSize(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, computeType, ref workspaceInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnGetrf_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return workspaceInBytes;
        }
        [Obsolete("Deprecated in Cuda version 11.1")]
        public SizeT Syevd_bufferSize(Params parameters,
            cusolverEigMode jobz,
            FillMode uplo,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType dataTypeW,
            CUdeviceptr W,
            cudaDataType computeType)
        {
            SizeT workspaceInBytes = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSyevd_bufferSize(_handle, parameters.ParamsHandle, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, ref workspaceInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSyevd_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return workspaceInBytes;
        }

        [Obsolete("Deprecated in Cuda version 11.1")]
        public SizeT Syevdx_bufferSize(Params parameters,
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
            cudaDataType computeType)
        {
            SizeT workspaceInBytes = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSyevdx_bufferSize(_handle, parameters.ParamsHandle, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, ref h_meig, dataTypeW, W, computeType, ref workspaceInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSyevdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return workspaceInBytes;
        }






        [Obsolete("Deprecated in Cuda version 11.1")]
        public void Potrf(Params parameters, FillMode uplo, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            cudaDataType computeType, CudaDeviceVariable<byte> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnPotrf(_handle, parameters.ParamsHandle, uplo, n, dataTypeA, A, lda, computeType, work.DevicePointer, work.SizeInBytes, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnPotrf", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        [Obsolete("Deprecated in Cuda version 11.1")]
        public void Potrs(Params parameters, FillMode uplo, long n, long nrhs, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            cudaDataType dataTypeB, CUdeviceptr B, long ldb, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnPotrs(_handle, parameters.ParamsHandle, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnPotrs", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        [Obsolete("Deprecated in Cuda version 11.1")]
        public void Geqrf(Params parameters, long m, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            cudaDataType dataTypeTau, CUdeviceptr tau, cudaDataType computeType, CudaDeviceVariable<byte> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnGeqrf(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, work.DevicePointer, work.SizeInBytes, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnGeqrf", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        [Obsolete("Deprecated in Cuda version 11.1")]
        public void Getrf(Params parameters, long m, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            CudaDeviceVariable<long> ipiv, cudaDataType computeType, CudaDeviceVariable<byte> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnGetrf(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, ipiv.DevicePointer, computeType, work.DevicePointer, work.SizeInBytes, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnGetrf", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        [Obsolete("Deprecated in Cuda version 11.1")]
        public void Getrs(Params parameters, Operation trans, long n, long nrhs, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            CudaDeviceVariable<long> ipiv, cudaDataType dataTypeB, CUdeviceptr B, long ldb, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnGetrs(_handle, parameters.ParamsHandle, trans, n, nrhs, dataTypeA, A, lda, ipiv.DevicePointer, dataTypeB, B, ldb, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnGetrs", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        [Obsolete("Deprecated in Cuda version 11.1")]
        public void Syevd(Params parameters, cusolverEigMode jobz, FillMode uplo, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            cudaDataType dataTypeW, CUdeviceptr W, cudaDataType computeType, CudaDeviceVariable<byte> work, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnSyevd(_handle, parameters.ParamsHandle, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, work.DevicePointer, work.SizeInBytes, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSyevd", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        [Obsolete("Deprecated in Cuda version 11.1")]
        public long Syevdx(Params parameters, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            IntPtr vl, IntPtr vu, long il, long iu, cudaDataType dataTypeW, CUdeviceptr W,
            cudaDataType computeType, CudaDeviceVariable<byte> work, CudaDeviceVariable<int> info)
        {
            long h_meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnSyevdx(_handle, parameters.ParamsHandle, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu,
                ref h_meig, dataTypeW, W, computeType, work.DevicePointer, work.SizeInBytes, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnSyevdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return h_meig;
        }

        /* 64-bit API for SYEVDX */



        #endregion

        #region new 64-bit API
        #region 64-bit API for POTRF


        public void XpotrfBufferSize(
            Params parameters,
            FillMode uplo,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType computeType,
            ref SizeT workspaceInBytesOnDevice,
            ref SizeT workspaceInBytesOnHost)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXpotrf_bufferSize(_handle, parameters.ParamsHandle, uplo, n, dataTypeA, A, lda, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXpotrf_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void Xpotrf(
            Params parameters, FillMode uplo, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
             cudaDataType computeType, CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXpotrf(_handle, parameters.ParamsHandle, uplo, n, dataTypeA, A, lda, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXpotrf", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        /* 64-bit API for POTRS */

        public void Xpotrs(Params parameters, FillMode uplo, long n, long nrhs, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            cudaDataType dataTypeB, CUdeviceptr B, long ldb, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXpotrs(_handle, parameters.ParamsHandle, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXpotrs", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        /* 64-bit API for GEQRF */

        public void XgeqrfBufferSize(Params parameters,
            long m,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType dataTypeTau,
            CUdeviceptr tau,
            cudaDataType computeType,
            ref SizeT workspaceInBytesOnDevice,
            ref SizeT workspaceInBytesOnHost)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXgeqrf_bufferSize(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgeqrf_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void Xgeqrf(Params parameters, long m, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            cudaDataType dataTypeTau, CUdeviceptr tau, cudaDataType computeType, CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXgeqrf(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgeqrf", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        /* 64-bit API for GETRF */

        public void XgetrfBufferSize(
            Params parameters,
            long m,
            long n,
            cudaDataType dataTypeA,
            CUdeviceptr A,
            long lda,
            cudaDataType computeType,
            ref SizeT workspaceInBytesOnDevice,
            ref SizeT workspaceInBytesOnHost)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXgetrf_bufferSize(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgetrf_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void Xgetrf(Params parameters, long m, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            CudaDeviceVariable<long> ipiv, cudaDataType computeType, CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXgetrf(_handle, parameters.ParamsHandle, m, n, dataTypeA, A, lda, ipiv.DevicePointer, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgetrf", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        /* 64-bit API for GETRS */

        public void Xgetrs(Params parameters, Operation trans, long n, long nrhs, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            CudaDeviceVariable<long> ipiv, cudaDataType dataTypeB, CUdeviceptr B, long ldb, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXgetrs(_handle, parameters.ParamsHandle, trans, n, nrhs, dataTypeA, A, lda, ipiv.DevicePointer, dataTypeB, B, ldb, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgetrs", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        /* 64-bit API for SYEVD */

        public void XsyevdBufferSize(
                Params parameters,
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
                ref SizeT workspaceInBytesOnHost)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXsyevd_bufferSize(_handle, parameters.ParamsHandle, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsyevd_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void Xsyevd(Params parameters, cusolverEigMode jobz, FillMode uplo, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            cudaDataType dataTypeW, CUdeviceptr W, cudaDataType computeType, CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXsyevd(_handle, parameters.ParamsHandle, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsyevd", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        /* 64-bit API for SYEVDX */

        public void XsyevdxBufferSize(
            Params parameters,
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
            ref SizeT workspaceInBytesOnHost)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXsyevdx_bufferSize(_handle, parameters.ParamsHandle, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, ref h_meig, dataTypeW, W, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsyevdx_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public long Xsyevdx(Params parameters, cusolverEigMode jobz, cusolverEigRange range, FillMode uplo, long n, cudaDataType dataTypeA, CUdeviceptr A, long lda,
            IntPtr vl, IntPtr vu, long il, long iu, cudaDataType dataTypeW, CUdeviceptr W,
            cudaDataType computeType, CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            long h_meig = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnXsyevdx(_handle, parameters.ParamsHandle, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu,
                ref h_meig, dataTypeW, W, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXsyevdx", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return h_meig;
        }

        /* 64-bit API for GESVD */

        public void XgesvdBufferSize(
            Params parameters,
            char jobu,
            char jobvt,
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
            ref SizeT workspaceInBytesOnHost)
        {
            sbyte jobusb = Convert.ToSByte(jobu);
            sbyte jobvtsb = Convert.ToSByte(jobvt);
            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvd_bufferSize(_handle, parameters.ParamsHandle, jobusb, jobvtsb, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U,
                ldu, dataTypeVT, VT, ldvt, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvd_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void Xgesvd(
            Params parameters,
            char jobu,
            char jobvt,
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
            CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            sbyte jobusb = Convert.ToSByte(jobu);
            sbyte jobvtsb = Convert.ToSByte(jobvt);
            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvd(_handle, parameters.ParamsHandle, jobusb, jobvtsb, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U,
                ldu, dataTypeVT, VT, ldvt, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvd", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }

        /* 64-bit API for GESVDP */

        public void XgesvdpBufferSize(
            Params parameters,
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
            ref SizeT workspaceInBytesOnHost)
        {

            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvdp_bufferSize(_handle, parameters.ParamsHandle, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U,
                ldu, dataTypeV, V, ldv, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvdp_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public double Xgesvdp(
            Params parameters,
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
            CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            double h_sigma_err = 0;
            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvdp(_handle, parameters.ParamsHandle, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U,
                ldu, dataTypeV, V, ldv, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer, ref h_sigma_err);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvdp", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
            return h_sigma_err;
        }
        #endregion

        public void XgesvdrBufferSize(
                    Params parameters,
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
            ref SizeT workspaceInBytesOnHost)
        {

            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvdr_bufferSize(_handle, parameters.ParamsHandle, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand,
                ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, ref workspaceInBytesOnDevice, ref workspaceInBytesOnHost);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvdr_bufferSize", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }


        public void Xgesvdr(
                    Params parameters,
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
            CudaDeviceVariable<byte> work, byte[] workHost, CudaDeviceVariable<int> info)
        {
            res = CudaSolveNativeMethods.Dense.cusolverDnXgesvdr(_handle, parameters.ParamsHandle, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand,
                ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, work.DevicePointer, work.SizeInBytes, workHost, workHost.Length, info.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnXgesvdr", res));
            if (res != cusolverStatus.Success) throw new CudaSolveException(res);
        }
        #endregion
    }
}
