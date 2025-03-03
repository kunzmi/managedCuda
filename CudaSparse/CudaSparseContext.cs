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


using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ManagedCuda.CudaSparse
{
    /// <summary>
    /// Wrapper class for cusparseContext. Provides all fundamental API functions as methods.
    /// </summary>
    public class CudaSparseContext : IDisposable
    {
        private cusparseContext _handle;
        private cusparseStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// Creates a new CudaSparseContext
        /// </summary>
        public CudaSparseContext()
        {
            _handle = new cusparseContext();
            res = CudaSparseNativeMethods.cusparseCreate(ref _handle);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreate", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// Creates a new CudaSparseContext and sets the cuda stream to use
        /// </summary>
        /// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
        public CudaSparseContext(CUstream stream)
        {
            _handle = new cusparseContext();
            res = CudaSparseNativeMethods.cusparseCreate(ref _handle);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreate", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            SetStream(stream);
        }

        /// <summary>
        /// For dispose
        /// </summary>
		~CudaSparseContext()
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
                //Ignore if failing
                res = CudaSparseNativeMethods.cusparseDestroy(_handle);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods
        /// <summary>
        /// Sets the cuda stream to use
        /// </summary>
        /// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
        public void SetStream(CUstream stream)
        {
            res = CudaSparseNativeMethods.cusparseSetStream(_handle, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSetStream", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
		/// Gets the cuda stream to use
        /// </summary>
        public CUstream GetStream()
        {
            CUstream stream = new CUstream();
            res = CudaSparseNativeMethods.cusparseGetStream(_handle, ref stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetStream", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return stream;
        }

        /// <summary>
		/// Returns the version of the underlying CUSPARSE library
        /// </summary>
        public Version GetVersion()
        {
            int version = 0;
            res = CudaSparseNativeMethods.cusparseGetVersion(_handle, ref version);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetVersion", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return new Version((int)version / 1000, (int)version % 100);
        }

        /// <summary>
        /// Returns the pointer mode for scalar values (host or device pointer)
        /// </summary>
        /// <returns></returns>
        public cusparsePointerMode GetPointerMode()
        {
            cusparsePointerMode pointerMode = new cusparsePointerMode();
            res = CudaSparseNativeMethods.cusparseGetPointerMode(_handle, ref pointerMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetPointerMode", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return pointerMode;
        }

        /// <summary>
        /// Sets the pointer mode for scalar values (host or device pointer)
        /// </summary>
        /// <param name="pointerMode"></param>
        public void SetPointerMode(cusparsePointerMode pointerMode)
        {
            res = CudaSparseNativeMethods.cusparseSetPointerMode(_handle, pointerMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSetPointerMode", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion

        #region Sparse Level 2 routines





        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, float alpha, CudaDeviceVariable<float> A, int lda, int nnz, CudaDeviceVariable<float> xVal, CudaDeviceVariable<int> xInd,
            float beta, CudaDeviceVariable<float> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSgemvi(_handle, transA, m, n, ref alpha, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, ref beta, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, double alpha, CudaDeviceVariable<double> A, int lda, int nnz, CudaDeviceVariable<double> xVal, CudaDeviceVariable<int> xInd,
            double beta, CudaDeviceVariable<double> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDgemvi(_handle, transA, m, n, ref alpha, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, ref beta, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, int nnz, CudaDeviceVariable<cuFloatComplex> xVal, CudaDeviceVariable<int> xInd,
            cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseCgemvi(_handle, transA, m, n, ref alpha, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, ref beta, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, int nnz, CudaDeviceVariable<cuDoubleComplex> xVal, CudaDeviceVariable<int> xInd,
            cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseZgemvi(_handle, transA, m, n, ref alpha, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, ref beta, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, int nnz, CudaDeviceVariable<float> xVal, CudaDeviceVariable<int> xInd,
            CudaDeviceVariable<float> beta, CudaDeviceVariable<float> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSgemvi(_handle, transA, m, n, alpha.DevicePointer, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, beta.DevicePointer, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, int nnz, CudaDeviceVariable<double> xVal, CudaDeviceVariable<int> xInd,
            CudaDeviceVariable<double> beta, CudaDeviceVariable<double> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDgemvi(_handle, transA, m, n, alpha.DevicePointer, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, beta.DevicePointer, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, int nnz, CudaDeviceVariable<cuFloatComplex> xVal, CudaDeviceVariable<int> xInd,
            CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseCgemvi(_handle, transA, m, n, alpha.DevicePointer, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, beta.DevicePointer, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + B * y <para/>
        /// A is an m x n dense matrix and a sparse vector x that is defined in a sparse storage format
        /// by the two arrays xVal, xInd of length nnz, and y is a dense vector; alpha and beta are scalars.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="A">the pointer to dense matrix A.</param>
        /// <param name="lda">size of the leading dimension of A.</param>
        /// <param name="nnz">number of nonzero elements of vector x.</param>
        /// <param name="xVal">sparse vector of nnz elements of size n if op(A) = A, and of size m if op(A) = A^T or op(A) = A^H</param>
        /// <param name="xInd">Indices of non-zero values in x</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">dense vector of m elements if op(A) = A, and of n elements if op(A) = A^T or op(A) = A^H</param>
        /// <param name="idxBase">0 or 1, for 0 based or 1 based indexing, respectively</param>
        /// <param name="pBuffer">working space buffer, of size given by Xgemvi_getBufferSize()</param>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Gemvi(cusparseOperation transA, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, int nnz, CudaDeviceVariable<cuDoubleComplex> xVal, CudaDeviceVariable<int> xInd,
            CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y, IndexBase idxBase, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseZgemvi(_handle, transA, m, n, alpha.DevicePointer, A.DevicePointer, lda, nnz, xVal.DevicePointer, xInd.DevicePointer, beta.DevicePointer, y.DevicePointer, idxBase, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgemvi", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function returns size of buffer used in gemvi(). A is an (m)x(n) dense matrix.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y.</param>
        /// <param name="nnz">number of nonzero entries of vector x multiplying A.</param>
        /// <returns>number of elements needed the buffer used in gemvi().</returns>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public int GemviSBufferSize(cusparseOperation transA, int m, int n, int nnz)
        {
            int size = 0;
            res = CudaSparseNativeMethods.cusparseSgemvi_bufferSize(_handle, transA, m, n, nnz, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgemvi_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in gemvi(). A is an (m)x(n) dense matrix.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y.</param>
        /// <param name="nnz">number of nonzero entries of vector x multiplying A.</param>
        /// <returns>number of elements needed the buffer used in gemvi().</returns>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public int GemviDBufferSize(cusparseOperation transA, int m, int n, int nnz)
        {
            int size = 0;
            res = CudaSparseNativeMethods.cusparseDgemvi_bufferSize(_handle, transA, m, n, nnz, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgemvi_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in gemvi(). A is an (m)x(n) dense matrix.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y.</param>
        /// <param name="nnz">number of nonzero entries of vector x multiplying A.</param>
        /// <returns>number of elements needed the buffer used in gemvi().</returns>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public int GemviCBufferSize(cusparseOperation transA, int m, int n, int nnz)
        {
            int size = 0;
            res = CudaSparseNativeMethods.cusparseCgemvi_bufferSize(_handle, transA, m, n, nnz, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgemvi_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in gemvi(). A is an (m)x(n) dense matrix.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y.</param>
        /// <param name="nnz">number of nonzero entries of vector x multiplying A.</param>
        /// <returns>number of elements needed the buffer used in gemvi().</returns>
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public int GemviZBufferSize(cusparseOperation transA, int m, int n, int nnz)
        {
            int size = 0;
            res = CudaSparseNativeMethods.cusparseZgemvi_bufferSize(_handle, transA, m, n, nnz, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgemvi_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }




        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXbsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called bsrsv2_analysis() or bsrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Bsrsv2ZeroPivot(CudaSparseBsrsv2Info info, CudaDeviceVariable<int> position)
        {
            res = CudaSparseNativeMethods.cusparseXbsrsv2_zeroPivot(_handle, info.Bsrsv2Info, position.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsrsv2_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }


        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXbsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called bsrsv2_analysis() or bsrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Bsrsv2ZeroPivot(CudaSparseBsrsv2Info info, ref int position)
        {
            res = CudaSparseNativeMethods.cusparseXbsrsv2_zeroPivot(_handle, info.Bsrsv2Info, ref position);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsrsv2_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }

        /// <summary>
        /// This function returns the size of the buffer used in bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsv2BufferSize(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSbsrsv2_bufferSizeExt(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns the size of the buffer used in bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsv2BufferSize(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDbsrsv2_bufferSizeExt(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns the size of the buffer used in bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsv2BufferSize(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCbsrsv2_bufferSizeExt(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns the size of the buffer used in bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsv2BufferSize(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZbsrsv2_bufferSizeExt(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function performs the analysis phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Analysis(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsrsv2_analysis(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsv2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Analysis(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsrsv2_analysis(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsv2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Analysis(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsrsv2_analysis(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsv2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Analysis(cusparseOperation transA, cusparseDirection dirA, int mb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsrsv2_analysis(_handle, dirA, transA, mb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsv2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, ref float alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<float> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<float> y)
        {
            res = CudaSparseNativeMethods.cusparseSbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, ref double alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<double> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<double> y)
        {
            res = CudaSparseNativeMethods.cusparseDbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, ref cuFloatComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<cuFloatComplex> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<cuFloatComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseCbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, ref cuDoubleComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<cuDoubleComplex> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<cuDoubleComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseZbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, CudaDeviceVariable<float> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<float> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<float> y)
        {
            res = CudaSparseNativeMethods.cusparseSbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, CudaDeviceVariable<double> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<double> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<double> y)
        {
            res = CudaSparseNativeMethods.cusparseDbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, CudaDeviceVariable<cuFloatComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<cuFloatComplex> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<cuFloatComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseCbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of bsrsv2, a new sparse triangular
        /// linear system op(A)*y = x.
        /// </summary>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtrA(mb) - bsrRowPtrA(0)) column indices of the nonzero blocks of matrix A. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="blockDim">block dimension of sparse matrix A; must be larger than zero.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="x">right-hand-side vector of size m.</param>
        /// <param name="y">solution vector of size m.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsv2Solve(cusparseOperation transA, cusparseDirection dirA, int mb, CudaDeviceVariable<cuDoubleComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrsv2Info info,
            CudaDeviceVariable<cuDoubleComplex> x, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer, CudaDeviceVariable<cuDoubleComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseZbsrsv2_solve(_handle, dirA, transA, mb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrsv2Info, x.DevicePointer, y.DevicePointer, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsv2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }












        #endregion

        #region Sparse Level 3 routines

        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) is either a structural zero or a numerical zero (singular block). Otherwise
        /// position=-1. <para/>
        /// The position can be 0-base or 1-base, the same as the matrix.
        /// Function cusparseXbsrsm2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done.<para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains a structural zero or a
        /// numerical zero if the user already called bsrsm2_analysis() or bsrsm2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1;
        /// otherwise, if A(j,j) is missing or U(j,j) is zero,
        /// position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Xbsrsm2ZeroPivot(CudaSparseBsrsm2Info info, ref int position)
        {
            res = CudaSparseNativeMethods.cusparseXbsrsm2_zeroPivot(_handle, info.Bsrsm2Info, ref position);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsrsm2_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }

        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) is either a structural zero or a numerical zero (singular block). Otherwise
        /// position=-1. <para/>
        /// The position can be 0-base or 1-base, the same as the matrix.
        /// Function cusparseXbsrsm2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done.<para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains a structural zero or a
        /// numerical zero if the user already called bsrsm2_analysis() or bsrsm2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1;
        /// otherwise, if A(j,j) is missing or U(j,j) is zero,
        /// position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Xbsrsm2ZeroPivot(CudaSparseBsrsm2Info info, CudaDeviceVariable<int> position)
        {
            res = CudaSparseNativeMethods.cusparseXbsrsm2_zeroPivot(_handle, info.Bsrsm2Info, position.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsrsm2_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }

        /// <summary>
        /// This function returns size of buffer used in bsrsm2(), a new sparse triangular linear
        /// system op(A)*Y = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either 
        /// CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <returns>number of bytes of the buffer used in bsrsm2_analysis() and bsrsm2_solve().</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsm2BufferSize(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrVal, CudaDeviceVariable<int> bsrRowPtr,
                                    CudaDeviceVariable<int> bsrColInd, int blockSize, CudaSparseBsrsm2Info info)
        {
            SizeT buffersize = 0;
            res = CudaSparseNativeMethods.cusparseSbsrsm2_bufferSizeExt(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, ref buffersize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsm2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return buffersize;
        }

        /// <summary>
        /// This function returns size of buffer used in bsrsm2(), a new sparse triangular linear
        /// system op(A)*Y = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either 
        /// CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <returns>number of bytes of the buffer used in bsrsm2_analysis() and bsrsm2_solve().</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsm2BufferSize(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrVal, CudaDeviceVariable<int> bsrRowPtr,
                                    CudaDeviceVariable<int> bsrColInd, int blockSize, CudaSparseBsrsm2Info info)
        {
            SizeT buffersize = 0;
            res = CudaSparseNativeMethods.cusparseDbsrsm2_bufferSizeExt(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, ref buffersize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsm2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return buffersize;
        }

        /// <summary>
        /// This function returns size of buffer used in bsrsm2(), a new sparse triangular linear
        /// system op(A)*Y = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either 
        /// CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <returns>number of bytes of the buffer used in bsrsm2_analysis() and bsrsm2_solve().</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsm2BufferSize(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr,
                                    CudaDeviceVariable<int> bsrColInd, int blockSize, CudaSparseBsrsm2Info info)
        {
            SizeT buffersize = 0;
            res = CudaSparseNativeMethods.cusparseCbsrsm2_bufferSizeExt(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, ref buffersize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsm2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return buffersize;
        }

        /// <summary>
        /// This function returns size of buffer used in bsrsm2(), a new sparse triangular linear
        /// system op(A)*Y = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either 
        /// CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <returns>number of bytes of the buffer used in bsrsm2_analysis() and bsrsm2_solve().</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrsm2BufferSize(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr,
                                    CudaDeviceVariable<int> bsrColInd, int blockSize, CudaSparseBsrsm2Info info)
        {
            SizeT buffersize = 0;
            res = CudaSparseNativeMethods.cusparseZbsrsm2_bufferSizeExt(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, ref buffersize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsm2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return buffersize;
        }


        /// <summary>
        /// This function performs the analysis phase of bsrsm2(), a new sparse triangular linear
        /// system op(A)*op(Y) = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are 
        /// CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is return by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Analysis(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrVal,
                                    CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int blockSize,
                                    CudaSparseBsrsm2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsrsm2_analysis(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer,
                blockSize, info.Bsrsm2Info, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsm2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the analysis phase of bsrsm2(), a new sparse triangular linear
        /// system op(A)*op(Y) = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are 
        /// CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is return by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Analysis(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrVal,
                                    CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int blockSize,
                                    CudaSparseBsrsm2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsrsm2_analysis(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer,
                blockSize, info.Bsrsm2Info, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsm2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of bsrsm2(), a new sparse triangular linear
        /// system op(A)*op(Y) = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are 
        /// CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is return by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Analysis(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrVal,
                                    CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int blockSize,
                                    CudaSparseBsrsm2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsrsm2_analysis(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer,
                blockSize, info.Bsrsm2Info, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsm2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of bsrsm2(), a new sparse triangular linear
        /// system op(A)*op(Y) = alpha op(X).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(X).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of matrix Y and op(X).</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL, while the supported diagonal types are CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrVal">array of nnzb bsrRowPtrA(mb) 
        /// bsrRowPtrA(0) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb +1 elements that contains the
        /// start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb (= bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A; larger than 
        /// zero.</param>
        /// <param name="info">record internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are 
        /// CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is return by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Analysis(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY,
                                    int mb, int n, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrVal,
                                    CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int blockSize,
                                    CudaSparseBsrsm2Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsrsm2_analysis(_handle, dirA, transA, transXY, mb, n, nnzb, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer,
                blockSize, info.Bsrsm2Info, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsm2_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }




        #region host
        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        [Obsolete("Deprecated in Cuda 12.8, replace with cusparseSpMM")]
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    float alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<float> B, int ldb, float beta, CudaDeviceVariable<float> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseSbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }

        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        [Obsolete("Deprecated in Cuda 12.8, replace with cusparseSpMM")]
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    double alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<double> B, int ldb, double beta, CudaDeviceVariable<double> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseDbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }
        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        [Obsolete("Deprecated in Cuda 12.8, replace with cusparseSpMM")]
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    cuFloatComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<cuFloatComplex> B, int ldb, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseCbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }
        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        [Obsolete("Deprecated in Cuda 12.8, replace with cusparseSpMM")]
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    cuDoubleComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<cuDoubleComplex> B, int ldb, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseZbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }



        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, float alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<float> X, int ldx, CudaDeviceVariable<float> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, ref alpha, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, double alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<double> X, int ldx, CudaDeviceVariable<double> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, ref alpha, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, cuFloatComplex alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<cuFloatComplex> X, int ldx, CudaDeviceVariable<cuFloatComplex> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, ref alpha, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, cuDoubleComplex alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<cuDoubleComplex> X, int ldx, CudaDeviceVariable<cuDoubleComplex> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, ref alpha, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion

        #region ref device
        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    CudaDeviceVariable<float> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseSbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }

        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    CudaDeviceVariable<double> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseDbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }
        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    CudaDeviceVariable<cuFloatComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseCbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }
        /// <summary>
        /// This function performs one of the following matrix-matrix operations:
        /// C = alpha * op(A) * op(B) + beta * C
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transB">the operation op(B).</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="n">number of columns of dense matrix op(B) and A.</param>
        /// <param name="kb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of non-zero blocks of sparse matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb + 1elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="B">array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.</param>
        /// <param name="ldb">leading dimension of B. If op(B)=B, it must be at least max(l,k) If op(B) != B, it must be at least
        /// max(1, n).</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, C does not have to be a valid input.</param>
        /// <param name="C">array of dimensions (ldc, n).</param>
        /// <param name="ldc">leading dimension of C. It must be at least max(l,m) if op(A)=A and at least max(l,k) otherwise.</param>
        public void Bsrmm(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transB, int mb, int n, int kb, int nnzb,
                                    CudaDeviceVariable<cuDoubleComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                    CudaDeviceVariable<int> bsrColIndA, int blockSize, CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
        {
            res = CudaSparseNativeMethods.cusparseZbsrmm(_handle, dirA, transA, transB, mb, n, kb, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer,
                bsrColIndA.DevicePointer, blockSize, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrmm", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

        }



        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, CudaDeviceVariable<float> alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<float> X, int ldx, CudaDeviceVariable<float> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, CudaDeviceVariable<double> alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<double> X, int ldx, CudaDeviceVariable<double> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, CudaDeviceVariable<cuFloatComplex> alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<cuFloatComplex> X, int ldx, CudaDeviceVariable<cuFloatComplex> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the solve phase of the solution of a sparse triangular linear system:
        /// op(A) * op(Y) = alpha * op(X)
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A).</param>
        /// <param name="transXY">the operation op(x) and op(Y).</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="n">number of columns of dense matrix Y and op(X).</param>
        /// <param name="nnzb">number of non-zero blocks of matrix A</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrVal">array of nnzb ( = bsrRowPtrA(mb) - 
        /// bsrRowPtrA(0) ) nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb + 1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb ( =bsrRowPtrA(mb) -
        /// bsrRowPtrA(0) ) column indices of the nonzero blocks of matrix A.</param>
        /// <param name="blockSize">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="info">structure initialized using cusparseCreateBsrsm2Info().</param>
        /// <param name="X">right-hand-side array.</param>
        /// <param name="ldx">leading dimension of X. If op(X)=X, ldx&gt;=k; otherwise, ldx>=n.</param>
        /// <param name="Y">solution array of dimensions (ldy, n).</param>
        /// <param name="ldy">leading dimension of Y. If op(A)=A, then ldc&gt;=m. If op(A)!=A, then ldx>=k.</param>
        /// <param name="policy">the supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by bsrsm2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrsm2Solve(cusparseDirection dirA, cusparseOperation transA, cusparseOperation transXY, int mb, int n, int nnzb, CudaDeviceVariable<cuDoubleComplex> alpha,
                                            CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                            int blockSize, CudaSparseBsrsm2Info info, CudaDeviceVariable<cuDoubleComplex> X, int ldx, CudaDeviceVariable<cuDoubleComplex> Y, int ldy,
                                            cusparseSolvePolicy policy, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsrsm2_solve(_handle, dirA, transA, transXY, mb, n, nnzb, alpha.DevicePointer, descrA.Descriptor, bsrVal.DevicePointer, bsrRowPtr.DevicePointer,
                    bsrColInd.DevicePointer, blockSize, info.Bsrsm2Info, X.DevicePointer, ldx, Y.DevicePointer, ldy, policy, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrsm2_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion
















        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<float> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseScsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, ref double tol, ref float boost_val)
        {
            res = CudaSparseNativeMethods.cusparseScsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<double> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseDcsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, ref double tol, ref double boost_val)
        {
            res = CudaSparseNativeMethods.cusparseDcsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<cuFloatComplex> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseCcsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, ref double tol, ref cuFloatComplex boost_val)
        {
            res = CudaSparseNativeMethods.cusparseCcsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<cuDoubleComplex> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseZcsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// csrilu02(). To disable a boost value, the user can call csrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateCsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02NumericBoost(csrilu02Info info, int enable_boost, ref double tol, ref cuDoubleComplex boost_val)
        {
            res = CudaSparseNativeMethods.cusparseZcsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXcsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called csrsv2_analysis() or csrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Csrilu02ZeroPivot(CudaSparseCsrilu02Info info, CudaDeviceVariable<int> position)
        {
            res = CudaSparseNativeMethods.cusparseXcsrilu02_zeroPivot(_handle, info.Csrilu02Info, position.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsrilu02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }
        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXcsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called csrsv2_analysis() or csrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Csrilu02ZeroPivot(CudaSparseCsrilu02Info info, ref int position)
        {
            res = CudaSparseNativeMethods.cusparseXcsrilu02_zeroPivot(_handle, info.Csrilu02Info, ref position);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsrilu02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }

        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csrilu02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseScsrilu02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csrilu02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDcsrilu02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csrilu02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCcsrilu02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csrilu02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZcsrilu02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseScsrilu02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsrilu02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsrilu02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsrilu02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseScsrilu02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsrilu02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsrilu02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrilu02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsrilu02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }











        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<float> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseSbsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, ref double tol, ref float boost_val)
        {
            res = CudaSparseNativeMethods.cusparseSbsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<double> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseDbsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, ref double tol, ref double boost_val)
        {
            res = CudaSparseNativeMethods.cusparseDbsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<cuFloatComplex> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseCbsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, ref double tol, ref cuFloatComplex boost_val)
        {
            res = CudaSparseNativeMethods.cusparseCbsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, CudaDeviceVariable<double> tol, CudaDeviceVariable<cuDoubleComplex> boost_val)
        {
            res = CudaSparseNativeMethods.cusparseZbsrilu02_numericBoost(_handle, info, enable_boost, tol.DevicePointer, boost_val.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// The user can use a boost value to replace a numerical value in incomplete LU
        /// factorization. The tol is used to determine a numerical zero, and the boost_val is used
        /// to replace a numerical zero. The behavior is <para/>
        /// if tol >= fabs(A(j,j)), then A(j,j)=boost_val.<para/>
        /// To enable a boost value, the user has to set parameter enable_boost to 1 before calling
        /// bsrilu02(). To disable a boost value, the user can call bsrilu02_numericBoost()
        /// again with parameter enable_boost=0.<para/>
        /// If enable_boost=0, tol and boost_val are ignored.
        /// </summary>
        /// <param name="info">structure initialized using cusparseCreateBsrilu02Info().</param>
        /// <param name="enable_boost">disable boost by enable_boost=0; otherwise, boost is enabled.</param>
        /// <param name="tol">tolerance to determine a numerical zero.</param>
        /// <param name="boost_val">boost value to replace a numerical zero.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02NumericBoost(bsrilu02Info info, int enable_boost, ref double tol, ref cuDoubleComplex boost_val)
        {
            res = CudaSparseNativeMethods.cusparseZbsrilu02_numericBoost(_handle, info, enable_boost, ref tol, ref boost_val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrilu02_numericBoost", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXbsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called bsrsv2_analysis() or bsrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Bsrilu02ZeroPivot(CudaSparseBsrilu02Info info, CudaDeviceVariable<int> position)
        {
            res = CudaSparseNativeMethods.cusparseXbsrilu02_zeroPivot(_handle, info.Bsrilu02Info, position.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsrilu02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }
        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXbsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called bsrsv2_analysis() or bsrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Bsrilu02ZeroPivot(CudaSparseBsrilu02Info info, ref int position)
        {
            res = CudaSparseNativeMethods.cusparseXbsrilu02_zeroPivot(_handle, info.Bsrilu02Info, ref position);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsrilu02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }

        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrilu02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSbsrilu02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrilu02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDbsrilu02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrilu02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCbsrilu02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of the buffer used in computing the incomplete-LU
        /// factorization with fill-in and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsrilu02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZbsrilu02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrilu02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with 0 fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsrilu02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsrilu02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsrilu02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-LU factorization with fillin
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsrilu02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrilu02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsrilu02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsrilu02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsrilu02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-LU factorization with fill-in
        /// and no pivoting: A = LU
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsrilu02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsrilu02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsrilu02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsrilu02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrilu02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }















        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXcsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called csrsv2_analysis() or csrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Csric02ZeroPivot(CudaSparseCsric02Info info, CudaDeviceVariable<int> position)
        {
            res = CudaSparseNativeMethods.cusparseXcsric02_zeroPivot(_handle, info.Csric02Info, position.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsric02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }
        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXcsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called csrsv2_analysis() or csrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Csric02ZeroPivot(CudaSparseCsric02Info info, ref int position)
        {
            res = CudaSparseNativeMethods.cusparseXcsric02_zeroPivot(_handle, info.Csric02Info, ref position);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsric02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }

        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csric02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseScsric02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csric02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDcsric02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csric02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCcsric02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csric02BufferSize(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZcsric02_bufferSizeExt(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseScsric02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsric02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsric02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02Analysis(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsric02_analysis(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseScsric02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsric02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsric02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="csrValA_ValM">array of nnz (= csrRowPtrA(m)-csrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of csrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by csrsv2_bufferSizeExt().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csric02(int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA_ValM, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaSparseCsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsric02(_handle, m, (int)csrColIndA.Size, descrA.Descriptor, csrValA_ValM.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Csric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }













        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXbsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called bsrsv2_analysis() or bsrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Bsric02ZeroPivot(CudaSparseBsric02Info info, CudaDeviceVariable<int> position)
        {
            res = CudaSparseNativeMethods.cusparseXbsric02_zeroPivot(_handle, info.Bsric02Info, position.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsric02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }
        /// <summary>
        /// If the returned error code is CUSPARSE_STATUS_ZERO_PIVOT, position=j means
        /// A(j,j) has either a structural zero or a numerical zero. Otherwise position=-1. <para/>
        /// The position can be 0-based or 1-based, the same as the matrix. <para/>
        /// Function cusparseXbsrsv2_zeroPivot() is a blocking call. It calls
        /// cudaDeviceSynchronize() to make sure all previous kernels are done. <para/>
        /// The position can be in the host memory or device memory. The user can set the proper
        /// mode with cusparseSetPointerMode().
        /// </summary>
        /// <param name="info">info contains structural zero or numerical zero if the user already called bsrsv2_analysis() or bsrsv2_solve().</param>
        /// <param name="position">if no structural or numerical zero, position is -1; otherwise, if A(j,j) is missing or U(j,j) is zero, position=j.</param>
        /// <returns>If true, position=j means A(j,j) has either a structural zero or a numerical zero; otherwise, position=-1.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public bool Bsric02ZeroPivot(CudaSparseBsric02Info info, ref int position)
        {
            res = CudaSparseNativeMethods.cusparseXbsric02_zeroPivot(_handle, info.Bsric02Info, ref position);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXbsric02_zeroPivot", res));
            if (res == cusparseStatus.ZeroPivot) return true;
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return false;
        }

        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsric02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSbsric02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsric02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDbsric02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsric02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCbsric02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Bsric02BufferSize(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZbsric02_bufferSizeExt(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsric02_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsric02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsric02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsric02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the analysis phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02Analysis(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info, cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsric02_analysis(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsric02_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSbsric02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDbsric02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCbsric02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the solve phase of the incomplete-Cholesky
        /// factorization with fill-in and no pivoting: A = LL^H
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_TRIANGULAR and diagonal 
        /// types CUSPARSE_DIAG_TYPE_UNIT and CUSPARSE_DIAG_TYPE_NON_UNIT.</param>
        /// <param name="bsrValA_ValM">array of nnz (= bsrRowPtrA(m)-bsrRowPtrA(0)) non-zero elements of matrix A. <para/>Output: matrix containing the incomplete-LU lower and upper triangular factors.</param>
        /// <param name="bsrRowPtrA">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnz (= bsrRowPtrA(m) - bsrRowPtrA(0)) column indices of the non-zero elements of matrix A.
        /// Length of bsrColIndA gives the number nzz passed to CUSPARSE. </param>
        /// <param name="info">record of internal states based on different algorithms.</param>
        /// <param name="policy">The supported policies are CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL.</param>
        /// <param name="buffer">buffer allocated by the user, the size is returned by bsrsv2_bufferSizeExt().</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Bsric02(cusparseDirection dirA, int m, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA_ValM, CudaDeviceVariable<int> bsrRowPtrA,
            CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseBsric02Info info,
            cusparseSolvePolicy policy, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZbsric02(_handle, dirA, m, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA_ValM.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, info.Bsric02Info, policy, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsric02", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }










        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        public SizeT Gtsv2GetBufferSize(int m, int n, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSgtsv2_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        public SizeT Gtsv2GetBufferSize(int m, int n, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDgtsv2_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        public SizeT Gtsv2GetBufferSize(int m, int n, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseCgtsv2_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        public SizeT Gtsv2GetBufferSize(int m, int n, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseZgtsv2_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgtsv2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }






        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2(int m, int n, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSgtsv2(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsv2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2(int m, int n, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDgtsv2(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsv2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2(int m, int n, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCgtsv2(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsv2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// Solution of tridiagonal linear system A * B = B, with multiple right-hand-sides. The coefficient matrix A is 
        /// composed of lower (dl), main (d) and upper (du) diagonals, and the right-hand-sides B are overwritten with the solution.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="n">number of right-hand-sides, columns of matrix B.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal
        /// linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero.</param>
        /// <param name="B">dense right-hand-side array of dimensions (ldb, m).</param>
        /// <param name="ldb">leading dimension of B (that is >= max(1;m)).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2(int m, int n, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZgtsv2(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgtsv2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }










        /// <summary/>
        public SizeT Gtsv2_nopivotGetBufferSize(int m, int n, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSgtsv2_nopivot_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsv2_nopivot_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary/>
        public SizeT Gtsv2_nopivotGetBufferSize(int m, int n, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDgtsv2_nopivot_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsv2_nopivot_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary/>
        public SizeT Gtsv2_nopivotGetBufferSize(int m, int n, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseCgtsv2_nopivot_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsv2_nopivot_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary/>
        public SizeT Gtsv2_nopivotGetBufferSize(int m, int n, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> B, int ldb)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseZgtsv2_nopivot_bufferSizeExt(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgtsv2_nopivot_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }





        /// <summary/>
        public void Gtsv2_nopivot(int m, int n, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSgtsv2_nopivot(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsv2_nopivot", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary/>
        public void Gtsv2_nopivot(int m, int n, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDgtsv2_nopivot(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsv2_nopivot", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary/>
        public void Gtsv2_nopivot(int m, int n, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCgtsv2_nopivot(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsv2_nopivot", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary/>
        public void Gtsv2_nopivot(int m, int n, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZgtsv2_nopivot(_handle, m, n, dl.DevicePointer, d.DevicePointer, du.DevicePointer, B.DevicePointer, ldb, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgtsv2_nopivot", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }











        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        public SizeT Gtsv2StridedBatchGetBufferSize(int m, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> x, int batchCount, int batchStride)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSgtsv2StridedBatch_bufferSizeExt(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsv2StridedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        public SizeT Gtsv2StridedBatchGetBufferSize(int m, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> x, int batchCount, int batchStride)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDgtsv2StridedBatch_bufferSizeExt(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsv2StridedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        public SizeT Gtsv2StridedBatchGetBufferSize(int m, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> x, int batchCount, int batchStride)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseCgtsv2StridedBatch_bufferSizeExt(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsv2StridedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        public SizeT Gtsv2StridedBatchGetBufferSize(int m, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> x, int batchCount, int batchStride)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseZgtsv2StridedBatch_bufferSizeExt(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgtsv2StridedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }






        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2StridedBatch(int m, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> x, int batchCount, int batchStride, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSgtsv2StridedBatch(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsv2StridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2StridedBatch(int m, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> x, int batchCount, int batchStride, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDgtsv2StridedBatch(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsv2StridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2StridedBatch(int m, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> x, int batchCount, int batchStride, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCgtsv2StridedBatch(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsv2StridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// Solution of a set of tridiagonal linear systems A * x = x, each with a single right-hand-side. The coefficient 
        /// matrices A are composed of lower (dl), main (d) and upper (du) diagonals and stored separated by a batchStride, while the 
        /// right-hand-sides x are also separated by a batchStride.
        /// </summary>
        /// <param name="m">the size of the linear system (must be >= 3).</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal 
        /// linear system. The lower diagonal dl(i) that corresponds to the ith linear system starts at location dl + batchStride * i in memory.
        /// Also, the first element of each lower diagonal must be zero.</param>
        /// <param name="d">dense array containing the main diagonal of the tri-diagonal linear system. The main diagonal d(i) that corresponds to the ith
        /// linear system starts at location d + batchStride * i in memory.</param>
        /// <param name="du">dense array containing the upper diagonal of the tri-diagonal linear system. The upper diagonal du(i) that corresponds to the ith
        /// linear system starts at location du + batchStride * i in memory. Also, the last element of each upper diagonal must be zero.</param>
        /// <param name="x">dense array that contains the right-hand-side of the tridiagonal linear system. The right-hand-side x(i) that corresponds 
        /// to the ith linear system starts at location x + batchStride * i in memory.</param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="batchStride">stride (number of elements) that separates the vectors of every system (must be at least m).</param>
        /// <param name="buffer">Buffer</param>
        public void Gtsv2StridedBatch(int m, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> x, int batchCount, int batchStride, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZgtsv2StridedBatch(_handle, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, batchStride, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparse2ZgtsvStridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GtsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSgtsvInterleavedBatch_bufferSizeExt(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GtsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDgtsvInterleavedBatch_bufferSizeExt(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GtsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseCgtsvInterleavedBatch_bufferSizeExt(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GtsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseZgtsvInterleavedBatch_bufferSizeExt(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgtsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }



        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gtsvInterleavedBatch_bufferSizeExt. </param>
        public void GtsvInterleavedBatch(int algo, int m, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSgtsvInterleavedBatch(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgtsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gtsvInterleavedBatch_bufferSizeExt. </param>
        public void GtsvInterleavedBatch(int algo, int m, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDgtsvInterleavedBatch(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgtsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gtsvInterleavedBatch_bufferSizeExt. </param>
        public void GtsvInterleavedBatch(int algo, int m, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCgtsvInterleavedBatch(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgtsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function computes the solution of multiple tridiagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">algo = 0: cuThomas (unstable algorithm); algo = 1: LU with pivoting (stable algorithm); algo = 2: QR (stable algorithm)</param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="dl">dense array containing the lower diagonal of the tri-diagonal linear system. The first element of each lower diagonal must be zero.</param>
        /// <param name="d"> dense array containing the main diagonal of the tri-diagonal linear system. </param>
        /// <param name="du"> dense array containing the upper diagonal of the tri-diagonal linear system. The last element of each upper diagonal must be zero. </param>
        /// <param name="x"> dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gtsvInterleavedBatch_bufferSizeExt. </param>
        public void GtsvInterleavedBatch(int algo, int m, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZgtsvInterleavedBatch(_handle, algo, m, dl.DevicePointer, d.DevicePointer, du.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgtsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }





        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GpsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<float> ds, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> dw, CudaDeviceVariable<float> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSgpsvInterleavedBatch_bufferSizeExt(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgpsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GpsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<double> ds, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> dw, CudaDeviceVariable<double> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDgpsvInterleavedBatch_bufferSizeExt(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgpsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GpsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<cuFloatComplex> ds, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> dw, CudaDeviceVariable<cuFloatComplex> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseCgpsvInterleavedBatch_bufferSizeExt(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgpsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        public SizeT GpsvInterleavedBatch_bufferSizeExt(int algo, int m, CudaDeviceVariable<cuDoubleComplex> ds, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> dw, CudaDeviceVariable<cuDoubleComplex> x, int batchCount)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseZgpsvInterleavedBatch_bufferSizeExt(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgpsvInterleavedBatch_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gpsvInterleavedBatch_bufferSizeExt.</param>
        public void GpsvInterleavedBatch(int algo, int m, CudaDeviceVariable<float> ds, CudaDeviceVariable<float> dl, CudaDeviceVariable<float> d, CudaDeviceVariable<float> du, CudaDeviceVariable<float> dw, CudaDeviceVariable<float> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSgpsvInterleavedBatch(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgpsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gpsvInterleavedBatch_bufferSizeExt.</param>
        public void GpsvInterleavedBatch(int algo, int m, CudaDeviceVariable<double> ds, CudaDeviceVariable<double> dl, CudaDeviceVariable<double> d, CudaDeviceVariable<double> du, CudaDeviceVariable<double> dw, CudaDeviceVariable<double> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDgpsvInterleavedBatch(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgpsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gpsvInterleavedBatch_bufferSizeExt.</param>
        public void GpsvInterleavedBatch(int algo, int m, CudaDeviceVariable<cuFloatComplex> ds, CudaDeviceVariable<cuFloatComplex> dl, CudaDeviceVariable<cuFloatComplex> d, CudaDeviceVariable<cuFloatComplex> du, CudaDeviceVariable<cuFloatComplex> dw, CudaDeviceVariable<cuFloatComplex> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCgpsvInterleavedBatch(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgpsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function computes the solution of multiple penta-diagonal linear systems for i=0, ...,batchCount:
        /// </summary>
        /// <param name="algo">only support algo = 0 (QR) </param>
        /// <param name="m">the size of the linear system.</param>
        /// <param name="ds"> dense array containing the lower diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The first two elements must be zero.</param>
        /// <param name="dl"> dense array containing the lower diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The first element must be zero. </param>
        /// <param name="d">  dense array containing the main diagonal of the penta-diagonal linear system.  </param>
        /// <param name="du">  dense array containing the upper diagonal (distance 1 to the diagonal) of the penta-diagonal linear system. The last element must be zero. </param>
        /// <param name="dw">dense array containing the upper diagonal (distance 2 to the diagonal) of the penta-diagonal linear system. The last two elements must be zero. </param>
        /// <param name="x">dense right-hand-side array of dimensions (batchCount, n). </param>
        /// <param name="batchCount">Number of systems to solve.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gpsvInterleavedBatch_bufferSizeExt.</param>
        public void GpsvInterleavedBatch(int algo, int m, CudaDeviceVariable<cuDoubleComplex> ds, CudaDeviceVariable<cuDoubleComplex> dl, CudaDeviceVariable<cuDoubleComplex> d, CudaDeviceVariable<cuDoubleComplex> du, CudaDeviceVariable<cuDoubleComplex> dw, CudaDeviceVariable<cuDoubleComplex> x, int batchCount, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZgpsvInterleavedBatch(_handle, algo, m, ds.DevicePointer, dl.DevicePointer, d.DevicePointer, du.DevicePointer, dw.DevicePointer, x.DevicePointer, batchCount, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgpsvInterleavedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /* --- Sparse Format Conversion --- */

        /// <summary>
        /// This routine finds the total number of non-zero elements and 
        /// the number of non-zero elements per row in a noncompressed csr matrix A.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Nnz_compress(int m, CudaSparseMatrixDescriptor descr, CudaDeviceVariable<float> values, CudaDeviceVariable<int> rowPtr, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<int> nnzTotal, float tol)
        {
            res = CudaSparseNativeMethods.cusparseSnnz_compress(_handle, m, descr.Descriptor, values.DevicePointer, rowPtr.DevicePointer, nnzPerRow.DevicePointer, nnzTotal.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSnnz_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This routine finds the total number of non-zero elements and 
        /// the number of non-zero elements per row in a noncompressed csr matrix A.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Nnz_compress(int m, CudaSparseMatrixDescriptor descr, CudaDeviceVariable<double> values, CudaDeviceVariable<int> rowPtr, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<int> nnzTotal, double tol)
        {
            res = CudaSparseNativeMethods.cusparseDnnz_compress(_handle, m, descr.Descriptor, values.DevicePointer, rowPtr.DevicePointer, nnzPerRow.DevicePointer, nnzTotal.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnnz_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This routine finds the total number of non-zero elements and 
        /// the number of non-zero elements per row in a noncompressed csr matrix A.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Nnz_compress(int m, CudaSparseMatrixDescriptor descr, CudaDeviceVariable<cuFloatComplex> values, CudaDeviceVariable<int> rowPtr, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<int> nnzTotal, cuFloatComplex tol)
        {
            res = CudaSparseNativeMethods.cusparseCnnz_compress(_handle, m, descr.Descriptor, values.DevicePointer, rowPtr.DevicePointer, nnzPerRow.DevicePointer, nnzTotal.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCnnz_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This routine finds the total number of non-zero elements and 
        /// the number of non-zero elements per row in a noncompressed csr matrix A.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Nnz_compress(int m, CudaSparseMatrixDescriptor descr, CudaDeviceVariable<cuDoubleComplex> values, CudaDeviceVariable<int> rowPtr, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<int> nnzTotal, cuDoubleComplex tol)
        {
            res = CudaSparseNativeMethods.cusparseZnnz_compress(_handle, m, descr.Descriptor, values.DevicePointer, rowPtr.DevicePointer, nnzPerRow.DevicePointer, nnzTotal.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZnnz_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This routine takes as input a csr form where the values may have 0 elements
        /// and compresses it to return a csr form with no zeros.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csr_compress(int m, int n, CudaSparseMatrixDescriptor descra, CudaDeviceVariable<float> inVal, CudaDeviceVariable<int> inColInd, CudaDeviceVariable<int> inRowPtr,
                int inNnz, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<float> outVal, CudaDeviceVariable<int> outColInd, CudaDeviceVariable<int> outRowPtr, float tol)
        {
            res = CudaSparseNativeMethods.cusparseScsr2csr_compress(_handle, m, n, descra.Descriptor, inVal.DevicePointer, inColInd.DevicePointer, inRowPtr.DevicePointer, inNnz,
                nnzPerRow.DevicePointer, outVal.DevicePointer, outColInd.DevicePointer, outRowPtr.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsr2csr_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This routine takes as input a csr form where the values may have 0 elements
        /// and compresses it to return a csr form with no zeros.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csr_compress(int m, int n, CudaSparseMatrixDescriptor descra, CudaDeviceVariable<double> inVal, CudaDeviceVariable<int> inColInd, CudaDeviceVariable<int> inRowPtr,
                int inNnz, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<double> outVal, CudaDeviceVariable<int> outColInd, CudaDeviceVariable<int> outRowPtr, double tol)
        {
            res = CudaSparseNativeMethods.cusparseDcsr2csr_compress(_handle, m, n, descra.Descriptor, inVal.DevicePointer, inColInd.DevicePointer, inRowPtr.DevicePointer, inNnz,
                nnzPerRow.DevicePointer, outVal.DevicePointer, outColInd.DevicePointer, outRowPtr.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsr2csr_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This routine takes as input a csr form where the values may have 0 elements
        /// and compresses it to return a csr form with no zeros.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csr_compress(int m, int n, CudaSparseMatrixDescriptor descra, CudaDeviceVariable<cuFloatComplex> inVal, CudaDeviceVariable<int> inColInd, CudaDeviceVariable<int> inRowPtr,
                int inNnz, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<cuFloatComplex> outVal, CudaDeviceVariable<int> outColInd, CudaDeviceVariable<int> outRowPtr, cuFloatComplex tol)
        {
            res = CudaSparseNativeMethods.cusparseCcsr2csr_compress(_handle, m, n, descra.Descriptor, inVal.DevicePointer, inColInd.DevicePointer, inRowPtr.DevicePointer, inNnz,
                nnzPerRow.DevicePointer, outVal.DevicePointer, outColInd.DevicePointer, outRowPtr.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsr2csr_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This routine takes as input a csr form where the values may have 0 elements
        /// and compresses it to return a csr form with no zeros.
        /// </summary>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csr_compress(int m, int n, CudaSparseMatrixDescriptor descra, CudaDeviceVariable<cuDoubleComplex> inVal, CudaDeviceVariable<int> inColInd, CudaDeviceVariable<int> inRowPtr,
                int inNnz, CudaDeviceVariable<int> nnzPerRow, CudaDeviceVariable<cuDoubleComplex> outVal, CudaDeviceVariable<int> outColInd, CudaDeviceVariable<int> outRowPtr, cuDoubleComplex tol)
        {
            res = CudaSparseNativeMethods.cusparseZcsr2csr_compress(_handle, m, n, descra.Descriptor, inVal.DevicePointer, inColInd.DevicePointer, inRowPtr.DevicePointer, inNnz,
                nnzPerRow.DevicePointer, outVal.DevicePointer, outColInd.DevicePointer, outRowPtr.DevicePointer, tol);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsr2csr_compress", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #region ref host
        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, ref int nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseSnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, ref nnzTotalDevHostPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, ref int nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseDnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, ref nnzTotalDevHostPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, ref int nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseCnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, ref nnzTotalDevHostPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, ref int nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseDnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, ref nnzTotalDevHostPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        #endregion
        #region ref device
        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseSnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, nnzTotalDevHostPtr.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseDnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, nnzTotalDevHostPtr.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseCnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, nnzTotalDevHostPtr.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This routine finds the total number of non-zero elements and the number of non-zero elements per row or column in the dense matrix A.
        /// </summary>
        /// <param name="dirA">direction that specifies whether to count non-zero elements by CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="A">array of dimensions (lda, n).</param>
        /// <param name="lda">leading dimension of dense array A.</param>
        /// <param name="nnzPerRowCol">Output: array of size m or n containing the number of non-zero elements per row or column, respectively.</param>
        /// <param name="nnzTotalDevHostPtr">Output: total number of non-zero elements in device or host memory.</param>
        public void Nnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<int> nnzPerRowCol, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseZnnz(_handle, dirA, m, n, descrA.Descriptor, A.DevicePointer, lda, nnzPerRowCol.DevicePointer, nnzTotalDevHostPtr.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZnnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        #endregion



        /// <summary>
        /// This routine compresses the indecis of rows or columns. It can be interpreted as a conversion from COO to CSR sparse storage format.
        /// </summary>
        /// <param name="cooRowInd">integer array of nnz uncompressed row indices. Length of cooRowInd gives the number nzz passed to CUSPARSE.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="csrRowPtr">Output: integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="idxBase">Index base.</param>
        public void Xcoo2csr(CudaDeviceVariable<int> cooRowInd, int m, CudaDeviceVariable<int> csrRowPtr, IndexBase idxBase)
        {
            res = CudaSparseNativeMethods.cusparseXcoo2csr(_handle, cooRowInd.DevicePointer, (int)cooRowInd.Size, m, csrRowPtr.DevicePointer, idxBase);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcoo2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This routine uncompresses the indecis of rows or columns. It can be interpreted as a conversion from CSR to COO sparse storage format.
        /// </summary>
        /// <param name="csrRowPtr">Output: integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="cooRowInd">integer array of nnz uncompressed row indices. Length of cooRowInd gives the number nzz passed to CUSPARSE.</param>
        /// <param name="idxBase">Index base.</param>
        public void Xcsr2coo(CudaDeviceVariable<int> csrRowPtr, int m, CudaDeviceVariable<int> cooRowInd, IndexBase idxBase)
        {
            res = CudaSparseNativeMethods.cusparseXcoo2csr(_handle, csrRowPtr.DevicePointer, (int)cooRowInd.Size, m, cooRowInd.DevicePointer, idxBase);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcoo2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion

        #region Sparse Level 4 routines

        /// <summary>
        /// determine csrRowPtrC and the total number of nonzero elements
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="nnzTotalDevHostPtr"> total number of nonzero elements in device or host memory. It is equal to (csrRowPtrC(m) - csrRowPtrC(0)). </param>
        /// <param name="buffer"></param>
        public void Csrgeam2Nnz(int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> nnzTotalDevHostPtr, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseXcsrgeam2Nnz(_handle, m, n, descrA.Descriptor, (int)csrColIndA.Size, csrRowPtrA.DevicePointer,
                csrColIndA.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor,
                csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsrgeam2Nnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// determine csrRowPtrC and the total number of nonzero elements
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="nnzTotalDevHostPtr"> total number of nonzero elements in device or host memory. It is equal to (csrRowPtrC(m) - csrRowPtrC(0)). </param>
        /// <param name="buffer"></param>
        public void Csrgeam2Nnz(int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, ref int nnzTotalDevHostPtr, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseXcsrgeam2Nnz(_handle, m, n, descrA.Descriptor, (int)csrColIndA.Size, csrRowPtrA.DevicePointer,
                csrColIndA.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor,
                csrRowPtrC.DevicePointer, ref nnzTotalDevHostPtr, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsrgeam2Nnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, CudaDeviceVariable<float> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<float> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<float> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseScsrgeam2_bufferSizeExt(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, CudaDeviceVariable<double> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<double> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<double> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDcsrgeam2_bufferSizeExt(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<cuFloatComplex> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuFloatComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<cuFloatComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseCcsrgeam2_bufferSizeExt(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<cuDoubleComplex> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuDoubleComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<cuDoubleComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseZcsrgeam2_bufferSizeExt(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, ref float alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            ref float beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<float> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseScsrgeam2_bufferSizeExt(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, ref double alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            ref double beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<double> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDcsrgeam2_bufferSizeExt(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, ref cuFloatComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            ref cuFloatComplex beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuFloatComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<cuFloatComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseCcsrgeam2_bufferSizeExt(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        public SizeT Csrgeam2_bufferSizeExt(int m, int n, ref cuDoubleComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            ref cuDoubleComplex beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuDoubleComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB, CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<cuDoubleComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseZcsrgeam2_bufferSizeExt(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrgeam2_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, CudaDeviceVariable<float> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<float> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<float> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseScsrgeam2(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, CudaDeviceVariable<double> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<double> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<double> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsrgeam2(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuFloatComplex> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuFloatComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuFloatComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsrgeam2(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuDoubleComplex> beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuDoubleComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuDoubleComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsrgeam2(_handle, m, n, alpha.DevicePointer, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, beta.DevicePointer, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, ref float alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, ref float beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<float> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseScsrgeam2(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, ref double alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, ref double beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<double> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsrgeam2(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, ref cuFloatComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, ref cuFloatComplex beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuFloatComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuFloatComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsrgeam2(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs following matrix-matrix operation<para/>
        /// C = alpha * A + beta * B <para/>
        /// where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and alpha and beta are scalars. 
        /// </summary>
        /// <param name="m">number of rows of sparse matrix A,B,C.</param>
        /// <param name="n">number of columns of sparse matrix A,B,C.</param>
        /// <param name="alpha">scalar used for multiplication.</param> 
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_
        /// MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValA">array of nnzA non-zero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnzA column indices of the non-zero elements of matrix A. Length of csrColIndA gives the number nzzA passed to CUSPARSE.</param>
        /// <param name="beta">scalar used for multiplication.</param>
        /// <param name="descrB">the descriptor of matrix B. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValB">array of nnzB non-zero elements of matrix B.</param>
        /// <param name="csrRowPtrB">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndB">integer array of nnzB column indices of the non-zero elements of matrix B. Length of csrColIndB gives the number nzzB passed to CUSPARSE.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL only.</param>
        /// <param name="csrValC">array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnzC (= csrRowPtrC(m) - csrRowPtrC(0)) column indices of the non-zero elements of matrix C.</param>
        /// <param name="buffer">Buffer</param>
        public void Csrgeam2(int m, int n, ref cuDoubleComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA,
            CudaDeviceVariable<int> csrColIndA, ref cuDoubleComplex beta, CudaSparseMatrixDescriptor descrB, CudaDeviceVariable<cuDoubleComplex> csrValB, CudaDeviceVariable<int> csrRowPtrB,
            CudaDeviceVariable<int> csrColIndB, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuDoubleComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsrgeam2(_handle, m, n, ref alpha, descrA.Descriptor, (int)csrColIndA.Size, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref beta, descrB.Descriptor, (int)csrColIndB.Size, csrValB.DevicePointer, csrRowPtrB.DevicePointer, csrColIndB.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrgeam2", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }




        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            float fractionToColor, ref int ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseScsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                ref fractionToColor, ref ncolors, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            double fractionToColor, ref int ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseDcsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                ref fractionToColor, ref ncolors, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            float fractionToColor, ref int ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseCcsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                ref fractionToColor, ref ncolors, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            double fractionToColor, ref int ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseZcsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                ref fractionToColor, ref ncolors, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            CudaDeviceVariable<float> fractionToColor, CudaDeviceVariable<int> ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseScsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                fractionToColor.DevicePointer, ncolors.DevicePointer, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            CudaDeviceVariable<double> fractionToColor, CudaDeviceVariable<int> ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseDcsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                fractionToColor.DevicePointer, ncolors.DevicePointer, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            CudaDeviceVariable<float> fractionToColor, CudaDeviceVariable<int> ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseCcsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                fractionToColor.DevicePointer, ncolors.DevicePointer, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the coloring of the adjacency graph associated with the matrix
        /// A stored in CSR format. The coloring is an assignment of colors (integer numbers)
        /// to nodes, such that neighboring nodes have distinct colors. An approximate coloring
        /// algorithm is used in this routine, and is stopped when a certain percentage of nodes has
        /// been colored. The rest of the nodes are assigned distinct colors (an increasing sequence
        /// of integers numbers, starting from the last integer used previously). The last two
        /// auxiliary routines can be used to extract the resulting number of colors, their assignment
        /// and the associated reordering. The reordering is such that nodes that have been assigned
        /// the same color are reordered to be next to each other.<para/>
        /// The matrix A passed to this routine, must be stored as a general matrix and have a
        /// symmetric sparsity pattern. If the matrix is nonsymmetric the user should pass A+A^T
        /// as a parameter to this routine.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are 
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrSortedValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) nonzero elements of matrix A.</param>
        /// <param name="csrSortedRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrSortedColIndA">integer array of nnz csrRowPtrA(m) csrRowPtrA(0) column indices of the nonzero elements of matrix A.</param>
        /// <param name="fractionToColor">fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.</param>
        /// <param name="ncolors">The number of distinct colors used (at most the size of the matrix, but likely much smaller).</param>
        /// <param name="coloring">The resulting coloring permutation.</param>
        /// <param name="reordering">The resulting reordering permutation (untouched if NULL)</param>
        /// <param name="info">structure with information to be passed to the coloring.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csrcolor(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrSortedValA, CudaDeviceVariable<int> csrSortedRowPtrA, CudaDeviceVariable<int> csrSortedColIndA,
            CudaDeviceVariable<double> fractionToColor, CudaDeviceVariable<int> ncolors, CudaDeviceVariable<int> coloring, CudaDeviceVariable<int> reordering, CudaSparseColorInfo info)
        {
            res = CudaSparseNativeMethods.cusparseZcsrcolor(_handle, m, nnz, descrA.Descriptor, csrSortedValA.DevicePointer, csrSortedRowPtrA.DevicePointer, csrSortedColIndA.DevicePointer,
                fractionToColor.DevicePointer, ncolors.DevicePointer, coloring.DevicePointer, reordering.DevicePointer, info.ColorInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsrcolor", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }















        /// <summary>
        /// This function converts a sparse matrix in CSR format (that is defined by the three arrays
        /// csrValA, csrRowPtrA and csrColIndA) into a sparse matrix in BSR format (that is
        /// defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// A is m x n sparse matrix and C is (mb*blockDim) x (nb*blockDim) sparse matrix.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column
        /// indices of the non-zero elements of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A. The range of blockDim is between
        /// 1 and min(m, n).</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Csr2bsrNnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseXcsr2bsrNnz(_handle, dirA, m, n, descrA.Descriptor, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, blockDim, descrC.Descriptor, bsrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsr2bsrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function converts a sparse matrix in CSR format (that is defined by the three arrays
        /// csrValA, csrRowPtrA and csrColIndA) into a sparse matrix in BSR format (that is
        /// defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// A is m x n sparse matrix and C is (mb*blockDim) x (nb*blockDim) sparse matrix.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column
        /// indices of the non-zero elements of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A. The range of blockDim is between
        /// 1 and min(m, n).</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Csr2bsrNnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> bsrRowPtrC, ref int nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseXcsr2bsrNnz(_handle, dirA, m, n, descrA.Descriptor, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, blockDim, descrC.Descriptor, bsrRowPtrC.DevicePointer, ref nnzTotalDevHostPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsr2bsrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function converts a sparse matrix in CSR format (that is defined by the three arrays
        /// csrValA, csrRowPtrA and csrColIndA) into a sparse matrix in BSR format (that is
        /// defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// A is m x n sparse matrix and C is (mb*blockDim) x (nb*blockDim) sparse matrix.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) non-zero 
        /// elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column
        /// indices of the non-zero elements of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A. The range of blockDim is between
        /// 1 and min(m, n).</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="bsrValC">array of nnzb*blockDim² non-zero elements of matrix C.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndC">integer array of nnzb column indices of the non-zero blocks of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Csr2bsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseScsr2bsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, blockDim, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsr2bsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function converts a sparse matrix in CSR format (that is defined by the three arrays
        /// csrValA, csrRowPtrA and csrColIndA) into a sparse matrix in BSR format (that is
        /// defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// A is m x n sparse matrix and C is (mb*blockDim) x (nb*blockDim) sparse matrix.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) non-zero 
        /// elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column
        /// indices of the non-zero elements of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A. The range of blockDim is between
        /// 1 and min(m, n).</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="bsrValC">array of nnzb*blockDim² non-zero elements of matrix C.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndC">integer array of nnzb column indices of the non-zero blocks of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Csr2bsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseDcsr2bsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, blockDim, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsr2bsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function converts a sparse matrix in CSR format (that is defined by the three arrays
        /// csrValA, csrRowPtrA and csrColIndA) into a sparse matrix in BSR format (that is
        /// defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// A is m x n sparse matrix and C is (mb*blockDim) x (nb*blockDim) sparse matrix.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) non-zero 
        /// elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column
        /// indices of the non-zero elements of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A. The range of blockDim is between
        /// 1 and min(m, n).</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="bsrValC">array of nnzb*blockDim² non-zero elements of matrix C.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndC">integer array of nnzb column indices of the non-zero blocks of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Csr2bsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuFloatComplex> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseCcsr2bsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, blockDim, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsr2bsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function converts a sparse matrix in CSR format (that is defined by the three arrays
        /// csrValA, csrRowPtrA and csrColIndA) into a sparse matrix in BSR format (that is
        /// defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// A is m x n sparse matrix and C is (mb*blockDim) x (nb*blockDim) sparse matrix.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="csrValA">array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) non-zero 
        /// elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz (= csrRowPtrA(m) - csrRowPtrA(0)) column
        /// indices of the non-zero elements of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A. The range of blockDim is between
        /// 1 and min(m, n).</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="bsrValC">array of nnzb*blockDim² non-zero elements of matrix C.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndC">integer array of nnzb column indices of the non-zero blocks of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Csr2bsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuDoubleComplex> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseZcsr2bsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, blockDim, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsr2bsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function converts a sparse matrix in BSR format (that is defined by the three arrays
        /// bsrValA, bsrRowPtrA and bsrColIndA) into a sparse matrix in CSR format (that is
        /// defined by arrays csrValC, csrRowPtrC, and csrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A. The number of rows of sparse matrix C is m(= mb*blockDim).</param>
        /// <param name="nb">number of block columns of sparse matrix A. The number of columns of sparse matrix C is n(= nb*blockDim).</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="bsrValA">array of nnzb*blockDim² non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the non-zero blocks of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="csrValC">array of nnz (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Bsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseSbsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function converts a sparse matrix in BSR format (that is defined by the three arrays
        /// bsrValA, bsrRowPtrA and bsrColIndA) into a sparse matrix in CSR format (that is
        /// defined by arrays csrValC, csrRowPtrC, and csrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A. The number of rows of sparse matrix C is m(= mb*blockDim).</param>
        /// <param name="nb">number of block columns of sparse matrix A. The number of columns of sparse matrix C is n(= nb*blockDim).</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="bsrValA">array of nnzb*blockDim² non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the non-zero blocks of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="csrValC">array of nnz (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Bsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseDbsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function converts a sparse matrix in BSR format (that is defined by the three arrays
        /// bsrValA, bsrRowPtrA and bsrColIndA) into a sparse matrix in CSR format (that is
        /// defined by arrays csrValC, csrRowPtrC, and csrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A. The number of rows of sparse matrix C is m(= mb*blockDim).</param>
        /// <param name="nb">number of block columns of sparse matrix A. The number of columns of sparse matrix C is n(= nb*blockDim).</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="bsrValA">array of nnzb*blockDim² non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the non-zero blocks of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="csrValC">array of nnz (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Bsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuFloatComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseCbsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function converts a sparse matrix in BSR format (that is defined by the three arrays
        /// bsrValA, bsrRowPtrA and bsrColIndA) into a sparse matrix in CSR format (that is
        /// defined by arrays csrValC, csrRowPtrC, and csrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A. The number of rows of sparse matrix C is m(= mb*blockDim).</param>
        /// <param name="nb">number of block columns of sparse matrix A. The number of columns of sparse matrix C is n(= nb*blockDim).</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="bsrValA">array of nnzb*blockDim² non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the non-zero blocks of matrix A.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="descrC">the descriptor of matrix C.</param>
        /// <param name="csrValC">array of nnz (= csrRowPtrC(m) - csrRowPtrC(0)) non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m + 1 elements that contains the start of every row
        /// and the end of the last row plus one.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Bsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuDoubleComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseZbsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        #region Removed in Cuda 5.5 production release, present in pre-release, again in Cuda 6
        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last
        /// block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <returns>number of bytes of the buffer used in the gebsr2gebsc.</returns>
        public SizeT Gebsr2gebscBufferSize(int mb, int nb, CudaDeviceVariable<float> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSgebsr2gebsc_bufferSizeExt(_handle, mb, nb, (int)bsrColInd.Size, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgebsr2gebsc_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last
        /// block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <returns>number of bytes of the buffer used in the gebsr2gebsc.</returns>
        public SizeT Gebsr2gebscBufferSize(int mb, int nb, CudaDeviceVariable<double> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDgebsr2gebsc_bufferSizeExt(_handle, mb, nb, (int)bsrColInd.Size, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgebsr2gebsc_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last
        /// block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <returns>number of bytes of the buffer used in the gebsr2gebsc.</returns>
        public SizeT Gebsr2gebscBufferSize(int mb, int nb, CudaDeviceVariable<cuFloatComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCgebsr2gebsc_bufferSizeExt(_handle, mb, nb, (int)bsrColInd.Size, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgebsr2gebsc_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last
        /// block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <returns>number of bytes of the buffer used in the gebsr2gebsc.</returns>
        public SizeT Gebsr2gebscBufferSize(int mb, int nb, CudaDeviceVariable<cuDoubleComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZgebsr2gebsc_bufferSizeExt(_handle, mb, nb, (int)bsrColInd.Size, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgebsr2gebsc_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function returns the size of the buffer used in computing csr2gebsrNnz and csr2gebsr.
        /// </summary>
        /// <param name="dir">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A.</param>
        /// <param name="csrColInd">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        public SizeT Csr2gebsrBufferSize(cusparseDirection dir, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrVal,
            CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseScsr2gebsr_bufferSizeExt(_handle, dir, m, n, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function returns the size of the buffer used in computing csr2gebsrNnz and csr2gebsr.
        /// </summary>
        /// <param name="dir">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A.</param>
        /// <param name="csrColInd">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        public SizeT Csr2gebsrBufferSize(cusparseDirection dir, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrVal,
            CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDcsr2gebsr_bufferSizeExt(_handle, dir, m, n, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function returns the size of the buffer used in computing csr2gebsrNnz and csr2gebsr.
        /// </summary>
        /// <param name="dir">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A.</param>
        /// <param name="csrColInd">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        public SizeT Csr2gebsrBufferSize(cusparseDirection dir, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrVal,
            CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCcsr2gebsr_bufferSizeExt(_handle, dir, m, n, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function returns the size of the buffer used in computing csr2gebsrNnz and csr2gebsr.
        /// </summary>
        /// <param name="dir">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A.</param>
        /// <param name="csrColInd">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        public SizeT Csr2gebsrBufferSize(cusparseDirection dir, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrVal,
            CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int rowBlockDim, int colBlockDim)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZcsr2gebsr_bufferSizeExt(_handle, dir, m, n, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, rowBlockDim, colBlockDim, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }



        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsrNnz and gebsr2gebsr.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C.</param>
        /// <param name="colBlockDimC">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        [Obsolete("Deprecated in Cuda 12.8")]
        public SizeT Gebsr2gebsrBufferSize(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA,
            CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSgebsr2gebsr_bufferSizeExt(_handle, dirA, mb, nb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgebsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsrNnz and gebsr2gebsr.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C.</param>
        /// <param name="colBlockDimC">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        [Obsolete("Deprecated in Cuda 12.8")]
        public SizeT Gebsr2gebsrBufferSize(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA,
            CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDgebsr2gebsr_bufferSizeExt(_handle, dirA, mb, nb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgebsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsrNnz and gebsr2gebsr.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C.</param>
        /// <param name="colBlockDimC">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        [Obsolete("Deprecated in Cuda 12.8")]
        public SizeT Gebsr2gebsrBufferSize(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA,
            CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCgebsr2gebsr_bufferSizeExt(_handle, dirA, mb, nb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgebsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }
        /// <summary>
        /// This function returns size of buffer used in computing gebsr2gebsrNnz and gebsr2gebsr.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL.
        /// Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C.</param>
        /// <param name="colBlockDimC">number of columns within a block of C.</param>
        /// <returns>number of bytes of the buffer used in csr2gebsrNnz() and csr2gebsr().</returns>
        [Obsolete("Deprecated in Cuda 12.8")]
        public SizeT Gebsr2gebsrBufferSize(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA,
            CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZgebsr2gebsr_bufferSizeExt(_handle, dirA, mb, nb, (int)bsrColIndA.Size, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgebsr2gebsr_bufferSizeExt(", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }


        /// <summary>
        /// This function can be seen as the same as csr2csc when regarding each block of size
        /// rowBlockDim*colBlockDim as a scalar.<para/>
        /// This sparsity pattern of result matrix can also be seen as the transpose of the original
        /// sparse matrix but memory layout of a block does not change.<para/>
        /// The user must know the size of buffer required by gebsr2gebsc by calling
        /// gebsr2gebsc_bufferSizeExt, allocate the buffer and pass the buffer pointer to gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="bscVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A. It is only filled-in if copyValues is set to CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="bscRowInd">integer array of nnzb row indices of the non-zero blocks of matrix A</param>
        /// <param name="bscColPtr">integer array of nb+1 elements that contains the start of every block column and the end of the last block column plus one.</param>
        /// <param name="copyValues">CUSPARSE_ACTION_SYMBOLIC or CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="baseIdx">CUSPARSE_INDEX_BASE_ZERO or CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsc_bufferSizeExt.</param>
        public void Gebsr2gebsc(int mb, int nb, int nnzb, CudaDeviceVariable<float> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                int rowBlockDim, int colBlockDim, CudaDeviceVariable<float> bscVal, CudaDeviceVariable<int> bscRowInd, CudaDeviceVariable<int> bscColPtr,
                                cusparseAction copyValues, IndexBase baseIdx, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSgebsr2gebsc(_handle, mb, nb, nnzb, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, bscVal.DevicePointer,
                bscRowInd.DevicePointer, bscColPtr.DevicePointer, copyValues, baseIdx, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgebsr2gebsc", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function can be seen as the same as csr2csc when regarding each block of size
        /// rowBlockDim*colBlockDim as a scalar.<para/>
        /// This sparsity pattern of result matrix can also be seen as the transpose of the original
        /// sparse matrix but memory layout of a block does not change.<para/>
        /// The user must know the size of buffer required by gebsr2gebsc by calling
        /// gebsr2gebsc_bufferSizeExt, allocate the buffer and pass the buffer pointer to gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="bscVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A. It is only filled-in if copyValues is set to CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="bscRowInd">integer array of nnzb row indices of the non-zero blocks of matrix A</param>
        /// <param name="bscColPtr">integer array of nb+1 elements that contains the start of every block column and the end of the last block column plus one.</param>
        /// <param name="copyValues">CUSPARSE_ACTION_SYMBOLIC or CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="baseIdx">CUSPARSE_INDEX_BASE_ZERO or CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsc_bufferSizeExt.</param>
        public void Gebsr2gebsc(int mb, int nb, int nnzb, CudaDeviceVariable<double> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                int rowBlockDim, int colBlockDim, CudaDeviceVariable<double> bscVal, CudaDeviceVariable<int> bscRowInd, CudaDeviceVariable<int> bscColPtr,
                                cusparseAction copyValues, IndexBase baseIdx, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDgebsr2gebsc(_handle, mb, nb, nnzb, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, bscVal.DevicePointer,
                bscRowInd.DevicePointer, bscColPtr.DevicePointer, copyValues, baseIdx, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgebsr2gebsc", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function can be seen as the same as csr2csc when regarding each block of size
        /// rowBlockDim*colBlockDim as a scalar.<para/>
        /// This sparsity pattern of result matrix can also be seen as the transpose of the original
        /// sparse matrix but memory layout of a block does not change.<para/>
        /// The user must know the size of buffer required by gebsr2gebsc by calling
        /// gebsr2gebsc_bufferSizeExt, allocate the buffer and pass the buffer pointer to gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="bscVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A. It is only filled-in if copyValues is set to CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="bscRowInd">integer array of nnzb row indices of the non-zero blocks of matrix A</param>
        /// <param name="bscColPtr">integer array of nb+1 elements that contains the start of every block column and the end of the last block column plus one.</param>
        /// <param name="copyValues">CUSPARSE_ACTION_SYMBOLIC or CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="baseIdx">CUSPARSE_INDEX_BASE_ZERO or CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsc_bufferSizeExt.</param>
        public void Gebsr2gebsc(int mb, int nb, int nnzb, CudaDeviceVariable<cuFloatComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                int rowBlockDim, int colBlockDim, CudaDeviceVariable<cuFloatComplex> bscVal, CudaDeviceVariable<int> bscRowInd, CudaDeviceVariable<int> bscColPtr,
                                cusparseAction copyValues, IndexBase baseIdx, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCgebsr2gebsc(_handle, mb, nb, nnzb, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, bscVal.DevicePointer,
                bscRowInd.DevicePointer, bscColPtr.DevicePointer, copyValues, baseIdx, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgebsr2gebsc", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// This function can be seen as the same as csr2csc when regarding each block of size
        /// rowBlockDim*colBlockDim as a scalar.<para/>
        /// This sparsity pattern of result matrix can also be seen as the transpose of the original
        /// sparse matrix but memory layout of a block does not change.<para/>
        /// The user must know the size of buffer required by gebsr2gebsc by calling
        /// gebsr2gebsc_bufferSizeExt, allocate the buffer and pass the buffer pointer to gebsr2gebsc.
        /// </summary>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtr">integer array of mb+1 elements that contains the start of every block row and the end of the last block row plus one.</param>
        /// <param name="bsrColInd">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="bscVal">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A. It is only filled-in if copyValues is set to CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="bscRowInd">integer array of nnzb row indices of the non-zero blocks of matrix A</param>
        /// <param name="bscColPtr">integer array of nb+1 elements that contains the start of every block column and the end of the last block column plus one.</param>
        /// <param name="copyValues">CUSPARSE_ACTION_SYMBOLIC or CUSPARSE_ACTION_NUMERIC.</param>
        /// <param name="baseIdx">CUSPARSE_INDEX_BASE_ZERO or CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsc_bufferSizeExt.</param>
        public void Gebsr2gebsc(int mb, int nb, int nnzb, CudaDeviceVariable<cuDoubleComplex> bsrVal, CudaDeviceVariable<int> bsrRowPtr, CudaDeviceVariable<int> bsrColInd,
                                int rowBlockDim, int colBlockDim, CudaDeviceVariable<cuDoubleComplex> bscVal, CudaDeviceVariable<int> bscRowInd, CudaDeviceVariable<int> bscColPtr,
                                cusparseAction copyValues, IndexBase baseIdx, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZgebsr2gebsc(_handle, mb, nb, nnzb, bsrVal.DevicePointer, bsrRowPtr.DevicePointer, bsrColInd.DevicePointer, rowBlockDim, colBlockDim, bscVal.DevicePointer,
                bscRowInd.DevicePointer, bscColPtr.DevicePointer, copyValues, baseIdx, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgebsr2gebsc", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in CSR format
        /// (that is defined by arrays csrValC, csrRowPtrC, and csrColIndC).<para/>
        /// Let m(=mb*rowBlockDim) be number of rows of A and n(=nb*colBlockDim) be
        /// number of columns of A, then A and C are m*n sparse matrices. General BSR format of
        /// A contains nnzb(=bsrRowPtrA[mb] - bsrRowPtrA[0]) non-zero blocks whereas
        /// sparse matrix A contains nnz(=nnzb*rowBlockDim*colBockDim) elements. The user
        /// must allocate enough space for arrays csrRowPtrC, csrColIndC and csrValC. The
        /// requirements are<para/>
        /// csrRowPtrC of m+1 elements,<para/>
        /// csrValC of nnz elements, and<para/>
        /// csrColIndC of nnz elements.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrRowPtrC">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix C.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the nonzero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Gebsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA,
                              int rowBlockDim, int colBlockDim, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseXgebsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDim, colBlockDim, descrC.Descriptor,
                csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXgebsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in CSR format
        /// (that is defined by arrays csrValC, csrRowPtrC, and csrColIndC).<para/>
        /// Let m(=mb*rowBlockDim) be number of rows of A and n(=nb*colBlockDim) be
        /// number of columns of A, then A and C are m*n sparse matrices. General BSR format of
        /// A contains nnzb(=bsrRowPtrA[mb] - bsrRowPtrA[0]) non-zero blocks whereas
        /// sparse matrix A contains nnz(=nnzb*rowBlockDim*colBockDim) elements. The user
        /// must allocate enough space for arrays csrRowPtrC, csrColIndC and csrValC. The
        /// requirements are<para/>
        /// csrRowPtrC of m+1 elements,<para/>
        /// csrValC of nnz elements, and<para/>
        /// csrColIndC of nnz elements.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrValC">array of nnz non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix C.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the nonzero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Gebsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA,
                              CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDim, int colBlockDim,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseSgebsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDim, colBlockDim, descrC.Descriptor,
                csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgebsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }




        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in CSR format
        /// (that is defined by arrays csrValC, csrRowPtrC, and csrColIndC).<para/>
        /// Let m(=mb*rowBlockDim) be number of rows of A and n(=nb*colBlockDim) be
        /// number of columns of A, then A and C are m*n sparse matrices. General BSR format of
        /// A contains nnzb(=bsrRowPtrA[mb] - bsrRowPtrA[0]) non-zero blocks whereas
        /// sparse matrix A contains nnz(=nnzb*rowBlockDim*colBockDim) elements. The user
        /// must allocate enough space for arrays csrRowPtrC, csrColIndC and csrValC. The
        /// requirements are<para/>
        /// csrRowPtrC of m+1 elements,<para/>
        /// csrValC of nnz elements, and<para/>
        /// csrColIndC of nnz elements.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrValC">array of nnz non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix C.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the nonzero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Gebsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA,
                              CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDim, int colBlockDim,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseDgebsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDim, colBlockDim, descrC.Descriptor,
                csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgebsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }




        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in CSR format
        /// (that is defined by arrays csrValC, csrRowPtrC, and csrColIndC).<para/>
        /// Let m(=mb*rowBlockDim) be number of rows of A and n(=nb*colBlockDim) be
        /// number of columns of A, then A and C are m*n sparse matrices. General BSR format of
        /// A contains nnzb(=bsrRowPtrA[mb] - bsrRowPtrA[0]) non-zero blocks whereas
        /// sparse matrix A contains nnz(=nnzb*rowBlockDim*colBockDim) elements. The user
        /// must allocate enough space for arrays csrRowPtrC, csrColIndC and csrValC. The
        /// requirements are<para/>
        /// csrRowPtrC of m+1 elements,<para/>
        /// csrValC of nnz elements, and<para/>
        /// csrColIndC of nnz elements.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrValC">array of nnz non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix C.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the nonzero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Gebsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA,
                              CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDim, int colBlockDim,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuFloatComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseCgebsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDim, colBlockDim, descrC.Descriptor,
                csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgebsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }




        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in CSR format
        /// (that is defined by arrays csrValC, csrRowPtrC, and csrColIndC).<para/>
        /// Let m(=mb*rowBlockDim) be number of rows of A and n(=nb*colBlockDim) be
        /// number of columns of A, then A and C are m*n sparse matrices. General BSR format of
        /// A contains nnzb(=bsrRowPtrA[mb] - bsrRowPtrA[0]) non-zero blocks whereas
        /// sparse matrix A contains nnz(=nnzb*rowBlockDim*colBockDim) elements. The user
        /// must allocate enough space for arrays csrRowPtrC, csrColIndC and csrValC. The
        /// requirements are<para/>
        /// csrRowPtrC of m+1 elements,<para/>
        /// csrValC of nnz elements, and<para/>
        /// csrColIndC of nnz elements.
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDim*colBlockDim non-zero elements of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDim">number of rows within a block of A.</param>
        /// <param name="colBlockDim">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrValC">array of nnz non-zero elements of matrix C.</param>
        /// <param name="csrRowPtrC">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix C.</param>
        /// <param name="csrColIndC">integer array of nnz column indices of the nonzero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.4")]
        public void Gebsr2csr(cusparseDirection dirA, int mb, int nb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA,
                              CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int rowBlockDim, int colBlockDim,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuDoubleComplex> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            res = CudaSparseNativeMethods.cusparseZgebsr2csr(_handle, dirA, mb, nb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDim, colBlockDim, descrC.Descriptor,
                csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgebsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix A in CSR format (that is defined by arrays
        /// csrValA, csrRowPtrA, and csrColIndA) into a sparse matrix C in general BSR format
        /// (that is defined by the three arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A</param>
        /// <param name="csrColIndA">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <param name="nnzTotalDevHostPtr">total number of nonzero blocks of matrix C. <para/>
        /// Pointer nnzTotalDevHostPtr can point to a device memory or host memory.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by csr2gebsr_bufferSizeExt().</param>
        public void Csr2gebsrNnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
                                 CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> bsrRowPtrC, int rowBlockDim, int colBlockDim, CudaDeviceVariable<int> nnzTotalDevHostPtr, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseXcsr2gebsrNnz(_handle, dirA, m, n, descrA.Descriptor, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, descrC.Descriptor, bsrRowPtrC.DevicePointer, rowBlockDim, colBlockDim, nnzTotalDevHostPtr.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsr2gebsrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix A in CSR format (that is defined by arrays
        /// csrValA, csrRowPtrA, and csrColIndA) into a sparse matrix C in general BSR format
        /// (that is defined by the three arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A</param>
        /// <param name="csrColIndA">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <param name="nnzTotalDevHostPtr">total number of nonzero blocks of matrix C. <para/>
        /// Pointer nnzTotalDevHostPtr can point to a device memory or host memory.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by csr2gebsr_bufferSizeExt().</param>
        public void Csr2gebsrNnz(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
                                 CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> bsrRowPtrC, int rowBlockDim, int colBlockDim, ref int nnzTotalDevHostPtr, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseXcsr2gebsrNnz(_handle, dirA, m, n, descrA.Descriptor, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, descrC.Descriptor, bsrRowPtrC.DevicePointer, rowBlockDim, colBlockDim, ref nnzTotalDevHostPtr, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsr2gebsrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix A in CSR format (that is defined by arrays
        /// csrValA, csrRowPtrA, and csrColIndA) into a sparse matrix C in general BSR format
        /// (that is defined by the three arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A</param>
        /// <param name="csrColIndA">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <param name="bsrColIndC"><para/>
        /// Pointer nnzTotalDevHostPtr can point to a device memory or host memory.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by csr2gebsr_bufferSizeExt().</param>
        /// <param name="csrValA">array of nnz nonzero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzb*rowBlockDim*colBlockDim nonzero elements of matrix C.</param>
        public void Csr2gebsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDim, int colBlockDim, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseScsr2gebsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDim, colBlockDim, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix A in CSR format (that is defined by arrays
        /// csrValA, csrRowPtrA, and csrColIndA) into a sparse matrix C in general BSR format
        /// (that is defined by the three arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A</param>
        /// <param name="csrColIndA">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <param name="bsrColIndC"><para/>
        /// Pointer nnzTotalDevHostPtr can point to a device memory or host memory.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by csr2gebsr_bufferSizeExt().</param>
        /// <param name="csrValA">array of nnz nonzero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzb*rowBlockDim*colBlockDim nonzero elements of matrix C.</param>
        public void Csr2gebsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDim, int colBlockDim, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsr2gebsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDim, colBlockDim, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix A in CSR format (that is defined by arrays
        /// csrValA, csrRowPtrA, and csrColIndA) into a sparse matrix C in general BSR format
        /// (that is defined by the three arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A</param>
        /// <param name="csrColIndA">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <param name="bsrColIndC"><para/>
        /// Pointer nnzTotalDevHostPtr can point to a device memory or host memory.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by csr2gebsr_bufferSizeExt().</param>
        /// <param name="csrValA">array of nnz nonzero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzb*rowBlockDim*colBlockDim nonzero elements of matrix C.</param>
        public void Csr2gebsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuFloatComplex> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDim, int colBlockDim, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsr2gebsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDim, colBlockDim, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function converts a sparse matrix A in CSR format (that is defined by arrays
        /// csrValA, csrRowPtrA, and csrColIndA) into a sparse matrix C in general BSR format
        /// (that is defined by the three arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="m">number of rows of sparse matrix A.</param>
        /// <param name="n">number of columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the
        /// start of every row and the end of the last row plus one of matrix A</param>
        /// <param name="csrColIndA">integer array of nnz column indices of the nonzero elements of matrix A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDim">number of rows within a block of C.</param>
        /// <param name="colBlockDim">number of columns within a block of C.</param>
        /// <param name="bsrColIndC"><para/>
        /// Pointer nnzTotalDevHostPtr can point to a device memory or host memory.</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by csr2gebsr_bufferSizeExt().</param>
        /// <param name="csrValA">array of nnz nonzero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzb*rowBlockDim*colBlockDim nonzero elements of matrix C.</param>
        public void Csr2gebsr(cusparseDirection dirA, int m, int n, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
                              CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuDoubleComplex> bsrValC, CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDim, int colBlockDim, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsr2gebsr(_handle, dirA, m, n, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDim, colBlockDim, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in another general
        /// BSR format (that is defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mc+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C</param>
        /// <param name="colBlockDimC">number of columns within a block of C</param>
        /// <param name="nnzTotalDevHostPtr">total number of nonzero blocks of C.<para/>
        /// nnzTotalDevHostPtr is the same as bsrRowPtrC[mc]-bsrRowPtrC[0]</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsr_bufferSizeExt.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Gebsr2gebsrNnz(cusparseDirection dirA, int mb, int nb, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA,
                                   int rowBlockDimA, int colBlockDimA, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> bsrRowPtrC, int rowBlockDimC, int colBlockDimC, CudaDeviceVariable<int> nnzTotalDevHostPtr, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseXgebsr2gebsrNnz(_handle, dirA, mb, nb, nnzb, descrA.Descriptor, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA, descrC.Descriptor, bsrRowPtrC.DevicePointer, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr.DevicePointer, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXgebsr2gebsrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in another general
        /// BSR format (that is defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mc+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C</param>
        /// <param name="colBlockDimC">number of columns within a block of C</param>
        /// <param name="nnzTotalDevHostPtr">total number of nonzero blocks of C.<para/>
        /// nnzTotalDevHostPtr is the same as bsrRowPtrC[mc]-bsrRowPtrC[0]</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsr_bufferSizeExt.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Gebsr2gebsrNnz(cusparseDirection dirA, int mb, int nb, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA,
                                   int rowBlockDimA, int colBlockDimA, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> bsrRowPtrC, int rowBlockDimC, int colBlockDimC, ref int nnzTotalDevHostPtr, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseXgebsr2gebsrNnz(_handle, dirA, mb, nb, nnzb, descrA.Descriptor, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA, descrC.Descriptor, bsrRowPtrC.DevicePointer, rowBlockDimC, colBlockDimC, ref nnzTotalDevHostPtr, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXgebsr2gebsrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in another general
        /// BSR format (that is defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mc+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="bsrColIndC">integer array of nnzc block column indices of the non-zero blocks of matrix C.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C</param>
        /// <param name="colBlockDimC">number of columns within a block of C</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsr_bufferSizeExt.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzc*rowBlockDimC*colBlockDimC non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Gebsr2gebsr(cusparseDirection dirA, int mb, int nb, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> bsrValC,
                                CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDimC, int colBlockDimC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseSgebsr2gebsr(_handle, dirA, mb, nb, nnzb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA,
                descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDimC, colBlockDimC, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSgebsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in another general
        /// BSR format (that is defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mc+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="bsrColIndC">integer array of nnzc block column indices of the non-zero blocks of matrix C.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C</param>
        /// <param name="colBlockDimC">number of columns within a block of C</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsr_bufferSizeExt.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzc*rowBlockDimC*colBlockDimC non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Gebsr2gebsr(cusparseDirection dirA, int mb, int nb, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> bsrValC,
                                CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDimC, int colBlockDimC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseDgebsr2gebsr(_handle, dirA, mb, nb, nnzb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA,
                descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDimC, colBlockDimC, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDgebsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in another general
        /// BSR format (that is defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mc+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="bsrColIndC">integer array of nnzc block column indices of the non-zero blocks of matrix C.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C</param>
        /// <param name="colBlockDimC">number of columns within a block of C</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsr_bufferSizeExt.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzc*rowBlockDimC*colBlockDimC non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Gebsr2gebsr(cusparseDirection dirA, int mb, int nb, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuFloatComplex> bsrValC,
                                CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDimC, int colBlockDimC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseCgebsr2gebsr(_handle, dirA, mb, nb, nnzb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA,
                descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDimC, colBlockDimC, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCgebsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// This function converts a sparse matrix in general BSR format (that is defined by the three
        /// arrays bsrValA, bsrRowPtrA, and bsrColIndA) into a sparse matrix in another general
        /// BSR format (that is defined by arrays bsrValC, bsrRowPtrC, and bsrColIndC).
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="mb">number of block rows of sparse matrix A.</param>
        /// <param name="nb">number of block columns of sparse matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="nnzb">number of nonzero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix A.</param>
        /// <param name="bsrColIndA">integer array of nnzb column indices of the nonzero blocks of matrix A.</param>
        /// <param name="rowBlockDimA">number of rows within a block of A.</param>
        /// <param name="colBlockDimA">number of columns within a block of A.</param>
        /// <param name="descrC">the descriptor of matrix C. The supported matrix 
        /// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are
        /// CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrRowPtrC">integer array of mc+1 elements that contains the
        /// start of every block row and the end of the last block row plus one of matrix C.</param>
        /// <param name="bsrColIndC">integer array of nnzc block column indices of the non-zero blocks of matrix C.</param>
        /// <param name="rowBlockDimC">number of rows within a block of C</param>
        /// <param name="colBlockDimC">number of columns within a block of C</param>
        /// <param name="buffer">buffer allocated by the user, the size is return by gebsr2gebsr_bufferSizeExt.</param>
        /// <param name="bsrValA">array of nnzb*rowBlockDimA*colBlockDimA non-zero elements of matrix A.</param>
        /// <param name="bsrValC">array of nnzc*rowBlockDimC*colBlockDimC non-zero elements of matrix C.</param>
        [Obsolete("Deprecated in Cuda 12.8")]
        public void Gebsr2gebsr(cusparseDirection dirA, int mb, int nb, int nnzb, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA,
                                CudaDeviceVariable<int> bsrColIndA, int rowBlockDimA, int colBlockDimA, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<cuDoubleComplex> bsrValC,
                                CudaDeviceVariable<int> bsrRowPtrC, CudaDeviceVariable<int> bsrColIndC, int rowBlockDimC, int colBlockDimC, CudaDeviceVariable<byte> buffer)
        {
            res = CudaSparseNativeMethods.cusparseZgebsr2gebsr(_handle, dirA, mb, nb, nnzb, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, rowBlockDimA, colBlockDimA,
                descrC.Descriptor, bsrValC.DevicePointer, bsrRowPtrC.DevicePointer, bsrColIndC.DevicePointer, rowBlockDimC, colBlockDimC, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZgebsr2gebsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion




        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, float alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<float> x, float beta, CudaDeviceVariable<float> y)
        {
            res = CudaSparseNativeMethods.cusparseSbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, ref beta, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, double alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<double> x, double beta, CudaDeviceVariable<double> y)
        {
            res = CudaSparseNativeMethods.cusparseDbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, ref beta, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, cuFloatComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<cuFloatComplex> x, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseCbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, ref beta, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, cuDoubleComplex alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<cuDoubleComplex> x, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseZbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, ref alpha, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, ref beta, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, CudaDeviceVariable<float> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<float> x, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> y)
        {
            res = CudaSparseNativeMethods.cusparseSbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, beta.DevicePointer, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, CudaDeviceVariable<double> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<double> x, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> y)
        {
            res = CudaSparseNativeMethods.cusparseDbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, beta.DevicePointer, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, CudaDeviceVariable<cuFloatComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<cuFloatComplex> x, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseCbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, beta.DevicePointer, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// This function performs the matrix-vector operation <para/>
        /// y = alpha * op(A) * x + beta * y<para/>
        /// where A is (mb*blockDim) x (nb*blockDim) sparse matrix (that is defined in BSR
        /// storage format by the three arrays bsrVal, bsrRowPtr, and bsrColInd), x and y are
        /// vectors, alpha and beta are scalars. 
        /// </summary>
        /// <param name="dirA">storage format of blocks, either CUSPARSE_DIRECTION_ROW or CUSPARSE_DIRECTION_COLUMN.</param>
        /// <param name="transA">the operation op(A). Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported.</param>
        /// <param name="mb">number of block rows of matrix A.</param>
        /// <param name="nb">number of block columns of matrix A.</param>
        /// <param name="alpha">scalar used for multiplication.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases
        /// are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="bsrValA">array of nnzb (= bsrRowPtr(mb) - bsrRowPtr(0)) non-zero blocks of matrix A.</param>
        /// <param name="bsrRowPtrA">integer array of mb+1 elements that contains the start of every block
        /// row and the end of the last block row plus one.</param>
        /// <param name="bsrColIndA">integer array of nnzb (= bsrRowPtr(m) - bsrRowPtr(0)) column indices of the non-zero blocks of matrix A.
        /// Length of bsrColIndA gives the number nzzb passed to CUSPARSE.</param>
        /// <param name="blockDim">block dimension of sparse matrix A, larger than zero.</param>
        /// <param name="x">vector of nb*blockDim elements.</param>
        /// <param name="beta">scalar used for multiplication. If beta is zero, y does not have to be a valid input.</param>
        /// <param name="y">vector of mb*blockDim element.</param>
        public void Bsrmv(cusparseDirection dirA, cusparseOperation transA, int mb, int nb, CudaDeviceVariable<cuDoubleComplex> alpha, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> bsrValA, CudaDeviceVariable<int> bsrRowPtrA, CudaDeviceVariable<int> bsrColIndA, int blockDim, CudaDeviceVariable<cuDoubleComplex> x, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y)
        {
            res = CudaSparseNativeMethods.cusparseZbsrmv(_handle, dirA, transA, mb, nb, (int)bsrColIndA.Size, alpha.DevicePointer, descrA.Descriptor, bsrValA.DevicePointer, bsrRowPtrA.DevicePointer, bsrColIndA.DevicePointer, blockDim, x.DevicePointer, beta.DevicePointer, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZbsrmv", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }














        /* --- Sparse Matrix Sorting --- */

        /* Description: Create a identity sequence p=[0,1,...,n-1]. */

        /// <summary>
        /// This function creates an identity map. The output parameter p represents such map by p = 0:1:(n-1).<para/>
        /// This function is typically used with coosort, csrsort, cscsort, csr2csc_indexOnly.
        /// </summary>
        /// <param name="n">size of the map.</param>
        /// <param name="p">integer array of dimensions n.</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void CreateIdentityPermutation(int n, CudaDeviceVariable<int> p)
        {
            res = CudaSparseNativeMethods.cusparseCreateIdentityPermutation(_handle, n, p.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateIdentityPermutation", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /* Description: Sort sparse matrix stored in COO format */

        /// <summary>
        /// This function sorts COO format. The stable sorting is in-place. Also the user can sort by row or sort by column.<para/>
        /// A is an m x n sparse matrix that is defined in COO storage format by the three arrays cooVals, cooRows, and cooCols.<para/>
        /// The matrix must be base 0.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="cooRowsA">integer array of nnz unsorted row indices of A.</param>
        /// <param name="cooColsA">integer array of nnz unsorted column indices of A.</param>
        /// <returns>number of bytes of the buffer.</returns>
        public SizeT CoosortBufferSize(int m, int n, int nnz, CudaDeviceVariable<int> cooRowsA, CudaDeviceVariable<int> cooColsA)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseXcoosort_bufferSizeExt(_handle, m, n, nnz, cooRowsA.DevicePointer, cooColsA.DevicePointer, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcoosort_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        /// <summary>
        /// This function sorts COO format. The stable sorting is in-place. Also the user can sort by row or sort by column.<para/>
        /// A is an m x n sparse matrix that is defined in COO storage format by the three arrays cooVals, cooRows, and cooCols.<para/>
        /// The matrix must be base 0.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="cooRowsA">integer array of nnz unsorted row indices of A.</param>
        /// <param name="cooColsA">integer array of nnz unsorted column indices of A.</param>
        /// <param name="P">integer array of nnz sorted map indices.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by CoosortBufferSize().</param>
        public void CoosortByRow(int m, int n, int nnz, CudaDeviceVariable<int> cooRowsA, CudaDeviceVariable<int> cooColsA, CudaDeviceVariable<int> P, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseXcoosortByRow(_handle, m, n, nnz, cooRowsA.DevicePointer, cooColsA.DevicePointer, P.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcoosortByRow", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function sorts COO format. The stable sorting is in-place. Also the user can sort by row or sort by column.<para/>
        /// A is an m x n sparse matrix that is defined in COO storage format by the three arrays cooVals, cooRows, and cooCols.<para/>
        /// The matrix must be base 0.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="cooRowsA">integer array of nnz unsorted row indices of A.</param>
        /// <param name="cooColsA">integer array of nnz unsorted column indices of A.</param>
        /// <param name="P">integer array of nnz sorted map indices.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by CoosortBufferSize().</param>
        public void CoosortByColumn(int m, int n, int nnz, CudaDeviceVariable<int> cooRowsA, CudaDeviceVariable<int> cooColsA, CudaDeviceVariable<int> P, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseXcoosortByColumn(_handle, m, n, nnz, cooRowsA.DevicePointer, cooColsA.DevicePointer, P.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcoosortByColumn", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /* Description: Sort sparse matrix stored in CSR format */

        /// <summary>
        /// This function sorts CSR format. The stable sorting is in-place.<para/>
        /// The matrix type is regarded as CUSPARSE_MATRIX_TYPE_GENERAL implicitly. In other
        /// words, any symmetric property is ignored.<para/>
        /// This function csrsort() requires buffer size returned by csrsort_bufferSizeExt().<para/>
        /// The address of pBuffer must be multiple of 128 bytes. If not,
        /// CUSPARSE_STATUS_INVALID_VALUE is returned.<para/>
        /// The parameter P is both input and output. If the user wants to compute sorted csrVal,
        /// P must be set as 0:1:(nnz-1) before csrsort(), and after csrsort(), new sorted value
        /// array satisfies csrVal_sorted = csrVal(P).
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz unsorted column indices of A.</param>
        /// <returns>number of bytes of the buffer.</returns>
        public SizeT CsrsortBufferSize(int m, int n, int nnz, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseXcsrsort_bufferSizeExt(_handle, m, n, nnz, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsrsort_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        /// <summary>
        /// This function sorts CSR format. The stable sorting is in-place.<para/>
        /// The matrix type is regarded as CUSPARSE_MATRIX_TYPE_GENERAL implicitly. In other
        /// words, any symmetric property is ignored.<para/>
        /// This function csrsort() requires buffer size returned by csrsort_bufferSizeExt().<para/>
        /// The address of pBuffer must be multiple of 128 bytes. If not,
        /// CUSPARSE_STATUS_INVALID_VALUE is returned.<para/>
        /// The parameter P is both input and output. If the user wants to compute sorted csrVal,
        /// P must be set as 0:1:(nnz-1) before csrsort(), and after csrsort(), new sorted value
        /// array satisfies csrVal_sorted = csrVal(P).
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="csrRowPtrA">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColIndA">integer array of nnz unsorted column indices of A.</param>
        /// <param name="P">integer array of nnz sorted map indices.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by CsrsortBufferSize().</param>
        public void Csrsort(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<int> P, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseXcsrsort(_handle, m, n, nnz, descrA.Descriptor, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, P.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcsrsort", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /* Description: Sort sparse matrix stored in CSC format */

        /// <summary>
        /// This function sorts CSC format. The stable sorting is in-place.<para/>
        /// The matrix type is regarded as CUSPARSE_MATRIX_TYPE_GENERAL implicitly. In other
        /// words, any symmetric property is ignored. <para/>
        /// This function cscsort() requires buffer size returned by cscsort_bufferSizeExt().
        /// The address of pBuffer must be multiple of 128 bytes. If not,
        /// CUSPARSE_STATUS_INVALID_VALUE is returned.<para/>
        /// The parameter P is both input and output. If the user wants to compute sorted cscVal,
        /// P must be set as 0:1:(nnz-1) before cscsort(), and after cscsort(), new sorted value
        /// array satisfies cscVal_sorted = cscVal(P).
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="cscColPtrA">integer array of n+1 elements that contains the start of every column and the end of the last column plus one.</param>
        /// <param name="cscRowIndA">integer array of nnz unsorted row indices of A.</param>
        /// <returns>number of bytes of the buffer.</returns>
        public SizeT CscsortBufferSize(int m, int n, int nnz, CudaDeviceVariable<int> cscColPtrA, CudaDeviceVariable<int> cscRowIndA)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseXcscsort_bufferSizeExt(_handle, m, n, nnz, cscColPtrA.DevicePointer, cscRowIndA.DevicePointer, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcscsort_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        /// <summary>
        /// This function sorts CSC format. The stable sorting is in-place.<para/>
        /// The matrix type is regarded as CUSPARSE_MATRIX_TYPE_GENERAL implicitly. In other
        /// words, any symmetric property is ignored. <para/>
        /// This function cscsort() requires buffer size returned by cscsort_bufferSizeExt().
        /// The address of pBuffer must be multiple of 128 bytes. If not,
        /// CUSPARSE_STATUS_INVALID_VALUE is returned.<para/>
        /// The parameter P is both input and output. If the user wants to compute sorted cscVal,
        /// P must be set as 0:1:(nnz-1) before cscsort(), and after cscsort(), new sorted value
        /// array satisfies cscVal_sorted = cscVal(P).
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A.</param>
        /// <param name="cscColPtrA">integer array of n+1 elements that contains the start of every column and the end of the last column plus one.</param>
        /// <param name="cscRowIndA">integer array of nnz unsorted row indices of A.</param>
        /// <param name="P">integer array of nnz sorted map indices.</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by CscsortBufferSize().</param>
        public void Cscsort(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> cscColPtrA, CudaDeviceVariable<int> cscRowIndA, CudaDeviceVariable<int> P, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseXcscsort(_handle, m, n, nnz, descrA.Descriptor, cscColPtrA.DevicePointer, cscRowIndA.DevicePointer, P.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseXcscsort", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /* Description: Wrapper that sorts sparse matrix stored in CSR format 
		   (without exposing the permutation). */

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <returns>number of bytes of the buffer.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csru2csrBufferSize(int m, int n, int nnz, CudaDeviceVariable<float> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseScsru2csr_bufferSizeExt(_handle, m, n, nnz, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsru2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <returns>number of bytes of the buffer.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csru2csrBufferSize(int m, int n, int nnz, CudaDeviceVariable<double> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseDcsru2csr_bufferSizeExt(_handle, m, n, nnz, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsru2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <returns>number of bytes of the buffer.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csru2csrBufferSize(int m, int n, int nnz, CudaDeviceVariable<cuFloatComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseCcsru2csr_bufferSizeExt(_handle, m, n, nnz, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsru2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <returns>number of bytes of the buffer.</returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT Csru2csrBufferSize(int m, int n, int nnz, CudaDeviceVariable<cuDoubleComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info)
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseZcsru2csr_bufferSizeExt(_handle, m, n, nnz, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsru2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csru2csr(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseScsru2csr(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsru2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csru2csr(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsru2csr(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsru2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csru2csr(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsru2csr(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsru2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csru2csr(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsru2csr(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsru2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /* Description: Wrapper that un-sorts sparse matrix stored in CSR format 
		   (without exposing the permutation). */

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csru(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseScsr2csru(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScsr2csru", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csru(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDcsr2csru(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDcsr2csru", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csru(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseCcsr2csru(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCcsr2csru", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// This function transfers unsorted CSR format to CSR format, and vice versa. The operation is in-place.
        /// </summary>
        /// <param name="m">number of rows of matrix A.</param>
        /// <param name="n">number of columns of matrix A.</param>
        /// <param name="nnz">number of nonzero elements of matrix A.</param>
        /// <param name="descrA">the descriptor of matrix A. The supported matrix type is
        /// CUSPARSE_MATRIX_TYPE_GENERAL, Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
        /// <param name="csrVal">array of nnz unsorted nonzero elements of matrix A.</param>
        /// <param name="csrRowPtr">integer array of m+1 elements that contains the start of every row and the end of the last row plus one.</param>
        /// <param name="csrColInd">integer array of nnz unsorted column indices of A.</param>
        /// <param name="info">opaque structure initialized using cusparseCreateCsru2csrInfo().</param>
        /// <param name="pBuffer">buffer allocated by the user; the size is returned by Csru2csrBufferSize().</param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void Csr2csru(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, CudaSparseCsru2csrInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseZcsr2csru(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, info.Csru2csrInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseZcsr2csru", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        #endregion

        #region Prune
        #region No host/device pointer ambiguity
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format by percentage
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrByPercentageBufferSizeExt(int m, int n, CudaDeviceVariable<half> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseHpruneDense2csrByPercentage_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format by percentage
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrByPercentageBufferSizeExt(int m, int n, CudaDeviceVariable<float> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSpruneDense2csrByPercentage_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format by percentage
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrByPercentageBufferSizeExtt(int m, int n, CudaDeviceVariable<double> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDpruneDense2csrByPercentage_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }





        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrByPercentage(int m, int n, CudaDeviceVariable<half> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<half> csrValC,
            CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseHpruneDense2csrByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csrByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrByPercentage(int m, int n, CudaDeviceVariable<float> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC,
            CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSpruneDense2csrByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csrByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrByPercentage(int m, int n, CudaDeviceVariable<double> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC,
            CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDpruneDense2csrByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csrByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// Description: prune sparse matrix to a sparse matrix with CSR format by percentage
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrByPercentageBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csrByPercentage_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// Description: prune sparse matrix to a sparse matrix with CSR format by percentage
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrByPercentageBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csrByPercentage_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// Description: prune sparse matrix to a sparse matrix with CSR format by percentage
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrByPercentageBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csrByPercentage_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }





        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, float percentage,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csrByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, float percentage,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csrByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, float percentage,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csrByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion
        #region Host pointers
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrBufferSize(int m, int n, CudaDeviceVariable<half> A, int lda, half threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseHpruneDense2csr_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrBufferSize(int m, int n, CudaDeviceVariable<float> A, int lda, float threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSpruneDense2csr_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrBufferSize(int m, int n, CudaDeviceVariable<double> A, int lda, double threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDpruneDense2csr_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneDense2csrNnz(int m, int n, CudaDeviceVariable<half> A, int lda, half threshold,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseHpruneDense2csrNnz(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneDense2csrNnz(int m, int n, CudaDeviceVariable<float> A, int lda, float threshold,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseSpruneDense2csrNnz(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneDense2csrNnz(int m, int n, CudaDeviceVariable<double> A, int lda, double threshold,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseDpruneDense2csrNnz(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csr(int m, int n, CudaDeviceVariable<half> A, int lda, half threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseHpruneDense2csr(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csr(int m, int n, CudaDeviceVariable<float> A, int lda, float threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSpruneDense2csr(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csr(int m, int n, CudaDeviceVariable<double> A, int lda, double threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDpruneDense2csr(_handle, m, n, A.DevicePointer, lda, ref threshold, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// Description: prune sparse matrix with CSR format to another sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            half threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csr_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// Description: prune sparse matrix with CSR format to another sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csr_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// Description: prune sparse matrix with CSR format to another sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            double threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csr_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }



        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneCsr2csrNnz(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            half threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csrNnz(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneCsr2csrNnz(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csrNnz(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneCsr2csrNnz(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            double threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csrNnz(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }



        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csr(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            half threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csr(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csr(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csr(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csr(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            double threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csr(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, ref threshold,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }




        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneDense2csrNnzByPercentage(int m, int n, CudaDeviceVariable<half> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseHpruneDense2csrNnzByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneDense2csrNnzByPercentage(int m, int n, CudaDeviceVariable<float> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseSpruneDense2csrNnzByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneDense2csrNnzByPercentage(int m, int n, CudaDeviceVariable<double> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseDpruneDense2csrNnzByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }






        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneCsr2csrNnzByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csrNnzByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneCsr2csrNnzByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csrNnzByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public int PruneCsr2csrNnzByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer)
        {
            int ret = 0;
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csrNnzByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, ref ret, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        #endregion
        #region Device pointers

        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrBufferSize(int m, int n, CudaDeviceVariable<half> A, int lda, CudaDeviceVariable<half> threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseHpruneDense2csr_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrBufferSize(int m, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSpruneDense2csr_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }
        /// <summary>
        /// Description: prune dense matrix to a sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneDense2csrBufferSize(int m, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDpruneDense2csr_bufferSizeExt(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csr_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrNnz(int m, int n, CudaDeviceVariable<half> A, int lda, CudaDeviceVariable<half> threshold,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseHpruneDense2csrNnz(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrNnz(int m, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> threshold,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseSpruneDense2csrNnz(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrNnz(int m, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> threshold,
            CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseDpruneDense2csrNnz(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csr(int m, int n, CudaDeviceVariable<half> A, int lda, CudaDeviceVariable<half> threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseHpruneDense2csr(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csr(int m, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSpruneDense2csr(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csr(int m, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> threshold, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDpruneDense2csr(_handle, m, n, A.DevicePointer, lda, threshold.DevicePointer, descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// Description: prune sparse matrix with CSR format to another sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<half> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csr_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// Description: prune sparse matrix with CSR format to another sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<float> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csr_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }

        /// <summary>
        /// Description: prune sparse matrix with CSR format to another sparse matrix with CSR format
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <returns></returns>
        [Obsolete("Deprecated in Cuda 12.3")]
        public SizeT PruneCsr2csrBufferSizeExt(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<double> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC)
        {
            SizeT ret = new SizeT();
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csr_bufferSizeExt(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrByPercentage_bufferSizeExt", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return ret;
        }



        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrNnz(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<half> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csrNnz(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrNnz(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<float> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csrNnz(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrNnz(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<double> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csrNnz(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrNnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }



        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csr(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<half> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<half> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csr(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csr(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<float> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<float> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csr(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="threshold"></param>
        /// <param name="descrC"></param>
        /// <param name="csrValC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="csrColIndC"></param>
        /// <param name="pBuffer"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csr(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            CudaDeviceVariable<double> threshold, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<double> csrValC, CudaDeviceVariable<int> csrRowPtrC, CudaDeviceVariable<int> csrColIndC, CudaDeviceVariable<byte> pBuffer)
        {
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csr(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, threshold.DevicePointer,
                descrC.Descriptor, csrValC.DevicePointer, csrRowPtrC.DevicePointer, csrColIndC.DevicePointer, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }




        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrNnzByPercentage(int m, int n, CudaDeviceVariable<half> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseHpruneDense2csrNnzByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneDense2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrNnzByPercentage(int m, int n, CudaDeviceVariable<float> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseSpruneDense2csrNnzByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneDense2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="A"></param>
        /// <param name="lda"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneDense2csrNnzByPercentage(int m, int n, CudaDeviceVariable<double> A, int lda, float percentage, CudaSparseMatrixDescriptor descrC,
            CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseDpruneDense2csrNnzByPercentage(_handle, m, n, A.DevicePointer, lda, percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneDense2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }






        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrNnzByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<half> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseHpruneCsr2csrNnzByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseHpruneCsr2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrNnzByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseSpruneCsr2csrNnzByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpruneCsr2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <param name="nnzA"></param>
        /// <param name="descrA"></param>
        /// <param name="csrValA"></param>
        /// <param name="csrRowPtrA"></param>
        /// <param name="csrColIndA"></param>
        /// <param name="percentage"></param>
        /// <param name="descrC"></param>
        /// <param name="csrRowPtrC"></param>
        /// <param name="info"></param>
        /// <param name="pBuffer"></param>
        /// <param name="nnzTotalDevHostPtr"></param>
        [Obsolete("Deprecated in Cuda 12.3")]
        public void PruneCsr2csrNnzByPercentage(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA,
            float percentage, CudaSparseMatrixDescriptor descrC, CudaDeviceVariable<int> csrRowPtrC, CudaSparsePruneInfo info, CudaDeviceVariable<byte> pBuffer, CudaDeviceVariable<int> nnzTotalDevHostPtr)
        {
            res = CudaSparseNativeMethods.cusparseDpruneCsr2csrNnzByPercentage(_handle, m, n, nnzA, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer,
                percentage, descrC.Descriptor, csrRowPtrC.DevicePointer, nnzTotalDevHostPtr.DevicePointer, info.pruneInfo, pBuffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDpruneCsr2csrNnzByPercentage", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        #endregion



        #endregion

        #region Generic API

        #region Vector Vector

        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Axpby<indexT, dataTIn>(CudaDeviceVariable<dataTIn> alpha, ConstSparseVector<indexT, dataTIn> vecX,
            CudaDeviceVariable<dataTIn> beta, DenseVector<dataTIn> vecY)
            where indexT : struct where dataTIn : struct
        {
            res = CudaSparseNativeMethods.cusparseAxpby(_handle, alpha.DevicePointer, vecX.Descr, beta.DevicePointer, vecY.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseAxpby", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void Axpby<indexT, dataTIn>(dataTIn alpha, ConstSparseVector<indexT, dataTIn> vecX,
            dataTIn beta, DenseVector<dataTIn> vecY)
            where indexT : struct where dataTIn : struct
        {
            IntPtr ptr = IntPtr.Zero;
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                int size = CudaDataTypeTranslator.GetSize(typeof(dataTIn));
                ptr = Marshal.AllocHGlobal(2 * size);
                ptrAlpha = ptr + 0;
                ptrBeta = ptr + size;

                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseAxpby(_handle, ptrAlpha, vecX.Descr, ptrBeta, vecY.Descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseAxpby", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }

        public void Gather<indexT, dataTIn>(ConstDenseVector<dataTIn> vecY, SparseVector<indexT, dataTIn> vecX)
            where indexT : struct where dataTIn : struct
        {
            res = CudaSparseNativeMethods.cusparseGather(_handle, vecY.Descr, vecX.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGather", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void Scatter<indexT, dataTIn>(ConstSparseVector<indexT, dataTIn> vecX, DenseVector<dataTIn> vecY)
            where indexT : struct where dataTIn : struct
        {
            res = CudaSparseNativeMethods.cusparseScatter(_handle, vecX.Descr, vecY.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseScatter", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        [Obsolete("Deprecated in Cuda 12.3")]
        public void Rot<indexT, dataTIn>(CudaDeviceVariable<dataTIn> c_coeff, SparseVector<indexT, dataTIn> vecX,
            CudaDeviceVariable<dataTIn> s_coeff, DenseVector<dataTIn> vecY)
            where indexT : struct where dataTIn : struct
        {
            res = CudaSparseNativeMethods.cusparseRot(_handle, c_coeff.DevicePointer, s_coeff.DevicePointer, vecX.Descr, vecY.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseRot", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        [Obsolete("Deprecated in Cuda 12.3")]
        public void Rot<indexT, dataTIn>(dataTIn c_coeff, SparseVector<indexT, dataTIn> vecX,
            dataTIn s_coeff, DenseVector<dataTIn> vecY)
            where indexT : struct where dataTIn : struct
        {
            IntPtr ptr = IntPtr.Zero;
            IntPtr ptrCCoeff = IntPtr.Zero;
            IntPtr ptrSCoeff = IntPtr.Zero;
            try
            {
                int size = CudaDataTypeTranslator.GetSize(typeof(dataTIn));
                ptr = Marshal.AllocHGlobal(2 * size);
                ptrCCoeff = ptr + 0;
                ptrSCoeff = ptr + size;

                Marshal.StructureToPtr(c_coeff, ptrCCoeff, false);
                Marshal.StructureToPtr(s_coeff, ptrSCoeff, false);
                res = CudaSparseNativeMethods.cusparseRot(_handle, ptrCCoeff, ptrSCoeff, vecX.Descr, vecY.Descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseRot", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }

        [Obsolete("Marked deprecated in Cuda 12.8")]
        public SizeT VV_bufferSize<indexT, dataTIn, dataTCompute>(cusparseOperation opX, ConstSparseVector<indexT, dataTIn> vecX,
            ConstDenseVector<dataTIn> vecY, CudaDeviceVariable<dataTCompute> result)
            where indexT : struct where dataTIn : struct where dataTCompute : struct
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSpVV_bufferSize(_handle, opX, vecX.Descr, vecY.Descr, result.DevicePointer, CudaDataTypeTranslator.GetType(typeof(dataTCompute)), ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVV_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        [Obsolete("Marked deprecated in Cuda 12.8")]
        public SizeT VV_bufferSize<indexT, dataTIn, dataTCompute>(cusparseOperation opX, ConstSparseVector<indexT, dataTIn> vecX,
            ConstDenseVector<dataTIn> vecY, ref dataTCompute result)
            where indexT : struct where dataTIn : struct where dataTCompute : struct
        {
            SizeT size = 0;
            IntPtr ptr = IntPtr.Zero;
            try
            {

                ptr = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTCompute)));
                //Marshal.StructureToPtr(result, ptr, false);
                res = CudaSparseNativeMethods.cusparseSpVV_bufferSize(_handle, opX, vecX.Descr, vecY.Descr, ptr, CudaDataTypeTranslator.GetType(typeof(dataTCompute)), ref size);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVV_bufferSize", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptr);
            }
            return size;
        }
        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void VV<indexT, dataTIn, dataTCompute>(cusparseOperation opX, ConstSparseVector<indexT, dataTIn> vecX,
            ConstDenseVector<dataTIn> vecY, CudaDeviceVariable<dataTCompute> result, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTIn : struct where dataTCompute : struct
        {
            res = CudaSparseNativeMethods.cusparseSpVV(_handle, opX, vecX.Descr, vecY.Descr, result.DevicePointer, CudaDataTypeTranslator.GetType(typeof(dataTCompute)), buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVV", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        [Obsolete("Marked deprecated in Cuda 12.8")]
        public void VV<indexT, dataTIn, dataTCompute>(cusparseOperation opX, ConstSparseVector<indexT, dataTIn> vecX,
            ConstDenseVector<dataTIn> vecY, ref dataTCompute result, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTIn : struct where dataTCompute : struct
        {
            IntPtr ptr = IntPtr.Zero;
            try
            {

                ptr = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTCompute)));
                //Marshal.StructureToPtr(result, ptr, false);
                res = CudaSparseNativeMethods.cusparseSpVV(_handle, opX, vecX.Descr, vecY.Descr, ptr, CudaDataTypeTranslator.GetType(typeof(dataTCompute)), buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVV", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptr);
            }
        }
        #endregion

        #region Matrix Vector
        public SizeT MV_bufferSize<indexT, dataTAX, dataTY>(cusparseOperation opA, CudaDeviceVariable<dataTAX> alpha, ConstSparseMatrix<indexT, dataTAX> matA,
            ConstDenseVector<dataTAX> vecX, CudaDeviceVariable<dataTY> beta, DenseVector<dataTY> vecY, cudaDataType computeType, SpMVAlg alg)
            where indexT : struct where dataTAX : struct where dataTY : struct
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSpMV_bufferSize(_handle, opA, alpha.DevicePointer, matA.Descr, vecX.Descr, beta.DevicePointer, vecY.Descr, computeType, alg, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMV_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public SizeT MV_bufferSize<indexT, dataTAX, dataTY>(cusparseOperation opA, dataTAX alpha, ConstSparseMatrix<indexT, dataTAX> matA,
            ConstDenseVector<dataTAX> vecX, dataTY beta, DenseVector<dataTY> vecY, cudaDataType computeType, SpMVAlg alg)
            where indexT : struct where dataTAX : struct where dataTY : struct
        {
            SizeT size = 0;
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAX)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTY)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpMV_bufferSize(_handle, opA, ptrAlpha, matA.Descr, vecX.Descr, ptrBeta, vecY.Descr, computeType, alg, ref size);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMV_bufferSize", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
            return size;
        }
        public SizeT MV_preprocess<indexT, dataTAX, dataTY>(cusparseOperation opA, CudaDeviceVariable<dataTAX> alpha, ConstSparseMatrix<indexT, dataTAX> matA,
            ConstDenseVector<dataTAX> vecX, CudaDeviceVariable<dataTY> beta, DenseVector<dataTY> vecY, cudaDataType computeType, SpMVAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAX : struct where dataTY : struct
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSpMV_preprocess(_handle, opA, alpha.DevicePointer, matA.Descr, vecX.Descr, beta.DevicePointer, vecY.Descr, computeType, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMV_preprocess", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public SizeT MV_preprocess<indexT, dataTAX, dataTY>(cusparseOperation opA, dataTAX alpha, ConstSparseMatrix<indexT, dataTAX> matA,
            ConstDenseVector<dataTAX> vecX, dataTY beta, DenseVector<dataTY> vecY, cudaDataType computeType, SpMVAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAX : struct where dataTY : struct
        {
            SizeT size = 0;
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAX)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTY)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpMV_preprocess(_handle, opA, ptrAlpha, matA.Descr, vecX.Descr, ptrBeta, vecY.Descr, computeType, alg, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMV_preprocess", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
            return size;
        }
        public void MV<indexT, dataTAX, dataTY>(cusparseOperation opA, CudaDeviceVariable<dataTAX> alpha, ConstSparseMatrix<indexT, dataTAX> matA,
            ConstDenseVector<dataTAX> vecX, CudaDeviceVariable<dataTY> beta, DenseVector<dataTY> vecY, cudaDataType computeType, SpMVAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAX : struct where dataTY : struct
        {
            res = CudaSparseNativeMethods.cusparseSpMV(_handle, opA, alpha.DevicePointer, matA.Descr, vecX.Descr, beta.DevicePointer, vecY.Descr, computeType, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMV", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void MV<indexT, dataTAX, dataTY>(cusparseOperation opA, dataTAX alpha, ConstSparseMatrix<indexT, dataTAX> matA,
            ConstDenseVector<dataTAX> vecX, dataTY beta, DenseVector<dataTY> vecY, cudaDataType computeType, SpMVAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAX : struct where dataTY : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAX)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTY)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpMV(_handle, opA, ptrAlpha, matA.Descr, vecX.Descr, ptrBeta, vecY.Descr, computeType, alg, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMV", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
        }
        #endregion

        #region Matrix Matrix
        public SizeT MM_bufferSize<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataTAB> alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, CudaDeviceVariable<dataTC> beta, DenseMatrix<dataTC> matC, cudaDataType computeType, SpMMAlg alg)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSpMM_bufferSize(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMM_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public SizeT MM_bufferSize<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, dataTAB alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, dataTC beta, DenseMatrix<dataTC> matC, cudaDataType computeType, SpMMAlg alg)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            SizeT size = 0;
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAB)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTC)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpMM_bufferSize(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, ref size);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMM_bufferSize", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
            return size;
        }
        public void MMPreprocess<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataTAB> alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, CudaDeviceVariable<dataTC> beta, DenseMatrix<dataTC> matC, cudaDataType computeType, SpMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSpMM_preprocess(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMM_preprocess", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void MMPreprocess<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, dataTAB alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, dataTC beta, DenseMatrix<dataTC> matC, cudaDataType computeType, SpMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAB)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTC)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpMM_preprocess(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMM_preprocess", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
        }
        public void MM<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataTAB> alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, CudaDeviceVariable<dataTC> beta, DenseMatrix<dataTC> matC, cudaDataType computeType, SpMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSpMM(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMM", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void MM<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, dataTAB alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, dataTC beta, DenseMatrix<dataTC> matC, cudaDataType computeType, SpMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAB)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTC)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpMM(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMM", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
        }
        #endregion

        #region SpSV

        public SizeT SpSV_bufferSize<indexT, dataT>(cusparseOperation opA, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseVector<dataT> vecX, DenseVector<dataT> vecY, cudaDataType computeType, cusparseSpSVAlg alg, SpSVDescr spsvDescr)
            where indexT : struct where dataT : struct
        {
            SizeT size = 0;

            res = CudaSparseNativeMethods.cusparseSpSV_bufferSize(_handle, opA, alpha.DevicePointer, matA.Descr, vecX.Descr, vecY.Descr, computeType, alg, spsvDescr.Descr, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSV_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public void SpSV_analysis<indexT, dataT>(cusparseOperation opA, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseVector<dataT> vecX, DenseVector<dataT> vecY, cudaDataType computeType, cusparseSpSVAlg alg, SpSVDescr spsvDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSpSV_analysis(_handle, opA, alpha.DevicePointer, matA.Descr, vecX.Descr, vecY.Descr, computeType, alg, spsvDescr.Descr, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSV_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void SpSV_solve<indexT, dataT>(cusparseOperation opA, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseVector<dataT> vecX, DenseVector<dataT> vecY, cudaDataType computeType, cusparseSpSVAlg alg, SpSVDescr spsvDescr)
            where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSpSV_solve(_handle, opA, alpha.DevicePointer, matA.Descr, vecX.Descr, vecY.Descr, computeType, alg, spsvDescr.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSV_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void SpSV_updateMatrix<dataT>(cusparseContext handle,
                          SpSVDescr spsvDescr,
                          CudaDeviceVariable<dataT> newValues,
                          cusparseSpSVUpdate update)
            where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSpSV_updateMatrix(_handle, spsvDescr.Descr, newValues.DevicePointer, update);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSV_updateMatrix", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        public SizeT SpSV_bufferSize<indexT, dataT>(cusparseOperation opA, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseVector<dataT> vecX, DenseVector<dataT> vecY, cudaDataType computeType, cusparseSpSVAlg alg, SpSVDescr spsvDescr)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            SizeT size = 0;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                res = CudaSparseNativeMethods.cusparseSpSV_bufferSize(_handle, opA, ptrAlpha, matA.Descr, vecX.Descr, vecY.Descr, computeType, alg, spsvDescr.Descr, ref size);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSV_bufferSize", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
            }
            return size;
        }

        public void SpSV_analysis<indexT, dataT>(cusparseOperation opA, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseVector<dataT> vecX, DenseVector<dataT> vecY, cudaDataType computeType, cusparseSpSVAlg alg, SpSVDescr spsvDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                res = CudaSparseNativeMethods.cusparseSpSV_analysis(_handle, opA, ptrAlpha, matA.Descr, vecX.Descr, vecY.Descr, computeType, alg, spsvDescr.Descr, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSV_analysis", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
            }
        }

        public void SpSV_solve<indexT, dataT>(cusparseOperation opA, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseVector<dataT> vecX, DenseVector<dataT> vecY, cudaDataType computeType, cusparseSpSVAlg alg, SpSVDescr spsvDescr)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                res = CudaSparseNativeMethods.cusparseSpSV_solve(_handle, opA, ptrAlpha, matA.Descr, vecX.Descr, vecY.Descr, computeType, alg, spsvDescr.Descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSV_solve", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
            }
        }

        #endregion

        #region SpSM

        public SizeT SpSM_bufferSize<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseMatrix<dataT> matB, DenseMatrix<dataT> matC, cudaDataType computeType, cusparseSpSMAlg alg, SpSMDescr SpSMDescr)
            where indexT : struct where dataT : struct
        {
            SizeT size = 0;

            res = CudaSparseNativeMethods.cusparseSpSM_bufferSize(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, matC.Descr, computeType, alg, SpSMDescr.Descr, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSM_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public void SpSM_analysis<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseMatrix<dataT> matB, DenseMatrix<dataT> matC, cudaDataType computeType, cusparseSpSMAlg alg, SpSMDescr SpSMDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSpSM_analysis(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, matC.Descr, computeType, alg, SpSMDescr.Descr, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSM_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void SpSM_solve<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseMatrix<dataT> matB, DenseMatrix<dataT> matC, cudaDataType computeType, cusparseSpSMAlg alg, SpSMDescr SpSMDescr)
            where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSpSM_solve(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, matC.Descr, computeType, alg, SpSMDescr.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSM_solve", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        public SizeT SpSM_bufferSize<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseMatrix<dataT> matB, DenseMatrix<dataT> matC, cudaDataType computeType, cusparseSpSMAlg alg, SpSMDescr SpSMDescr)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            SizeT size = 0;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                res = CudaSparseNativeMethods.cusparseSpSM_bufferSize(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, matC.Descr, computeType, alg, SpSMDescr.Descr, ref size);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSM_bufferSize", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
            }
            return size;
        }

        public void SpSM_analysis<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseMatrix<dataT> matB, DenseMatrix<dataT> matC, cudaDataType computeType, cusparseSpSMAlg alg, SpSMDescr SpSMDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                res = CudaSparseNativeMethods.cusparseSpSM_analysis(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, matC.Descr, computeType, alg, SpSMDescr.Descr, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSM_analysis", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
            }
        }

        public void SpSM_solve<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstDenseMatrix<dataT> matB, DenseMatrix<dataT> matC, cudaDataType computeType, cusparseSpSMAlg alg, SpSMDescr SpSMDescr)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                res = CudaSparseNativeMethods.cusparseSpSM_solve(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, matC.Descr, computeType, alg, SpSMDescr.Descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSM_solve", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
            }
        }

        public void SpSM_updateMatrix<dataT>(SpSMDescr spsmDescr, CudaDeviceVariable<dataT> newValues,
            cusparseSpSMUpdate updatePart)
            where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSpSM_updateMatrix(_handle, spsmDescr.Descr, newValues.DevicePointer, updatePart);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpSM_updateMatrix", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion

        #region SpGeMM

        public SizeT SpGEMM_workEstimation<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstSparseMatrix<indexT, dataT> matB, CudaDeviceVariable<dataT> beta, SparseMatrix<indexT, dataT> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            SizeT size = 0;
            CUdeviceptr bufferPtr = new CUdeviceptr();
            if (buffer != null)
            {
                size = buffer.SizeInBytes;
                bufferPtr = buffer.DevicePointer;
            }


            res = CudaSparseNativeMethods.cusparseSpGEMM_workEstimation(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, spgemmDescr.Descr, ref size, bufferPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMM_workEstimation", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public SizeT SpGEMM_compute<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstSparseMatrix<indexT, dataT> matB, CudaDeviceVariable<dataT> beta, SparseMatrix<indexT, dataT> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            SizeT size = 0;
            CUdeviceptr bufferPtr = new CUdeviceptr();
            if (buffer != null)
            {
                size = buffer.SizeInBytes;
                bufferPtr = buffer.DevicePointer;
            }


            res = CudaSparseNativeMethods.cusparseSpGEMM_compute(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, spgemmDescr.Descr, ref size, bufferPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMM_compute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public void SpGEMM_copy<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataT> alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstSparseMatrix<indexT, dataT> matB, CudaDeviceVariable<dataT> beta, SparseMatrix<indexT, dataT> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr)
            where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSpGEMM_copy(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, spgemmDescr.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMM_copy", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        public SizeT SpGEMM_workEstimation<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstSparseMatrix<indexT, dataT> matB, dataT beta, SparseMatrix<indexT, dataT> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            SizeT size = 0;
            CUdeviceptr bufferPtr = new CUdeviceptr();
            if (buffer != null)
            {
                size = buffer.SizeInBytes;
                bufferPtr = buffer.DevicePointer;
            }

            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpGEMM_workEstimation(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, spgemmDescr.Descr, ref size, bufferPtr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMM_workEstimation", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
            return size;
        }

        public SizeT SpGEMM_compute<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstSparseMatrix<indexT, dataT> matB, dataT beta, SparseMatrix<indexT, dataT> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            SizeT size = 0;
            CUdeviceptr bufferPtr = new CUdeviceptr();
            if (buffer != null)
            {
                size = buffer.SizeInBytes;
                bufferPtr = buffer.DevicePointer;
            }

            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpGEMM_compute(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, spgemmDescr.Descr, ref size, bufferPtr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMM_compute", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
            return size;
        }

        public void SpGEMM_copy<indexT, dataT>(cusparseOperation opA, cusparseOperation opB, dataT alpha, ConstSparseMatrix<indexT, dataT> matA,
            ConstSparseMatrix<indexT, dataT> matB, dataT beta, SparseMatrix<indexT, dataT> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr)
            where indexT : struct where dataT : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataT)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpGEMM_copy(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, spgemmDescr.Descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMM_copy", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
        }

        #endregion

        #region GeMM


        public SizeT SDDMM_bufferSize<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataTAB> alpha, ConstDenseMatrix<dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, CudaDeviceVariable<dataTC> beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSDDMMAlg alg)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            SizeT size = 0;
            res = CudaSparseNativeMethods.cusparseSDDMM_bufferSize(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSDDMM_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return size;
        }

        public SizeT SDDMM_bufferSize<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, dataTAB alpha, ConstDenseMatrix<dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, dataTC beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSDDMMAlg alg)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            SizeT size = 0;
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAB)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTC)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSDDMM_bufferSize(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, ref size);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSDDMM_bufferSize", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
            return size;
        }
        public void SDDMMPreprocess<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataTAB> alpha, ConstDenseMatrix<dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, CudaDeviceVariable<dataTC> beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSDDMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSDDMM_preprocess(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSDDMM_preprocess", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void SDDMMPreprocess<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, dataTAB alpha, ConstDenseMatrix<dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, dataTC beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSDDMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAB)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTC)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSDDMM_preprocess(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSDDMM_preprocess", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
        }
        public void SDDMM<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataTAB> alpha, ConstDenseMatrix<dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, CudaDeviceVariable<dataTC> beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSDDMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSDDMM(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSDDMM", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void SDDMM<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, dataTAB alpha, ConstDenseMatrix<dataTAB> matA,
            ConstDenseMatrix<dataTAB> matB, dataTC beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSDDMMAlg alg, CudaDeviceVariable<byte> buffer)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAB)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTC)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSDDMM(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, buffer.DevicePointer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSDDMM", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
        }

        #endregion

        #region GeMMReuse



        public void GEMMreuse_workEstimation<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstSparseMatrix<indexT, dataTAB> matB, SparseMatrix<indexT, dataTC> matC, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr, ref SizeT bufferSize1, CudaDeviceVariable<byte> buffer1)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSpGEMMreuse_workEstimation(_handle, opA, opB, matA.Descr, matB.Descr, matC.Descr, alg, spgemmDescr.Descr, ref bufferSize1, buffer1.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMMreuse_workEstimation", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void GEMMreuse_nnz<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstSparseMatrix<indexT, dataTAB> matB, SparseMatrix<indexT, dataTC> matC, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr,
            ref SizeT bufferSize2, CudaDeviceVariable<byte> buffer2, ref SizeT bufferSize3, CudaDeviceVariable<byte> buffer3, ref SizeT bufferSize4, CudaDeviceVariable<byte> buffer4)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSpGEMMreuse_nnz(_handle, opA, opB, matA.Descr, matB.Descr, matC.Descr, alg, spgemmDescr.Descr,
                ref bufferSize2, buffer2.DevicePointer, ref bufferSize3, buffer3.DevicePointer, ref bufferSize4, buffer4.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMMreuse_nnz", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        public void GEMMreuse_copy<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstSparseMatrix<indexT, dataTAB> matB, SparseMatrix<indexT, dataTC> matC, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr, ref SizeT bufferSize5, CudaDeviceVariable<byte> buffer5)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSpGEMMreuse_copy(_handle, opA, opB, matA.Descr, matB.Descr, matC.Descr, alg, spgemmDescr.Descr, ref bufferSize5, buffer5.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMMreuse_copy", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void GEMMreuse_compute<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, dataTAB alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstSparseMatrix<indexT, dataTAB> matB, dataTC beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            IntPtr ptrAlpha = IntPtr.Zero;
            IntPtr ptrBeta = IntPtr.Zero;
            try
            {
                ptrAlpha = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTAB)));
                ptrBeta = Marshal.AllocHGlobal(CudaDataTypeTranslator.GetSize(typeof(dataTC)));
                Marshal.StructureToPtr(alpha, ptrAlpha, false);
                Marshal.StructureToPtr(beta, ptrBeta, false);
                res = CudaSparseNativeMethods.cusparseSpGEMMreuse_compute(_handle, opA, opB, ptrAlpha, matA.Descr, matB.Descr, ptrBeta, matC.Descr, computeType, alg, spgemmDescr.Descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMMreuse_compute", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
            }
            finally
            {
                //result = (dataTCompute)Marshal.PtrToStructure(ptr, typeof(dataTCompute));
                Marshal.FreeHGlobal(ptrAlpha);
                Marshal.FreeHGlobal(ptrBeta);
            }
        }

        public void GEMMreuse_compute<indexT, dataTAB, dataTC>(cusparseOperation opA, cusparseOperation opB, CudaDeviceVariable<dataTAB> alpha, ConstSparseMatrix<indexT, dataTAB> matA,
            ConstSparseMatrix<indexT, dataTAB> matB, CudaDeviceVariable<dataTC> beta, SparseMatrix<indexT, dataTC> matC, cudaDataType computeType, cusparseSpGEMMAlg alg, SpGEMMDescr spgemmDescr)
            where indexT : struct where dataTAB : struct where dataTC : struct
        {
            res = CudaSparseNativeMethods.cusparseSpGEMMreuse_compute(_handle, opA, opB, alpha.DevicePointer, matA.Descr, matB.Descr, beta.DevicePointer, matC.Descr, computeType, alg, spgemmDescr.Descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpGEMMreuse_compute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        #endregion

        #region Sparse/Dense conversion

        /// <summary>
        /// SparseToDenseBufferSize
        /// </summary>
        /// <param name="pointerMode"></param>
        public SizeT SparseToDenseBufferSize<indexT, dataT>(SparseMatrix<indexT, dataT> matA, DenseMatrix<dataT> matB, cusparseSparseToDenseAlg alg) where indexT : struct where dataT : struct
        {
            SizeT bufferSize = new SizeT();
            res = CudaSparseNativeMethods.cusparseSparseToDense_bufferSize(_handle, matA.Descr, matB.Descr, alg, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSparseToDense_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return bufferSize;
        }

        /// <summary>
        /// SparseToDense
        /// </summary>
        /// <param name="pointerMode"></param>
        public void SparseToDense<indexT, dataT>(SparseMatrix<indexT, dataT> matA, DenseMatrix<dataT> matB, cusparseSparseToDenseAlg alg, CudaDeviceVariable<byte> buffer) where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseSparseToDense(_handle, matA.Descr, matB.Descr, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSparseToDense", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// DenseToSparseBufferSize
        /// </summary>
        /// <param name="pointerMode"></param>
        public SizeT DenseToSparseBufferSize<indexT, dataT>(ConstDenseMatrix<dataT> matA, SparseMatrix<indexT, dataT> matB, cusparseDenseToSparseAlg alg) where indexT : struct where dataT : struct
        {
            SizeT bufferSize = new SizeT();
            res = CudaSparseNativeMethods.cusparseDenseToSparse_bufferSize(_handle, matA.Descr, matB.Descr, alg, ref bufferSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDenseToSparse_bufferSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return bufferSize;
        }

        /// <summary>
        /// DenseToSparse
        /// </summary>
        /// <param name="pointerMode"></param>
        public void DenseToSparseAnalysis<indexT, dataT>(ConstDenseMatrix<dataT> matA, SparseMatrix<indexT, dataT> matB, cusparseDenseToSparseAlg alg, CudaDeviceVariable<byte> buffer) where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseDenseToSparse_analysis(_handle, matA.Descr, matB.Descr, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDenseToSparse_analysis", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// DenseToSparse
        /// </summary>
        /// <param name="pointerMode"></param>
        public void DenseToSparseConvert<indexT, dataT>(ConstDenseMatrix<dataT> matA, SparseMatrix<indexT, dataT> matB, cusparseDenseToSparseAlg alg, CudaDeviceVariable<byte> buffer) where indexT : struct where dataT : struct
        {
            res = CudaSparseNativeMethods.cusparseDenseToSparse_convert(_handle, matA.Descr, matB.Descr, alg, buffer.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDenseToSparse_convert", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        #endregion

        #endregion

        public SpMMOpPlan CreateSpMMOpPlan<IndexTypeA, DataTypeA, DataTypeB, DataTypeC>(cusparseOperation opA,
                          cusparseOperation opB,
                          ConstSparseMatrix<IndexTypeA, DataTypeA> matA,
                          ConstDenseMatrix<DataTypeB> matB,
                          DenseMatrix<DataTypeC> matC,
                          cudaDataType computeType,
                          cusparseSpMMOpAlg alg,
                          byte[] addOperationNvvmBuffer,
                          byte[] mulOperationNvvmBuffer,
                          byte[] epilogueNvvmBuffer) where IndexTypeA : struct where DataTypeA : struct where DataTypeB : struct where DataTypeC : struct
        {
            return new SpMMOpPlan(_handle, opA, opB, matA.Descr, matB.Descr, matC.Descr, computeType, alg, addOperationNvvmBuffer, mulOperationNvvmBuffer, epilogueNvvmBuffer);
        }

        /// <summary>
        /// Returns the wrapped cusparseContext handle
        /// </summary>
        public cusparseContext Handle
        {
            get { return _handle; }
        }
    }
}
