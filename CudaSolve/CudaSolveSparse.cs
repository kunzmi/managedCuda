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
using ManagedCuda.CudaSparse;

namespace ManagedCuda.CudaSolve
{
	/// <summary>
	/// CudaSolvSparse: The cuSolverSP library was mainly designed to a solve sparse linear system AxB and the least-squares problem
	/// x = argmin||A*z-b||
	/// </summary>
	public class CudaSolveSparse : IDisposable
	{
		bool disposed;
		cusolverStatus res;
		cusolverSpHandle _handle;

		#region Constructor
		/// <summary>
		/// Create new sparse solve instance
		/// </summary>
		public CudaSolveSparse()
		{
			_handle = new cusolverSpHandle();
			res = CudaSolveNativeMethods.Sparse.cusolverSpCreate(ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDestroy", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// Create new sparse solve instance using stream stream
		/// </summary>
		/// <param name="stream"></param>
		public CudaSolveSparse(CudaStream stream)
			: this()
		{
			SetStream(stream);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaSolveSparse()
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
				res = CudaSolveNativeMethods.Sparse.cusolverSpDestroy(_handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Stream

		/// <summary>
		/// This function sets the stream to be used by the cuSolverSP library to execute its routines.
		/// </summary>
		/// <param name="stream">the stream to be used by the library.</param>
		public void SetStream(CudaStream stream)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpSetStream(_handle, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpSetStream", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function gets the stream to be used by the cuSolverSP library to execute its routines.
		/// </summary>
		public CudaStream GetStream()
		{
			CUstream stream = new CUstream();
			res = CudaSolveNativeMethods.Sparse.cusolverSpGetStream(_handle, ref stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpGetStream", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);

			return new CudaStream(stream);
		}
		#endregion


		/// <summary>
		/// This function checks if A has symmetric pattern or not. The output parameter issym
		/// reports 1 if A is symmetric; otherwise, it reports 0.
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix
		/// type is CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrRowPtrA">integer array of m elements that contains the start of every row.</param>
		/// <param name="csrEndPtrA">integer array of m elements that contains the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of matrix A.</param>
		/// <returns>1 if A is symmetric; 0 otherwise.</returns>
		public int CsrissymHost(int m, int nnzA, CudaSparseMatrixDescriptor descrA, int[] csrRowPtrA, int[] csrEndPtrA, int[] csrColIndA)
		{
			int issym = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpXcsrissymHost(_handle, m, nnzA, descrA.Descriptor, csrRowPtrA, csrEndPtrA, csrColIndA, ref issym);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpXcsrissymHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return issym;
		}

		#region linear solver based on LU factorization
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvluHost(int n, int nnzA, CudaSparseMatrixDescriptor descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float[] b, float tol, int reorder, float[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrlsvluHost(_handle, n, nnzA, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrlsvluHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvluHost(int n, int nnzA, CudaSparseMatrixDescriptor descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double[] b, double tol, int reorder, double[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrlsvluHost(_handle, n, nnzA, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrlsvluHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvluHost(int n, int nnzA, CudaSparseMatrixDescriptor descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex[] b, float tol, int reorder, cuFloatComplex[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrlsvluHost(_handle, n, nnzA, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrlsvluHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvluHost(int n, int nnzA, CudaSparseMatrixDescriptor descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex[] b, double tol, int reorder, cuDoubleComplex[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrlsvluHost(_handle, n, nnzA, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrlsvluHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		#endregion

		#region linear solver based on QR factorization
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvqr(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<float> b, float tol, int reorder, CudaDeviceVariable<float> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrlsvqr(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrlsvqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvqr(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<double> b, float tol, int reorder, CudaDeviceVariable<double> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrlsvqr(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrlsvqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvqr(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuFloatComplex> b, float tol, int reorder, CudaDeviceVariable<cuFloatComplex> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrlsvqr(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrlsvqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvqr(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuDoubleComplex> b, float tol, int reorder, CudaDeviceVariable<cuDoubleComplex> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrlsvqr(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrlsvqr", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}

		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvqrHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float[] b, float tol, int reorder, float[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrlsvqrHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrlsvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvqrHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double[] b, float tol, int reorder, double[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrlsvqrHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrlsvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvqrHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex[] b, float tol, int reorder, cuFloatComplex[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrlsvqrHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrlsvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
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
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvqrHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex[] b, float tol, int reorder, cuDoubleComplex[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrlsvqrHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrlsvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		#endregion

		#region linear solver based on Cholesky factorization
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvcholHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float[] b, float tol, int reorder, float[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrlsvcholHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrlsvcholHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvcholHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double[] b, float tol, int reorder, double[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrlsvcholHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrlsvcholHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvcholHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex[] b, float tol, int reorder, cuFloatComplex[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrlsvcholHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrlsvcholHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int CsrlsvcholHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex[] b, float tol, int reorder, cuDoubleComplex[] x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrlsvcholHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrlsvcholHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}

		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvchol(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<float> b, float tol, int reorder, CudaDeviceVariable<float> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrlsvchol(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrlsvchol", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvchol(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<double> b, float tol, int reorder, CudaDeviceVariable<double> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrlsvchol(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrlsvchol", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvchol(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuFloatComplex> b, float tol, int reorder, CudaDeviceVariable<cuFloatComplex> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrlsvchol(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrlsvchol", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}
		/// <summary>
		/// This function solves the linear system A*x=b
		/// </summary>
		/// <param name="m">number of rows and columns of matrix A.</param>
		/// <param name="nnz">number of nonzeros of matrix A.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrValA">array of nnz (= csrRowPtrA(n) * csrRowPtrA(0)) nonzero elements of matrix A.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="b">right hand side vector of size m.</param>
		/// <param name="tol">tolerance to decide singularity.</param>
		/// <param name="reorder">no effect.</param>
		/// <param name="x">solution vector of size m, x = inv(A)*b.</param>
		/// <returns>-1 if A is invertible. Otherwise, first index j such that U(j,j)≈0</returns>
		public int Csrlsvchol(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuDoubleComplex> b, float tol, int reorder, CudaDeviceVariable<cuDoubleComplex> x)
		{
			int singularity = 0;
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrlsvchol(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, tol, reorder, x.DevicePointer, ref singularity);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrlsvchol", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return singularity;
		}

		#endregion

		#region least square solver based on QR factorization
		/// <summary>
		/// This function solves the following least-square problem x = argmin||A*z-b||
		/// </summary>
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
		public void CsrlsqvqrHost(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float[] b, float tol, ref int rankA, float[] x, int[] p, ref float min_norm)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrlsqvqrHost(_handle, m, n, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, ref rankA, x, p, ref min_norm);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrlsqvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the following least-square problem x = argmin||A*z-b||
		/// </summary>
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
		public void CsrlsqvqrHost(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double[] b, double tol, ref int rankA, double[] x, int[] p, ref double min_norm)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrlsqvqrHost(_handle, m, n, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, ref rankA, x, p, ref min_norm);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrlsqvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the following least-square problem x = argmin||A*z-b||
		/// </summary>
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
		public void CsrlsqvqrHost(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex[] b, float tol, ref int rankA, cuFloatComplex[] x, int[] p, ref float min_norm)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrlsqvqrHost(_handle, m, n, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, ref rankA, x, p, ref min_norm);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrlsqvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the following least-square problem x = argmin||A*z-b||
		/// </summary>
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
		public void CsrlsqvqrHost(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex[] b, double tol, ref int rankA, cuDoubleComplex[] x, int[] p, ref double min_norm)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrlsqvqrHost(_handle, m, n, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, b, tol, ref rankA, x, p, ref min_norm);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrlsqvqrHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		#endregion

		#region eigenvalue solver based on shift inverse
		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void CsreigvsiHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, float mu0, float[] x0, int maxite, float tol, ref float mu, float[] x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsreigvsiHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, ref mu, x);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsreigvsiHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void CsreigvsiHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, double mu0, double[] x0, int maxite, double tol, ref double mu, double[] x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsreigvsiHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, ref mu, x);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsreigvsiHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void CsreigvsiHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex mu0, cuFloatComplex[] x0, int maxite, float tol, ref cuFloatComplex mu, cuFloatComplex[] x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsreigvsiHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, ref mu, x);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsreigvsiHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void CsreigvsiHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex mu0, cuDoubleComplex[] x0, int maxite, double tol, ref cuDoubleComplex mu, cuDoubleComplex[] x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsreigvsiHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, ref mu, x);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsreigvsiHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void Csreigvsi(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, float mu0, CudaDeviceVariable<float> x0, int maxite, float tol, ref float mu, CudaDeviceVariable<float> x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsreigvsi(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, mu0, x0.DevicePointer, maxite, tol, ref mu, x.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsreigvsi", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void Csreigvsi(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, double mu0, CudaDeviceVariable<double> x0, int maxite, double tol, ref double mu, CudaDeviceVariable<double> x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsreigvsi(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, mu0, x0.DevicePointer, maxite, tol, ref mu, x.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsreigvsi", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void Csreigvsi(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, cuFloatComplex mu0, CudaDeviceVariable<cuFloatComplex> x0, int maxite, float tol, ref cuFloatComplex mu, CudaDeviceVariable<cuFloatComplex> x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsreigvsi(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, mu0, x0.DevicePointer, maxite, tol, ref mu, x.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsreigvsi", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This function solves the simple eigenvalue problem A*x=lambda*x by shift-inverse method.
		/// </summary>
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
		public void Csreigvsi(int m, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, cuDoubleComplex mu0, CudaDeviceVariable<cuDoubleComplex> x0, int maxite, double tol, ref cuDoubleComplex mu, CudaDeviceVariable<cuDoubleComplex> x)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsreigvsi(_handle, m, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, mu0, x0.DevicePointer, maxite, tol, ref mu, x.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsreigvsi", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


		#endregion

		#region enclosed eigenvalues
		/// <summary/>
		public void CsreigsHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, float[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex left_bottom_corner, cuFloatComplex right_upper_corner, ref int num_eigs)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsreigsHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, ref num_eigs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsreigsHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary/>
		public void CsreigsHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, double[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex left_bottom_corner, cuDoubleComplex right_upper_corner, ref int num_eigs)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsreigsHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, ref num_eigs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsreigsHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary/>
		public void CsreigsHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuFloatComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuFloatComplex left_bottom_corner, cuFloatComplex right_upper_corner, ref int num_eigs)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsreigsHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, ref num_eigs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsreigsHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary/>
		public void CsreigsHost(int m, int nnz, CudaSparseMatrixDescriptor descrA, cuDoubleComplex[] csrValA, int[] csrRowPtrA, int[] csrColIndA, cuDoubleComplex left_bottom_corner, cuDoubleComplex right_upper_corner, ref int num_eigs)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsreigsHost(_handle, m, nnz, descrA.Descriptor, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, ref num_eigs);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsreigsHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		#endregion

		#region CPU symrcm
		/// <summary>
		/// This function implements Symmetric Reverse Cuthill-McKee permutation. It returns a
		/// permutation vector p such that A(p,p) would concentrate nonzeros to diagonal. This is
		/// equivalent to symrcm in MATLAB, however the result may not be the same because of
		/// different heuristics in the pseudoperipheral finder.
		/// </summary>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="p">permutation vector of size n.</param>
		public void CsrsymrcmHost(int n, int nnzA, CudaSparseMatrixDescriptor descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpXcsrsymrcmHost(_handle, n, nnzA, descrA.Descriptor, csrRowPtrA, csrColIndA, p);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpXcsrsymrcmHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		#endregion

		#region CPU symmdq
		/// <summary>
		/// Symmetric minimum degree algorithm based on quotient graph.<para/>
		/// This function implements Symmetric Minimum Degree Algorithm based on Quotient
		/// Graph. It returns a permutation vector p such that A(p,p) would have less zero fill-in
		/// during Cholesky factorization.
		/// </summary>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="p">permutation vector of size n.</param>
		public void CsrsymmdqHost(int n, int nnzA, CudaSparseMatrixDescriptor descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpXcsrsymmdqHost(_handle, n, nnzA, descrA.Descriptor, csrRowPtrA, csrColIndA, p);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpXcsrsymmdqHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// Symmetric Approximate minimum degree algorithm based on quotient graph.<para/>
		/// This function implements Symmetric Approximate Minimum Degree Algorithm based
		/// on Quotient Graph. It returns a permutation vector p such that A(p,p) would have less
		/// zero fill-in during Cholesky factorization.
		/// </summary>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="p">permutation vector of size n.</param>
		public void CsrsymamdHost(int n, int nnzA, CudaSparseMatrixDescriptor descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpXcsrsymamdHost(_handle, n, nnzA, descrA.Descriptor, csrRowPtrA, csrColIndA, p);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpXcsrsymamdHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		#endregion

		#region CPU permuation
		/// <summary>
		/// Given a left permutation vector p which corresponds to permutation matrix P and a
		/// right permutation vector q which corresponds to permutation matrix Q, this function
		/// computes permutation of matrix A by B = P*A*Q^T
		/// </summary>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="nnzA">number of nonzeros of matrix A. It is the size of csrValA and csrColIndA.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrRowPtrA">integer array of n + 1 elements that contains the start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnz (=csrRowPtrA(n) * csrRowPtrA(0)) column indices of the nonzero elements of matrix A.</param>
		/// <param name="p">left permutation vector of size m.</param>
		/// <param name="q">right permutation vector of size n.</param>
		/// <returns>number of bytes of the buffer.</returns>
		public SizeT Csrperm_bufferSizeHost(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p, int[] q)
		{
			SizeT bufferSizeInBytes = new SizeT();
			res = CudaSolveNativeMethods.Sparse.cusolverSpXcsrperm_bufferSizeHost(_handle, m, n, nnzA, descrA.Descriptor, csrRowPtrA, csrColIndA, p, q, ref bufferSizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpXcsrperm_bufferSizeHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
			return bufferSizeInBytes;
		}


		/// <summary>
		/// Given a left permutation vector p which corresponds to permutation matrix P and a
		/// right permutation vector q which corresponds to permutation matrix Q, this function
		/// computes permutation of matrix A by B = P*A*Q^T
		/// </summary>
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
		public void CsrpermHost(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, int[] csrRowPtrA, int[] csrColIndA, int[] p, int[] q, int[] map, byte[] pBuffer)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpXcsrpermHost(_handle, m, n, nnzA, descrA.Descriptor, csrRowPtrA, csrColIndA, p, q, map, pBuffer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpXcsrpermHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		#endregion

		#region Low-level API: Batched QR


		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
		/// <param name="m">number of rows of each matrix Aj.</param>
		/// <param name="n">number of columns of each matrix Aj.</param>
		/// <param name="nnzA">number of nonzeros of each matrix Aj. It is the size csrColIndA.</param>
		/// <param name="descrA">the descriptor of matrix A. The supported matrix type is
		/// CUSPARSE_MATRIX_TYPE_GENERAL. Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE.</param>
		/// <param name="csrRowPtrA">integer array of m+1 elements that contains the
		/// start of every row and the end of the last row plus one.</param>
		/// <param name="csrColIndA">integer array of nnzAcolumn indices of the nonzero elements of each matrix Aj.</param>
		/// <param name="info">opaque structure for QR factorization.</param>
		public void CsrqrAnalysisBatched(int m, int n, int nnzA, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CsrQrInfo info)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpXcsrqrAnalysisBatched(_handle, m, n, nnzA, descrA.Descriptor, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, info.Info);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpXcsrqrAnalysisBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrBufferInfoBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int batchSize, CsrQrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrqrBufferInfoBatched(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, batchSize, info.Info, ref internalDataInBytes, ref workspaceInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrqrBufferInfoBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrBufferInfoBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int batchSize, CsrQrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrqrBufferInfoBatched(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, batchSize, info.Info, ref internalDataInBytes, ref workspaceInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrqrBufferInfoBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrBufferInfoBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int batchSize, CsrQrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrqrBufferInfoBatched(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, batchSize, info.Info, ref internalDataInBytes, ref workspaceInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrqrBufferInfoBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrBufferInfoBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrVal, CudaDeviceVariable<int> csrRowPtr, CudaDeviceVariable<int> csrColInd, int batchSize, CsrQrInfo info, ref SizeT internalDataInBytes, ref SizeT workspaceInBytes)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrqrBufferInfoBatched(_handle, m, n, nnz, descrA.Descriptor, csrVal.DevicePointer, csrRowPtr.DevicePointer, csrColInd.DevicePointer, batchSize, info.Info, ref internalDataInBytes, ref workspaceInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrqrBufferInfoBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}




		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrsvBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<float> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<float> b, CudaDeviceVariable<float> x, int batchSize, CsrQrInfo info, CudaDeviceVariable<byte> pBuffer)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpScsrqrsvBatched(_handle, m, n, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, x.DevicePointer, batchSize, info.Info, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpScsrqrsvBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrsvBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<double> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<double> b, CudaDeviceVariable<double> x, int batchSize, CsrQrInfo info, CudaDeviceVariable<byte> pBuffer)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpDcsrqrsvBatched(_handle, m, n, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, x.DevicePointer, batchSize, info.Info, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpDcsrqrsvBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrsvBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuFloatComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuFloatComplex> b, CudaDeviceVariable<cuFloatComplex> x, int batchSize, CsrQrInfo info, CudaDeviceVariable<byte> pBuffer)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpCcsrqrsvBatched(_handle, m, n, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, x.DevicePointer, batchSize, info.Info, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpCcsrqrsvBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The batched sparse QR factorization is used to solve either a set of least-squares
		/// problems or a set of linear systems
		/// </summary>
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
		public void CsrqrsvBatched(int m, int n, int nnz, CudaSparseMatrixDescriptor descrA, CudaDeviceVariable<cuDoubleComplex> csrValA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<cuDoubleComplex> b, CudaDeviceVariable<cuDoubleComplex> x, int batchSize, CsrQrInfo info, CudaDeviceVariable<byte> pBuffer)
		{
			res = CudaSolveNativeMethods.Sparse.cusolverSpZcsrqrsvBatched(_handle, m, n, nnz, descrA.Descriptor, csrValA.DevicePointer, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, b.DevicePointer, x.DevicePointer, batchSize, info.Info, pBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverSpZcsrqrsvBatched", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


		#endregion
	}
}
