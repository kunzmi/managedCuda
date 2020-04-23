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

namespace ManagedCuda.CudaSolve
{
	/// <summary>
	/// CudaSolveRefactorization: The cuSolverRF library was designed to accelerate solution of sets of linear systems by
	/// fast re-factorization when given new coefficients in the same sparsity pattern
	/// A_i x_i = f_i
	/// </summary>
	public class CudaSolveRefactorization : IDisposable
	{
		bool disposed;
		cusolverStatus res;
		cusolverRfHandle _handle;

		#region Constructor
		/// <summary>
		/// Create new refactorization solve instance
		/// </summary>
		public CudaSolveRefactorization()
		{
			_handle = new cusolverRfHandle();
			res = CudaSolveNativeMethods.Refactorization.cusolverRfCreate(ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfCreate", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaSolveRefactorization()
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
				res = CudaSolveNativeMethods.Refactorization.cusolverRfDestroy(_handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Init

		/// <summary>
		/// This routine gets the matrix format used in the cusolverRfSetup(),
		/// cusolverRfSetupHost(), cusolverRfResetValues(), cusolverRfExtractBundledFactorsHost() and cusolverRfExtractSplitFactorsHost() routines.
		/// </summary>
		/// <param name="format">the enumerated matrix format type.</param>
		/// <param name="diag">the enumerated unit diagonal type.</param>
		public void GetMatrixFormat(ref MatrixFormat format, ref UnitDiagonal diag)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfGetMatrixFormat(_handle, ref format, ref diag);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfGetMatrixFormat", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This routine sets the matrix format used in the cusolverRfSetup(),
		/// cusolverRfSetupHost(), cusolverRfResetValues(), cusolverRfExtractBundledFactorsHost() and cusolverRfExtractSplitFactorsHost() routines.
		/// </summary>
		/// <param name="format">the enumerated matrix format type.</param>
		/// <param name="diag">the enumerated unit diagonal type.</param>
		public void SetMatrixFormat(MatrixFormat format, UnitDiagonal diag)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfSetMatrixFormat(_handle, format, diag);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfSetMatrixFormat", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


		/// <summary>
		/// This routine sets the numeric values used for checking for "zero" pivot and for boosting
		/// it in the cusolverRfRefactor() and cusolverRfSolve() routines. It may be called 
		/// multiple times prior to cusolverRfRefactor() and cusolverRfSolve() routines.
		/// The numeric boosting will be used only if boost &gt; 0.0.
		/// </summary>
		/// <param name="zero">the value below which zero pivot is flagged.</param>
		/// <param name="boost">the value which is substituted for zero pivot (if the later is flagged).</param>
		public void SetNumericProperties(double zero, double boost)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfSetNumericProperties(_handle, zero, boost);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfSetNumericProperties", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This routine gets the numeric values used for checking for "zero" pivot and for boosting
		/// it in the cusolverRfRefactor() and cusolverRfSolve() routines. It may be called 
		/// multiple times prior to cusolverRfRefactor() and cusolverRfSolve() routines.
		/// The numeric boosting will be used only if boost &gt; 0.0.
		/// </summary>
		/// <param name="zero">the value below which zero pivot is flagged.</param>
		/// <param name="boost">the value which is substituted for zero pivot (if the later is flagged).</param>
		public void GetNumericProperties(ref double zero, ref double boost)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfGetNumericProperties(_handle, ref zero, ref boost);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfGetNumericProperties", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


		/// <summary>
		/// This routine gets the report whether numeric boosting was used in the
		/// cusolverRfRefactor() and cusolverRfSolve() routines.
		/// </summary>
		/// <param name="report">the enumerated boosting report type.</param>
		public void GetNumericBoostReport(ref NumericBoostReport report)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfGetNumericBoostReport(_handle, ref report);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfGetNumericBoostReport", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This routine sets the algorithm used for the refactorization in cusolverRfRefactor()
		/// and the triangular solve in cusolverRfSolve(). It may be called once prior to
		/// cusolverRfAnalyze() routine.
		/// </summary>
		/// <param name="factAlg">the enumerated algorithm type.</param>
		/// <param name="solveAlg">the enumerated algorithm type.</param>
		public void SetAlgs(Factorization factAlg, TriangularSolve solveAlg)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfSetAlgs(_handle, factAlg, solveAlg);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfSetAlgs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This routine gets the algorithm used for the refactorization in cusolverRfRefactor()
		/// and the triangular solve in cusolverRfSolve(). It may be called once prior to
		/// cusolverRfAnalyze() routine.
		/// </summary>
		/// <param name="factAlg">the enumerated algorithm type.</param>
		/// <param name="solveAlg">the enumerated algorithm type.</param>
		public void GetAlgs(ref Factorization factAlg, ref TriangularSolve solveAlg)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfGetAlgs(_handle, ref factAlg, ref solveAlg);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfGetAlgs", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}



		/// <summary>
		/// This routine gets the mode used in the cusolverRfResetValues routine.
		/// </summary>
		/// <param name="fastMode">the enumerated mode type.</param>
		public void GetResetValuesFastMode(ref ResetValuesFastMode fastMode)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfGetResetValuesFastMode(_handle, ref fastMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfGetResetValuesFastMode", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		/// <summary>
		/// This routine sets the mode used in the cusolverRfResetValues routine.
		/// </summary>
		/// <param name="fastMode">the enumerated mode type.</param>
		public void SetResetValuesFastMode(ResetValuesFastMode fastMode)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfSetResetValuesFastMode(_handle, fastMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfSetResetValuesFastMode", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

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
		public void SetupHost(int n, int nnzA, int[] h_csrRowPtrA, int[] h_csrColIndA, double[] h_csrValA, int nnzL, int[] h_csrRowPtrL, int[] h_csrColIndL,
			double[] h_csrValL, int nnzU, int[] h_csrRowPtrU, int[] h_csrColIndU, double[] h_csrValU, int[] h_P, int[] h_Q)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfSetupHost(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL,
				nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfSetupHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

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
		public void cusolverRfSetupDevice(int n, int nnzA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<double> csrValA, int nnzL,
			CudaDeviceVariable<int> csrRowPtrL, CudaDeviceVariable<int> csrColIndL, CudaDeviceVariable<double> csrValL, int nnzU, CudaDeviceVariable<int> csrRowPtrU,
			CudaDeviceVariable<int> csrColIndU, CudaDeviceVariable<double> csrValU, CudaDeviceVariable<int> P, CudaDeviceVariable<int> Q)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfSetupDevice(n, nnzA, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, csrValA.DevicePointer, nnzL,
				csrRowPtrL.DevicePointer, csrColIndL.DevicePointer, csrValL.DevicePointer, nnzU,
				csrRowPtrU.DevicePointer, csrColIndU.DevicePointer, csrValU.DevicePointer, P.DevicePointer, Q.DevicePointer, _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfSetupDevice", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

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
		public void ResetValues(int n, int nnzA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<double> csrValA,
			CudaDeviceVariable<double> P, CudaDeviceVariable<double> Q)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfResetValues(n, nnzA, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, 
				csrValA.DevicePointer, P.DevicePointer, Q.DevicePointer, _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfResetValues", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


		/// <summary>
		/// This routine performs the appropriate analysis of parallelism available in the LU refactorization depending upon the algorithm chosen by the user.
		/// </summary>
		public void Analyze()
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfAnalyze(_handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfAnalyze", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This routine performs the LU re-factorization
		/// </summary>
		public void Refactor(cusolverRfHandle handle)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfRefactor(_handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfRefactor", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


		/// <summary>
		/// This routine allows direct access to the lower L and upper U triangular factors stored in
		/// the cuSolverRF library handle. The factors are compressed into a single matrix M=(LI)+
		/// U, where the unitary diagonal of L is not stored. It is assumed that a prior call to the
		/// cusolverRfRefactor() was done in order to generate these triangular factors.
		/// </summary>
		/// <param name="nnzM">the number of non-zero elements of matrix M.</param>
		/// <param name="Mp">the array of offsets corresponding to the start of each row in the arrays Mi and Mx.
		/// This array has also an extra entry at the end that stores the number of non-zero elements in the matrix $M$. The array size is n+1.</param>
		/// <param name="Mi">the array of column indices corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
		/// <param name="Mx">the array of values corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
		public void AccessBundledFactorsDevice(out int nnzM, out CudaDeviceVariable<int> Mp, out CudaDeviceVariable<int> Mi, out CudaDeviceVariable<double> Mx)
		{
			CUdeviceptr d_mp = new CUdeviceptr();
			CUdeviceptr d_mi = new CUdeviceptr();
			CUdeviceptr d_mx = new CUdeviceptr();
			nnzM = 0;

			res = CudaSolveNativeMethods.Refactorization.cusolverRfAccessBundledFactorsDevice(_handle, ref nnzM, ref d_mp, ref d_mi, ref d_mx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfAccessBundledFactorsDevice", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);

			Mp = new CudaDeviceVariable<int>(d_mp);
			Mi = new CudaDeviceVariable<int>(d_mi);
			Mx = new CudaDeviceVariable<double>(d_mx);
		}

		/// <summary>
		/// This routine allows direct access to the lower L and upper U triangular factors stored in
		/// the cuSolverRF library handle. The factors are compressed into a single matrix M=(LI)+
		/// U, where the unitary diagonal of L is not stored. It is assumed that a prior call to the
		/// cusolverRfRefactor() was done in order to generate these triangular factors.
		/// </summary>
		/// <param name="n">Size of Matrix M (n x n)</param>
		/// <param name="h_nnzM">the number of non-zero elements of matrix M.</param>
		/// <param name="h_Mp">the array of offsets corresponding to the start of each row in the arrays Mi and Mx.
		/// This array has also an extra entry at the end that stores the number of non-zero elements in the matrix $M$. The array size is n+1.</param>
		/// <param name="h_Mi">the array of column indices corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
		/// <param name="h_Mx">the array of values corresponding to the non-zero elements in the matrix M. It is assumed that this array is sorted by row and by column within each row. The array size is nnzM.</param>
		public void ExtractBundledFactorsHost(int n, out int h_nnzM, out int[] h_Mp, out int[] h_Mi, out double[] h_Mx)
		{
			h_nnzM = 0;
			IntPtr mp = new IntPtr();
			IntPtr mi = new IntPtr();
			IntPtr mx = new IntPtr();

			res = CudaSolveNativeMethods.Refactorization.cusolverRfExtractBundledFactorsHost(_handle, ref h_nnzM, ref mp, ref mi, ref mx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfExtractBundledFactorsHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);

			if (h_nnzM == 0)
			{
				h_Mp = null;
				h_Mi = null;
				h_Mx = null;
			}
			else
			{
				h_Mp = new int[n+1];
				h_Mi = new int[h_nnzM];
				h_Mx = new double[h_nnzM];
				Marshal.Copy(mp, h_Mp, 0, n + 1);
				Marshal.Copy(mi, h_Mi, 0, h_nnzM);
				Marshal.Copy(mx, h_Mx, 0, h_nnzM);
			}

		}

		/// <summary>
		/// This routine extracts lower (L) and upper (U) triangular factors from the
		/// cuSolverRF library handle into the host memory. It is assumed that a prior call to the
		/// cusolverRfRefactor() was done in order to generate these triangular factors.
		/// </summary>
		/// <param name="n">Size of Matrix M (n x n)</param>
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
		public void ExtractSplitFactorsHost(int n, out int h_nnzL, out int[] h_csrRowPtrL, out int[] h_csrColIndL, out double[] h_csrValL, out int h_nnzU, out int[] h_csrRowPtrU, out int[] h_csrColIndU, out double[] h_csrValU)
		{
			h_nnzL = 0;
			h_nnzU = 0;
			IntPtr RowPtrL = new IntPtr();
			IntPtr ColIndL = new IntPtr();
			IntPtr ValL = new IntPtr();
			IntPtr RowPtrU = new IntPtr();
			IntPtr ColIndU = new IntPtr();
			IntPtr ValU = new IntPtr();

			res = CudaSolveNativeMethods.Refactorization.cusolverRfExtractSplitFactorsHost(_handle, ref h_nnzL, ref RowPtrL, ref ColIndL, ref ValL, ref h_nnzU, ref RowPtrU, ref ColIndU, ref ValU);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfExtractSplitFactorsHost", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);

			if (h_nnzL == 0)
			{
				h_csrRowPtrL = null;
				h_csrColIndL = null;
				h_csrValL = null;
			}
			else
			{
				h_csrRowPtrL = new int[n + 1];
				h_csrColIndL = new int[h_nnzL];
				h_csrValL = new double[h_nnzL];
				Marshal.Copy(RowPtrL, h_csrRowPtrL, 0, n + 1);
				Marshal.Copy(ColIndL, h_csrColIndL, 0, h_nnzL);
				Marshal.Copy(ValL, h_csrValL, 0, h_nnzL);
			}
			if (h_nnzU == 0)
			{
				h_csrRowPtrU = null;
				h_csrColIndU = null;
				h_csrValU = null;
			}
			else
			{
				h_csrRowPtrU = new int[n + 1];
				h_csrColIndU = new int[h_nnzU];
				h_csrValU = new double[h_nnzU];
				Marshal.Copy(RowPtrU, h_csrRowPtrU, 0, n + 1);
				Marshal.Copy(ColIndU, h_csrColIndU, 0, h_nnzU);
				Marshal.Copy(ValU, h_csrValU, 0, h_nnzU);
			}
		}


		/// <summary>
		/// This routine performs the forward and backward solve with the lower and upper
		/// triangular factors resulting from the LU re-factorization
		/// </summary>
		/// <param name="P">the left permutation (often associated with pivoting). The array size in n.</param>
		/// <param name="Q">the right permutation (often associated with reordering). The array size in n.</param>
		/// <param name="nrhs">the number right-hand-sides to be solved.</param>
		/// <param name="Temp">the dense matrix that contains temporary workspace (of size ldt*nrhs).</param>
		/// <param name="ldt">the leading dimension of dense matrix Temp (ldt &gt;= n).</param>
		/// <param name="XF">the dense matrix that contains the righthand-sides F and solutions X (of size ldxf*nrhs).</param>
		/// <param name="ldxf">the leading dimension of dense matrix XF (ldxf &gt;= n).</param>
		public void Solve(CudaDeviceVariable<int> P, CudaDeviceVariable<int> Q, int nrhs, double[] Temp, int ldt, double[] XF, int ldxf)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfSolve(_handle, P.DevicePointer, Q.DevicePointer, nrhs, Temp, ldt, XF, ldxf);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfSolve", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
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
		public void BatchSetupHost(int batchSize, int n, int nnzA, int[] h_csrRowPtrA, int[] h_csrColIndA, double[][] h_csrValA_array, int nnzL,
			int[] h_csrRowPtrL, int[] h_csrColIndL, double[] h_csrValL, int nnzU, int[] h_csrRowPtrU, int[] h_csrColIndU, double[] h_csrValU, int[] h_P, int[] h_Q, cusolverRfHandle handle)
		{
			if (batchSize > h_csrValA_array.Length)
			{
				throw new ArgumentException("batchSize must be smaller or equal to the length of h_csrValA_array.");
			}

			IntPtr[] valA = new IntPtr[batchSize];
			GCHandle[] handles = new GCHandle[batchSize];

			try
			{
				for (int i = 0; i < batchSize; i++)
				{
					handles[i] = GCHandle.Alloc(h_csrValA_array[i], GCHandleType.Pinned);
					valA[i] = handles[i].AddrOfPinnedObject();
				}

				res = CudaSolveNativeMethods.Refactorization.cusolverRfBatchSetupHost(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, valA, nnzL,
				h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, _handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfBatchSetupHost", res));
			}
			catch
			{
				throw;
			}
			finally
			{
				for (int i = 0; i < batchSize; i++)
				{
					handles[i].Free();
				}
			}
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}


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
		public void BatchResetValues(int batchSize, int n, int nnzA, CudaDeviceVariable<int> csrRowPtrA, CudaDeviceVariable<int> csrColIndA, CudaDeviceVariable<double>[] csrValA_array, CudaDeviceVariable<int> P, CudaDeviceVariable<int> Q)
		{
			if (batchSize > csrValA_array.Length)
			{
				throw new ArgumentException("batchSize must be smaller or equal to the length of csrValA_array.");
			}

			CUdeviceptr[] valA = new CUdeviceptr[batchSize];
			for (int i = 0; i < batchSize; i++)
			{
				valA[i] = csrValA_array[i].DevicePointer;
			}

			res = CudaSolveNativeMethods.Refactorization.cusolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA.DevicePointer, csrColIndA.DevicePointer, valA, P.DevicePointer, Q.DevicePointer, _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfBatchResetValues", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This routine performs the appropriate analysis of parallelism available in the batched LU re-factorization.
		/// </summary>
		public void BatchAnalyze()
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfBatchAnalyze( _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfBatchAnalyze", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// This routine performs the LU re-factorization
		/// </summary>
		public void BatchRefactor()
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfBatchRefactor(_handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfBatchRefactor", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// To solve A_j * x_j = b_j, first we reform the equation by M_j * Q * x_j = P * b_j. Then do refactorization by
		/// cusolverRfBatch_Refactor(). Further cusolverRfBatch_Solve() takes over the remaining steps.
		/// </summary>
		/// <param name="P">the left permutation (often associated with pivoting). The array size in n.</param>
		/// <param name="Q">the right permutation (often associated with reordering). The array size in n.</param>
		/// <param name="nrhs">the number right-hand-sides to be solved.</param>
		/// <param name="Temp">the dense matrix that contains temporary workspace (of size ldt*nrhs).</param>
		/// <param name="ldt">the leading dimension of dense matrix Temp (ldt &gt;= n).</param>
		/// <param name="XF_array">array of pointers of size batchSize, each pointer points to the dense matrix that contains the right-hand-sides F and solutions X (of size ldxf*nrhs).</param>
		/// <param name="ldxf">the leading dimension of dense matrix XF (ldxf &gt;= n).</param>
		public void cusolverRfBatchSolve(CudaDeviceVariable<int> P, CudaDeviceVariable<int> Q, int nrhs, double[] Temp, int ldt, double[][] XF_array, int ldxf)
		{
			int batchSize = XF_array.Length;

			IntPtr[] XF = new IntPtr[batchSize];
			GCHandle[] handles = new GCHandle[batchSize];

			try
			{
				for (int i = 0; i < batchSize; i++)
				{
					handles[i] = GCHandle.Alloc(XF_array[i], GCHandleType.Pinned);
					XF[i] = handles[i].AddrOfPinnedObject();
				}

				res = CudaSolveNativeMethods.Refactorization.cusolverRfBatchSolve(_handle, P.DevicePointer, Q.DevicePointer, nrhs, Temp, ldt, XF, ldxf);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfBatchSolve", res));
			}
			catch
			{
				throw;
			}
			finally
			{
				for (int i = 0; i < batchSize; i++)
				{
					handles[i].Free();
				}
			}
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}

		/// <summary>
		/// The user can query which matrix failed LU refactorization by checking
		/// corresponding value in position array. The input parameter position is an integer array of size batchSize.
		/// </summary>
		/// <param name="position">integer array of size batchSize. The value of position(j) reports singularity
		/// of matrix Aj, -1 if no structural / numerical zero, k &gt;= 0 if Aj(k,k) is either structural zero or numerical zero.</param>
		public void BatchZeroPivot(int[] position)
		{
			res = CudaSolveNativeMethods.Refactorization.cusolverRfBatchZeroPivot(_handle, position);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverRfBatchZeroPivot", res));
			if (res != cusolverStatus.Success) throw new CudaSolveException(res);
		}
		#endregion
	}
}
