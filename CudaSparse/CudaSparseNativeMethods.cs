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
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;


namespace ManagedCuda.CudaSparse
{
	/// <summary>
	/// C# wrapper for cusparse.h
	/// </summary>
	public static class CudaSparseNativeMethods
	{
		internal const string CUSPARSE_API_DLL_NAME = "cusparse64_100";

		#region CUSPARSE initialization and managment routines
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreate(ref cusparseContext handle);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroy(cusparseContext handle);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseGetVersion(cusparseContext handle, ref int version);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSetStream(cusparseContext handle, CUstream streamId);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseGetStream(cusparseContext handle, ref CUstream streamId);
        #endregion

        #region CUSPARSE type creation, destruction, set and get routines

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseGetPointerMode(cusparseContext handle, ref cusparsePointerMode mode);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSetPointerMode(cusparseContext handle, cusparsePointerMode mode);
		#endregion

		#region sparse matrix descriptor
		/* When the matrix descriptor is created, its fields are initialized to: 
		CUSPARSE_MATRIXYPE_GENERAL
		CUSPARSE_INDEX_BASE_ZERO
		All other fields are uninitialized
		*/
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateMatDescr(ref cusparseMatDescr descrA);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyMatDescr(cusparseMatDescr descrA);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCopyMatDescr(cusparseMatDescr dest, cusparseMatDescr src);
		

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSetMatType(cusparseMatDescr descrA, cusparseMatrixType type);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseMatrixType cusparseGetMatType(cusparseMatDescr descrA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSetMatFillMode(cusparseMatDescr descrA, cusparseFillMode fillMode);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseFillMode cusparseGetMatFillMode(cusparseMatDescr descrA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSetMatDiagType(cusparseMatDescr descrA, cusparseDiagType diagType);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseDiagType cusparseGetMatDiagType(cusparseMatDescr descrA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSetMatIndexBase(cusparseMatDescr descrA, cusparseIndexBase indexBase);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseIndexBase cusparseGetMatIndexBase(cusparseMatDescr descrA);
		#endregion

		#region sparse traingular solve
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateSolveAnalysisInfo(ref cusparseSolveAnalysisInfo info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseGetLevelInfo(cusparseContext handle, 
                                                  cusparseSolveAnalysisInfo info, 
                                                  ref int nlevels,
												  ref CUdeviceptr levelPtr,
												  ref CUdeviceptr levelInd);

		#endregion

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateCsrsv2Info(ref csrsv2Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyCsrsv2Info(csrsv2Info info);

		#region incomplete Cholesky
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateCsric02Info(ref csric02Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyCsric02Info(csric02Info info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateBsric02Info(ref bsric02Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyBsric02Info(bsric02Info info);
		#endregion

		#region incomplete LU
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateCsrilu02Info(ref csrilu02Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyCsrilu02Info(csrilu02Info info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateBsrilu02Info(ref bsrilu02Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyBsrilu02Info(bsrilu02Info info);

		#endregion

		#region BSR triangular solber
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateBsrsv2Info(ref bsrsv2Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyBsrsv2Info(bsrsv2Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateBsrsm2Info(ref bsrsm2Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyBsrsm2Info(bsrsm2Info info);
		#endregion

		#region hybrid (HYB) format
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateHybMat(ref cusparseHybMat hybA);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyHybMat(cusparseHybMat hybA);
		#endregion

		#region sorting information
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateCsru2csrInfo(ref csru2csrInfo info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyCsru2csrInfo(csru2csrInfo info);
		#endregion

		#region coloring info
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateColorInfo(ref cusparseColorInfo info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyColorInfo(cusparseColorInfo info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSetColorAlgs(cusparseColorInfo info, cusparseColorAlg alg);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseGetColorAlgs(cusparseColorInfo info, ref cusparseColorAlg alg);
        #endregion

        #region prune information
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreatePruneInfo(ref pruneInfo info);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDestroyPruneInfo(pruneInfo info);
        #endregion

        #region Csrgemm2Info
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateCsrgemm2Info(ref csrgemm2Info info);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDestroyCsrgemm2Info(csrgemm2Info info);
		#endregion

		#region Sparse Level 1 routines
		/* Description: Addition of a scalar multiple of a sparse vector x  
		and a dense vector y. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSaxpyi(cusparseContext handle, int nnz, ref float alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDaxpyi(cusparseContext handle, int nnz, ref double alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCaxpyi(cusparseContext handle, int nnz, ref cuFloatComplex alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZaxpyi(cusparseContext handle, int nnz, ref cuDoubleComplex alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSaxpyi(cusparseContext handle, int nnz, CUdeviceptr alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDaxpyi(cusparseContext handle, int nnz, CUdeviceptr alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCaxpyi(cusparseContext handle, int nnz, CUdeviceptr alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZaxpyi(cusparseContext handle, int nnz, CUdeviceptr alpha, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);
		#endregion

		/* Description: dot product of a sparse vector x and a dense vector y. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref float resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref double resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref cuFloatComplex resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref cuDoubleComplex resultDevHostPtr, cusparseIndexBase idxBase);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZdoti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr resultDevHostPtr, cusparseIndexBase idxBase);
		#endregion

		/* Description: dot product of complex conjugate of a sparse vector x
		and a dense vector y. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCdotci(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref cuFloatComplex resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZdotci(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref cuDoubleComplex resultDevHostPtr, cusparseIndexBase idxBase);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCdotci(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr resultDevHostPtr, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZdotci(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr resultDevHostPtr, cusparseIndexBase idxBase);
		#endregion


		/* Description: Gather of non-zero elements from dense vector y into 
		sparse vector x. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgthr(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgthr(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgthr(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgthr(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/* Description: Gather of non-zero elements from desne vector y into 
		sparse vector x (also replacing these elements in y by zeros). */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgthrz(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgthrz(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgthrz(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgthrz(cusparseContext handle, int nnz, CUdeviceptr y, CUdeviceptr xVal, CUdeviceptr xInd, cusparseIndexBase idxBase);

		/* Description: Scatter of elements of the sparse vector x into 
		dense vector y. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSsctr(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDsctr(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsctr(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZsctr(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, cusparseIndexBase idxBase);

		/* Description: Givens rotation, where c and s are cosine and sine, 
		x and y are sparse and dense vectors, respectively. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSroti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref float c, ref float s, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDroti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, ref double c, ref double s, cusparseIndexBase idxBase);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSroti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr c, CUdeviceptr s, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDroti(cusparseContext handle, int nnz, CUdeviceptr xVal, CUdeviceptr xInd, CUdeviceptr y, CUdeviceptr c, CUdeviceptr s, cusparseIndexBase idxBase);
		#endregion

		#endregion

		#region Sparse Level 2 routines
		//new in Cuda 8.0

		#region ref host
		//Returns number of bytes
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrmvEx_bufferSize(cusparseContext handle, 
                                                        cusparseAlgMode alg,
                                                        cusparseOperation transA, 
                                                        int m, 
                                                        int n, 
                                                        int nnz,
                                                        IntPtr alpha,
                                                        cudaDataType alphatype,
                                                        cusparseMatDescr descrA,
                                                        CUdeviceptr csrValA,
                                                        cudaDataType csrValAtype,
                                                        CUdeviceptr csrRowPtrA,
                                                        CUdeviceptr csrColIndA,
                                                        CUdeviceptr x,
                                                        cudaDataType xtype,
                                                        IntPtr beta,
                                                        cudaDataType betatype,
                                                        CUdeviceptr y,
                                                        cudaDataType ytype,
                                                        cudaDataType executiontype,
                                                        ref SizeT bufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrmvEx(cusparseContext handle, 
                                             cusparseAlgMode alg,
                                             cusparseOperation transA, 
                                             int m, 
                                             int n, 
                                             int nnz,
                                             IntPtr alpha,
                                             cudaDataType alphatype,
                                             cusparseMatDescr descrA,
                                             CUdeviceptr csrValA,
                                             cudaDataType csrValAtype,
                                             CUdeviceptr csrRowPtrA,
                                             CUdeviceptr csrColIndA,
                                             CUdeviceptr x,
                                             cudaDataType xtype,
                                             IntPtr beta,
                                             cudaDataType betatype,
                                             CUdeviceptr y,
                                             cudaDataType ytype,
                                             cudaDataType executiontype,
											 CUdeviceptr buffer);

		/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
		   where A is a sparse matrix in CSR storage format, x and y are dense vectors
		   using a Merge Path load-balancing implementation. */ 
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmv_mp(cusparseContext handle,
                                            cusparseOperation transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            ref float alpha,
                                            cusparseMatDescr descrA, 
                                            CUdeviceptr csrSortedValA, 
                                            CUdeviceptr csrSortedRowPtrA, 
                                            CUdeviceptr csrSortedColIndA, 
                                            CUdeviceptr x, 
                                            ref float beta, 
                                            CUdeviceptr y);
    
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmv_mp(cusparseContext handle,
                                            cusparseOperation transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            ref double alpha,
                                            cusparseMatDescr descrA, 
                                            CUdeviceptr csrSortedValA, 
                                            CUdeviceptr csrSortedRowPtrA, 
                                            CUdeviceptr csrSortedColIndA, 
                                            CUdeviceptr x, 
                                            ref double beta,  
                                            CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmv_mp(cusparseContext handle,
                                            cusparseOperation transA, 
                                            int m, 
                                            int n,
                                            int nnz,
                                            ref cuFloatComplex alpha,
                                            cusparseMatDescr descrA,
											CUdeviceptr csrSortedValA, 
                                            CUdeviceptr csrSortedRowPtrA, 
                                            CUdeviceptr csrSortedColIndA,
											CUdeviceptr x,
											ref cuFloatComplex beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmv_mp(cusparseContext handle,
                                            cusparseOperation transA, 
                                            int m, 
                                            int n, 
                                            int nnz,
                                            ref cuDoubleComplex alpha,
                                            cusparseMatDescr descrA, 
                                            CUdeviceptr csrSortedValA, 
                                            CUdeviceptr csrSortedRowPtrA, 
                                            CUdeviceptr csrSortedColIndA, 
                                            CUdeviceptr x, 
                                            ref cuDoubleComplex beta, 
                                            CUdeviceptr y);
		#endregion

		#region ref device
		//Returns number of bytes
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrmvEx_bufferSize(cusparseContext handle,
														cusparseAlgMode alg,
														cusparseOperation transA,
														int m,
														int n,
														int nnz,
														CUdeviceptr alpha,
														cudaDataType alphatype,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														cudaDataType csrValAtype,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														CUdeviceptr x,
														cudaDataType xtype,
														CUdeviceptr beta,
														cudaDataType betatype,
														CUdeviceptr y,
														cudaDataType ytype,
														cudaDataType executiontype,
														ref SizeT bufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrmvEx(cusparseContext handle,
											 cusparseAlgMode alg,
											 cusparseOperation transA,
											 int m,
											 int n,
											 int nnz,
											 CUdeviceptr alpha,
											 cudaDataType alphatype,
											 cusparseMatDescr descrA,
											 CUdeviceptr csrValA,
											 cudaDataType csrValAtype,
											 CUdeviceptr csrRowPtrA,
											 CUdeviceptr csrColIndA,
											 CUdeviceptr x,
											 cudaDataType xtype,
											 CUdeviceptr beta,
											 cudaDataType betatype,
											 CUdeviceptr y,
											 cudaDataType ytype,
											 cudaDataType executiontype,
											 CUdeviceptr buffer);

		/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
		   where A is a sparse matrix in CSR storage format, x and y are dense vectors
		   using a Merge Path load-balancing implementation. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmv_mp(cusparseContext handle,
											cusparseOperation transA,
											int m,
											int n,
											int nnz,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr csrSortedValA,
											CUdeviceptr csrSortedRowPtrA,
											CUdeviceptr csrSortedColIndA,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmv_mp(cusparseContext handle,
											cusparseOperation transA,
											int m,
											int n,
											int nnz,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr csrSortedValA,
											CUdeviceptr csrSortedRowPtrA,
											CUdeviceptr csrSortedColIndA,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmv_mp(cusparseContext handle,
											cusparseOperation transA,
											int m,
											int n,
											int nnz,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr csrSortedValA,
											CUdeviceptr csrSortedRowPtrA,
											CUdeviceptr csrSortedColIndA,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmv_mp(cusparseContext handle,
											cusparseOperation transA,
											int m,
											int n,
											int nnz,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr csrSortedValA,
											CUdeviceptr csrSortedRowPtrA,
											CUdeviceptr csrSortedColIndA,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);
		#endregion


		//new in Cuda 7.5
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgemvi(cusparseContext handle,
                                    cusparseOperation transA,
                                    int m,
                                    int n,
                                    ref float alpha, /* host or device pointer */
                                    CUdeviceptr A,
                                    int lda,
                                    int nnz,
                                    CUdeviceptr xVal,
                                    CUdeviceptr xInd,
                                    ref float beta, /* host or device pointer */
                                    CUdeviceptr y,
                                    cusparseIndexBase   idxBase,
                                    CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgemvi_bufferSize(cusparseContext handle,
									cusparseOperation transA, int m, int n, int nnz, ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgemvi(cusparseContext handle,
                                    cusparseOperation transA,
                                    int m,
                                    int n,
                                    ref double alpha, /* host or device pointer */
                                    CUdeviceptr A,
                                    int lda,
                                    int nnz,
                                    CUdeviceptr xVal,
                                    CUdeviceptr xInd,
                                    ref double beta, /* host or device pointer */
                                    CUdeviceptr y,
                                    cusparseIndexBase   idxBase,
                                    CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgemvi_bufferSize(cusparseContext handle,
									cusparseOperation transA, int m, int n, int nnz, ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgemvi(cusparseContext handle,
                                    cusparseOperation transA,
                                    int m,
                                    int n,
                                    ref cuFloatComplex alpha, /* host or device pointer */
                                    CUdeviceptr A,
                                    int lda,
                                    int nnz,
                                    CUdeviceptr xVal,
                                    CUdeviceptr xInd,
                                    ref cuFloatComplex beta, /* host or device pointer */
                                    CUdeviceptr y,
                                    cusparseIndexBase   idxBase,
                                    CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgemvi_bufferSize(cusparseContext handle,
									cusparseOperation transA, int m, int n, int nnz, ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgemvi(cusparseContext handle,
                                    cusparseOperation transA,
                                    int m,
                                    int n,
                                    ref cuDoubleComplex alpha, /* host or device pointer */
                                    CUdeviceptr A,
                                    int lda,
                                    int nnz,
                                    CUdeviceptr xVal,
									CUdeviceptr xInd,
                                    ref cuDoubleComplex beta, /* host or device pointer */
									CUdeviceptr y,
                                    cusparseIndexBase   idxBase,
									CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgemvi_bufferSize( cusparseContext handle,
									cusparseOperation transA, int m, int n, int nnz, ref int pBufferSize);


		#endregion

		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgemvi(cusparseContext handle,
									cusparseOperation transA,
									int m,
									int n,
									CUdeviceptr alpha, /* host or device pointer */
									CUdeviceptr A,
									int lda,
									int nnz,
									CUdeviceptr xVal,
									CUdeviceptr xInd,
									CUdeviceptr beta, /* host or device pointer */
									CUdeviceptr y,
									cusparseIndexBase idxBase,
									CUdeviceptr pBuffer);
		
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgemvi(cusparseContext handle,
									cusparseOperation transA,
									int m,
									int n,
									CUdeviceptr alpha, /* host or device pointer */
									CUdeviceptr A,
									int lda,
									int nnz,
									CUdeviceptr xVal,
									CUdeviceptr xInd,
									CUdeviceptr beta, /* host or device pointer */
									CUdeviceptr y,
									cusparseIndexBase idxBase,
									CUdeviceptr pBuffer);
		
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgemvi(cusparseContext handle,
									cusparseOperation transA,
									int m,
									int n,
									CUdeviceptr alpha, /* host or device pointer */
									CUdeviceptr A,
									int lda,
									int nnz,
									CUdeviceptr xVal,
									CUdeviceptr xInd,
									CUdeviceptr beta, /* host or device pointer */
									CUdeviceptr y,
									cusparseIndexBase idxBase,
									CUdeviceptr pBuffer);
		
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgemvi(cusparseContext handle,
									cusparseOperation transA,
									int m,
									int n,
									CUdeviceptr alpha, /* host or device pointer */
									CUdeviceptr A,
									int lda,
									int nnz,
									CUdeviceptr xVal,
									CUdeviceptr xInd,
									CUdeviceptr beta, /* host or device pointer */
									CUdeviceptr y,
									cusparseIndexBase idxBase,
									CUdeviceptr pBuffer);

		#endregion


		/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, ref float alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, ref float beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, ref double alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, ref double beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, ref cuFloatComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, ref cuFloatComplex beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, ref cuDoubleComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, ref cuDoubleComplex beta, CUdeviceptr y);

		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmv(cusparseContext handle, cusparseOperation transA, int m, int n, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);
		#endregion

		/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
		   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShybmv(cusparseContext handle, cusparseOperation transA, ref float alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, ref float beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhybmv(cusparseContext handle, cusparseOperation transA, ref double alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, ref double beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChybmv(cusparseContext handle, cusparseOperation transA, ref cuFloatComplex alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, ref cuFloatComplex beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhybmv(cusparseContext handle, cusparseOperation transA, ref cuDoubleComplex alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, ref cuDoubleComplex beta, CUdeviceptr y);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShybmv(cusparseContext handle, cusparseOperation transA, CUdeviceptr alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhybmv(cusparseContext handle, cusparseOperation transA, CUdeviceptr alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChybmv(cusparseContext handle, cusparseOperation transA, CUdeviceptr alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhybmv(cusparseContext handle, cusparseOperation transA, CUdeviceptr alpha, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr x, CUdeviceptr beta, CUdeviceptr y);
		#endregion

		/* Description: Solution of triangular linear system op(A) * y = alpha * x, 
		   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */  
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrsv_analysisEx(cusparseContext handle, 
                                                     cusparseOperation transA, 
                                                     int m, 
                                                     int nnz,
                                                     cusparseMatDescr descrA, 
                                                     CUdeviceptr csrSortedValA,
                                                     cudaDataType csrSortedValAtype,
                                                     CUdeviceptr csrSortedRowPtrA, 
                                                     CUdeviceptr csrSortedColIndA, 
                                                     cusparseSolveAnalysisInfo info,
                                                     cudaDataType executiontype);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrsv_solveEx(cusparseContext handle, 
                                                   cusparseOperation transA, 
                                                   int m,
                                                   IntPtr alpha, 
                                                   cudaDataType alphatype,
                                                   cusparseMatDescr descrA,
												   CUdeviceptr csrSortedValA, 
                                                   cudaDataType csrSortedValAtype,
												   CUdeviceptr csrSortedRowPtrA,
												   CUdeviceptr csrSortedColIndA, 
                                                   cusparseSolveAnalysisInfo info,
												   CUdeviceptr f, 
                                                   cudaDataType ftype,
												   CUdeviceptr x,
                                                   cudaDataType xtype,
                                                   cudaDataType executiontype);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrsv_solveEx(cusparseContext handle, 
                                                   cusparseOperation transA, 
                                                   int m,
												   CUdeviceptr alpha, 
                                                   cudaDataType alphatype,
                                                   cusparseMatDescr descrA,
												   CUdeviceptr csrSortedValA, 
                                                   cudaDataType csrSortedValAtype,
												   CUdeviceptr csrSortedRowPtrA,
												   CUdeviceptr csrSortedColIndA, 
                                                   cusparseSolveAnalysisInfo info,
												   CUdeviceptr f, 
                                                   cudaDataType ftype,
												   CUdeviceptr x,
                                                   cudaDataType xtype,
                                                   cudaDataType executiontype);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);


		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, ref float alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, ref double alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, ref cuFloatComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, ref cuDoubleComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv_solve(cusparseContext handle, cusparseOperation transA, int m, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);
		#endregion


		/* Description: Solution of triangular linear system op(A) * y = alpha * x, 
		   where A is a sparse matrix in CSR storage format, x and y are dense vectors. 
		   The new API provides utility function to query size of buffer used. */

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv2_bufferSize(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv2_bufferSize(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv2_bufferSize(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv2_bufferSize(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref int pBufferSize);



		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseOperation transA,
														int m,
														int nnz,
														cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
														csrsv2Info info,
														ref SizeT pBufferSize);




		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv2_analysis(cusparseContext handle,
													  cusparseOperation transA,
													  int m,
													  int nnz,
													  cusparseMatDescr descrA,
													  CUdeviceptr csrValA,
													  CUdeviceptr csrRowPtrA,
													  CUdeviceptr csrColIndA,
													  csrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv2_analysis(cusparseContext handle,
													  cusparseOperation transA,
													  int m,
													  int nnz,
													  cusparseMatDescr descrA,
													  CUdeviceptr csrValA,
													  CUdeviceptr csrRowPtrA,
													  CUdeviceptr csrColIndA,
													  csrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv2_analysis(cusparseContext handle,
													  cusparseOperation transA,
													  int m,
													  int nnz,
													  cusparseMatDescr descrA,
													  CUdeviceptr csrValA,
													  CUdeviceptr csrRowPtrA,
													  CUdeviceptr csrColIndA,
													  csrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv2_analysis(cusparseContext handle,
													  cusparseOperation transA,
													  int m,
													  int nnz,
													  cusparseMatDescr descrA,
													  CUdeviceptr csrValA,
													  CUdeviceptr csrRowPtrA,
													  CUdeviceptr csrColIndA,
													  csrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsv2_bufferSize(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsv2_bufferSize(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsv2_bufferSize(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsv2_bufferSize(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref int pBufferSize);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsv2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														int mb,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockDim,
														bsrsv2Info info,
														ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsv2_analysis(cusparseContext handle,
													  cusparseDirection dirA,
													  cusparseOperation transA,
													  int mb,
													  int nnzb,
													  cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
													  int blockDim,
													  bsrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsv2_analysis(cusparseContext handle,
													  cusparseDirection dirA,
													  cusparseOperation transA,
													  int mb,
													  int nnzb,
													  cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
													  int blockDim,
													  bsrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsv2_analysis(cusparseContext handle,
													  cusparseDirection dirA,
													  cusparseOperation transA,
													  int mb,
													  int nnzb,
													  cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
													  int blockDim,
													  bsrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsv2_analysis(cusparseContext handle,
													  cusparseDirection dirA,
													  cusparseOperation transA,
													  int mb,
													  int nnzb,
													  cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
													  int blockDim,
													  bsrsv2Info info,
													  cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);


#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrsv2_zeroPivot(cusparseContext handle,
                                                       csrsv2Info info,
                                                       ref int position);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   ref float alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   ref double alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   ref cuFloatComplex alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   ref cuDoubleComplex alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsrsv2_zeroPivot(cusparseContext handle,
													   bsrsv2Info info,
													   ref int position);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   ref float alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   ref double alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   ref cuFloatComplex alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   ref cuDoubleComplex alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
#endregion

#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrsv2_zeroPivot(cusparseContext handle,
                                                       csrsv2Info info,
                                                       CUdeviceptr position);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   CUdeviceptr alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   CUdeviceptr alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   CUdeviceptr alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsv2_solve(cusparseContext handle,
												   cusparseOperation transA,
												   int m,
												   int nnz,
												   CUdeviceptr alpha,
												   cusparseMatDescr descra,
												   CUdeviceptr csrValA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   csrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsrsv2_zeroPivot(cusparseContext handle,
													   bsrsv2Info info,
													   CUdeviceptr position);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsv2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   int mb,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockDim,
												   bsrsv2Info info,
												   CUdeviceptr x,
												   CUdeviceptr y,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
#endregion












		/* Description: Solution of triangular linear system op(A) * y = alpha * x, 
		   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShybsv_analysis(cusparseContext handle, cusparseOperation transA, cusparseMatDescr descrA, cusparseHybMat hybA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhybsv_analysis(cusparseContext handle, cusparseOperation transA, cusparseMatDescr descrA, cusparseHybMat hybA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChybsv_analysis(cusparseContext handle, cusparseOperation transA, cusparseMatDescr descrA, cusparseHybMat hybA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhybsv_analysis(cusparseContext handle, cusparseOperation transA, cusparseMatDescr descrA, cusparseHybMat hybA, cusparseSolveAnalysisInfo info);

		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShybsv_solve(cusparseContext handle, cusparseOperation trans, ref float alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChybsv_solve(cusparseContext handle, cusparseOperation trans, ref cuFloatComplex alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhybsv_solve(cusparseContext handle, cusparseOperation trans, ref double alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhybsv_solve(cusparseContext handle, cusparseOperation trans, ref cuDoubleComplex alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShybsv_solve(cusparseContext handle, cusparseOperation trans, CUdeviceptr alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChybsv_solve(cusparseContext handle, cusparseOperation trans, CUdeviceptr alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhybsv_solve(cusparseContext handle, cusparseOperation trans, CUdeviceptr alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhybsv_solve(cusparseContext handle, cusparseOperation trans, CUdeviceptr alpha, cusparseMatDescr descra, cusparseHybMat hybA, cusparseSolveAnalysisInfo info, CUdeviceptr x, CUdeviceptr y);
		#endregion

		#endregion

		#region Sparse Level 3 routines

		//new in Cuda 8.0

		/* Description: dense - sparse matrix multiplication C = alpha * A * B  + beta * C, 
		   where A is column-major dense matrix, B is a sparse matrix in CSC format, 
		   and C is column-major dense matrix. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgemmi(cusparseContext handle,
                                             int m,
                                             int n,
											 int k,
											 int nnz, 
                                             ref float alpha, /* host or device pointer */
                                             CUdeviceptr A,
                                             int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB, 
                                             ref float beta, /* host or device pointer */
											 CUdeviceptr C,
                                             int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgemmi(cusparseContext handle,
                                             int m,
                                             int n,
											 int k,
											 int nnz, 
                                             ref double alpha, /* host or device pointer */
											 CUdeviceptr A,
                                             int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB, 
                                             ref double beta, /* host or device pointer */
											 CUdeviceptr C,
                                             int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgemmi(cusparseContext handle,
                                             int m,
                                             int n,
											 int k,
											 int nnz, 
                                             ref cuFloatComplex alpha, /* host or device pointer */
											 CUdeviceptr A,
                                             int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB, 
                                             ref cuFloatComplex beta, /* host or device pointer */
											 CUdeviceptr C,
                                             int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgemmi(cusparseContext handle,
                                             int m,
                                             int n,
											 int k,
											 int nnz, 
                                             ref cuDoubleComplex alpha, /* host or device pointer */
											 CUdeviceptr A,
                                             int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB, 
                                             ref cuDoubleComplex beta, /* host or device pointer */
											 CUdeviceptr C,
                                             int ldc);

		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgemmi(cusparseContext handle,
											 int m,
											 int n,
											 int k,
											 int nnz,
											 CUdeviceptr alpha, /* host or device pointer */
											 CUdeviceptr A,
											 int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB,
											 CUdeviceptr beta, /* host or device pointer */
											 CUdeviceptr C,
											 int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgemmi(cusparseContext handle,
											 int m,
											 int n,
											 int k,
											 int nnz,
											 CUdeviceptr alpha, /* host or device pointer */
											 CUdeviceptr A,
											 int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB,
											 CUdeviceptr beta, /* host or device pointer */
											 CUdeviceptr C,
											 int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgemmi(cusparseContext handle,
											 int m,
											 int n,
											 int k,
											 int nnz,
											 CUdeviceptr alpha, /* host or device pointer */
											 CUdeviceptr A,
											 int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB,
											 CUdeviceptr beta, /* host or device pointer */
											 CUdeviceptr C,
											 int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgemmi(cusparseContext handle,
											 int m,
											 int n,
											 int k,
											 int nnz,
											 CUdeviceptr alpha, /* host or device pointer */
											 CUdeviceptr A,
											 int lda,
											 CUdeviceptr cscValB,
											 CUdeviceptr cscColPtrB,
											 CUdeviceptr cscRowIndB,
											 CUdeviceptr beta, /* host or device pointer */
											 CUdeviceptr C,
											 int ldc);

		#endregion

		/* Description: Matrix-matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix, B and C are dense and usually tall matrices. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, ref float alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref float beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, ref double alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref double beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, ref cuFloatComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref cuFloatComplex beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, ref cuDoubleComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref cuDoubleComplex beta, CUdeviceptr C, int ldc);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmm(cusparseContext handle, cusparseOperation transA, int m, int n, int k, int nnz, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);
		#endregion

		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
                                            ref float alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref float beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
                                            ref double alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref double beta, CUdeviceptr C, int ldc);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
                                            ref cuFloatComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref cuFloatComplex beta, CUdeviceptr C, int ldc);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
											ref cuDoubleComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, ref cuDoubleComplex beta, CUdeviceptr C, int ldc);
		#endregion
		#region ref device

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
											CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
											CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);
		/// <summary/> 
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
											CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrmm2(cusparseContext handle, cusparseOperation transa, cusparseOperation transb, int m, int n, int k, int nnz,
											CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr B, int ldb, CUdeviceptr beta, CUdeviceptr C, int ldc);

		#endregion

		

		/* Description: Solution of triangular linear system op(A) * Y = alpha * X, 
		   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
		   format, X and Y are dense and usually tall matrices. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsm_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsm_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsm_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsm_analysis(cusparseContext handle, cusparseOperation transA, int m, int nnz, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info);


		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, ref float alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, ref double alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, ref cuFloatComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, ref cuDoubleComplex alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrsm_solve(cusparseContext handle, cusparseOperation transA, int m, int n, CUdeviceptr alpha, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseSolveAnalysisInfo info, CUdeviceptr x, int ldx, CUdeviceptr y, int ldy);
        #endregion

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateCsrsm2Info(ref csrsm2Info info);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDestroyCsrsm2Info(csrsm2Info info);

        #region host memory
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseXcsrsm2_zeroPivot(cusparseContext handle, csrsm2Info info, ref int position);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref float alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref double alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref cuFloatComplex alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref cuDoubleComplex alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref float alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref double alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref cuFloatComplex alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref cuDoubleComplex alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref float alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref double alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref cuFloatComplex alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            ref cuDoubleComplex alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);
        #endregion

        #region device memory

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseXcsrsm2_zeroPivot(cusparseContext handle, csrsm2Info info, CUdeviceptr position);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrsm2_bufferSizeExt(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            ref SizeT pBufferSize);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrsm2_analysis(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrsm2_solve(
            cusparseContext handle,
            int algo, /* algo = 0, 1 */
            cusparseOperation transA,
            cusparseOperation transB,
            int m,
            int nrhs,
            int nnz,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr B,
            int ldb,
            csrsm2Info info,
            cusparseSolvePolicy policy,
            CUdeviceptr pBuffer);
        #endregion


        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsm2_bufferSize(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        cusparseOperation transA,
                                                        cusparseOperation transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
                                                        CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockSize,
                                                        bsrsm2Info info,
                                                        ref int pBufferSizeInBytes);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsm2_bufferSize(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        cusparseOperation transA,
                                                        cusparseOperation transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockSize,
                                                        bsrsm2Info info,
														ref int pBufferSizeInBytes);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsm2_bufferSize(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        cusparseOperation transA,
                                                        cusparseOperation transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockSize,
                                                        bsrsm2Info info,
														ref int pBufferSizeInBytes);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsm2_bufferSize(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        cusparseOperation transA,
                                                        cusparseOperation transXY,
                                                        int mb,
                                                        int n,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockSize,
                                                        bsrsm2Info info,
														ref int pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsm2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														cusparseOperation transXY,
														int mb,
														int n,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockSize,
														bsrsm2Info info,
														ref SizeT pBufferSizeInBytes);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsm2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														cusparseOperation transXY,
														int mb,
														int n,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockSize,
														bsrsm2Info info,
														ref SizeT pBufferSizeInBytes);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsm2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														cusparseOperation transXY,
														int mb,
														int n,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockSize,
														bsrsm2Info info,
														ref SizeT pBufferSizeInBytes);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsm2_bufferSizeExt(cusparseContext handle,
														cusparseDirection dirA,
														cusparseOperation transA,
														cusparseOperation transXY,
														int mb,
														int n,
														int nnzb,
														cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
														int blockSize,
														bsrsm2Info info,
														ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsm2_analysis(cusparseContext handle,
                                                      cusparseDirection dirA,
                                                      cusparseOperation transA,
                                                      cusparseOperation transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
                                                      int blockSize,
                                                      bsrsm2Info info,
                                                      cusparseSolvePolicy policy,
                                                      CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsm2_analysis(cusparseContext handle,
                                                      cusparseDirection dirA,
                                                      cusparseOperation transA,
                                                      cusparseOperation transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
                                                      int blockSize,
                                                      bsrsm2Info info,
                                                      cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsm2_analysis(cusparseContext handle,
                                                      cusparseDirection dirA,
                                                      cusparseOperation transA,
                                                      cusparseOperation transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
                                                      int blockSize,
                                                      bsrsm2Info info,
                                                      cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsm2_analysis(cusparseContext handle,
                                                      cusparseDirection dirA,
                                                      cusparseOperation transA,
                                                      cusparseOperation transXY,
                                                      int mb,
                                                      int n,
                                                      int nnzb,
                                                      cusparseMatDescr descrA,
													  CUdeviceptr bsrVal,
													  CUdeviceptr bsrRowPtr,
													  CUdeviceptr bsrColInd,
                                                      int blockSize,
                                                      bsrsm2Info info,
                                                      cusparseSolvePolicy policy,
													  CUdeviceptr pBuffer);




		#region host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsrsm2_zeroPivot(cusparseContext handle,
													   bsrsm2Info info,
													   ref int position);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											ref float alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											ref float beta,
											CUdeviceptr C,
											int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											ref double alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											ref double beta,
											CUdeviceptr C,
											int ldc);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											ref cuFloatComplex alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											ref cuFloatComplex beta,
											CUdeviceptr C,
											int ldc);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											ref cuDoubleComplex alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											ref cuDoubleComplex beta,
											CUdeviceptr C,
											int ldc);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   ref float alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   ref double alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   ref cuFloatComplex alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   ref cuDoubleComplex alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		#endregion

		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsrsm2_zeroPivot(cusparseContext handle,
													   bsrsm2Info info,
													   CUdeviceptr position);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											CUdeviceptr beta,
											CUdeviceptr C,
											int ldc);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											CUdeviceptr beta,
											CUdeviceptr C,
											int ldc);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											CUdeviceptr beta,
											CUdeviceptr C,
											int ldc);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrmm(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											cusparseOperation transB,
											int mb,
											int n,
											int kb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockSize,
											CUdeviceptr B,
											int ldb,
											CUdeviceptr beta,
											CUdeviceptr C,
											int ldc);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrsm2_solve(cusparseContext handle,
												   cusparseDirection dirA,
												   cusparseOperation transA,
												   cusparseOperation transXY,
												   int mb,
												   int n,
												   int nnzb,
												   CUdeviceptr alpha,
												   cusparseMatDescr descrA,
												   CUdeviceptr bsrVal,
												   CUdeviceptr bsrRowPtr,
												   CUdeviceptr bsrColInd,
												   int blockSize,
												   bsrsm2Info info,
												   CUdeviceptr X,
												   int ldx,
												   CUdeviceptr Y,
												   int ldy,
												   cusparseSolvePolicy policy,
												   CUdeviceptr pBuffer);
		#endregion



		/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   based on the information in the opaque structure info that was obtained 
   from the analysis phase (csrsv_analysis). */
		
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsrilu0Ex(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m, 
                                              cusparseMatDescr descrA, 
                                              CUdeviceptr csrSortedValA_ValM, 
                                              cudaDataType csrSortedValA_ValMtype,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              CUdeviceptr csrSortedRowPtrA, 
                                              CUdeviceptr csrSortedColIndA,
                                              cusparseSolveAnalysisInfo info,
                                              cudaDataType executiontype);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrilu0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m, 
                                              cusparseMatDescr descrA, 
											  CUdeviceptr csrValM,
											  /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */                                              
                                              CUdeviceptr csrRowPtrA, 
                                              CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrilu0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m, 
                                              cusparseMatDescr descrA,
											  CUdeviceptr csrValM,
											/* matrix A values are updated inplace 
											   to be the preconditioner M values */ 
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrilu0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m, 
                                              cusparseMatDescr descrA,
											  CUdeviceptr csrValM, 
											/* matrix A values are updated inplace 
											   to be the preconditioner M values */ 
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrilu0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m, 
                                              cusparseMatDescr descrA,
											  CUdeviceptr csrValM, 
											/* matrix A values are updated inplace 
											   to be the preconditioner M values */ 
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);



		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            ref double tol,
                                                            ref float boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            ref double tol,
                                                            ref double boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            ref double tol,
                                                            ref cuFloatComplex boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            ref double tol,
                                                            ref cuDoubleComplex boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrilu02_zeroPivot(cusparseContext handle,
                                                         csrilu02Info info,
                                                         ref int position);





		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															ref double tol,
															ref float boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															ref double tol,
															ref double boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															ref double tol,
															ref cuFloatComplex boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															ref double tol,
															ref cuDoubleComplex boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsrilu02_zeroPivot(cusparseContext handle,
														 bsrilu02Info info,
														 ref int position);

		#endregion

		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            CUdeviceptr tol,
                                                            CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            CUdeviceptr tol,
                                                            CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            CUdeviceptr tol,
                                                            CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrilu02_numericBoost(cusparseContext handle,
                                                            csrilu02Info info,
                                                            int enable_boost,    
                                                            CUdeviceptr tol,
                                                            CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrilu02_zeroPivot(cusparseContext handle,
                                                         csrilu02Info info,
                                                         CUdeviceptr position);




		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															CUdeviceptr tol,
															CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															CUdeviceptr tol,
															CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															CUdeviceptr tol,
															CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrilu02_numericBoost(cusparseContext handle,
															bsrilu02Info info,
															int enable_boost,
															CUdeviceptr tol,
															CUdeviceptr boost_val);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsrilu02_zeroPivot(cusparseContext handle,
														 bsrilu02Info info,
														 CUdeviceptr position);

		#endregion


		
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrilu02_bufferSize(cusparseContext handle,
                                                          int m,
                                                          int nnz,
                                                          cusparseMatDescr descrA,
														  CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
                                                          csrilu02Info info,
                                                          ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrilu02_bufferSize(cusparseContext handle,
                                                          int m,
                                                          int nnz,
                                                          cusparseMatDescr descrA,
														  CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
                                                          csrilu02Info info,
                                                          ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrilu02_bufferSize(cusparseContext handle,
                                                          int m,
                                                          int nnz,
                                                          cusparseMatDescr descrA,
														  CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
                                                          csrilu02Info info,
                                                          ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrilu02_bufferSize(cusparseContext handle,
                                                          int m,
                                                          int nnz,
                                                          cusparseMatDescr descrA,
                                                          CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
                                                          csrilu02Info info,
														  ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrilu02_bufferSizeExt(cusparseContext handle,
														  int m,
														  int nnz,
														  cusparseMatDescr descrA,
														  CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
														  csrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrilu02_bufferSizeExt(cusparseContext handle,
														  int m,
														  int nnz,
														  cusparseMatDescr descrA,
														  CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
														  csrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrilu02_bufferSizeExt(cusparseContext handle,
														  int m,
														  int nnz,
														  cusparseMatDescr descrA,
														  CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
														  csrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrilu02_bufferSizeExt(cusparseContext handle,
														  int m,
														  int nnz,
														  cusparseMatDescr descrA,
														  CUdeviceptr csrValA,
														  CUdeviceptr csrRowPtrA,
														  CUdeviceptr csrColIndA,
														  csrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrilu02_analysis(cusparseContext handle,
                                                        int m,
                                                        int nnz,
                                                        cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
                                                        csrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrilu02_analysis(cusparseContext handle,
                                                        int m,
                                                        int nnz,
                                                        cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
                                                        csrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrilu02_analysis(cusparseContext handle,
                                                        int m,
                                                        int nnz,
                                                        cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
                                                        csrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrilu02_analysis(cusparseContext handle,
                                                        int m,
                                                        int nnz,
                                                        cusparseMatDescr descrA,
														CUdeviceptr csrValA,
														CUdeviceptr csrRowPtrA,
														CUdeviceptr csrColIndA,
                                                        csrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrilu02(cusparseContext handle,
                                               int m,
                                               int nnz,
                                               cusparseMatDescr descrA,
											   CUdeviceptr csrValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
											   CUdeviceptr csrRowPtrA,
											   CUdeviceptr csrColIndA,
                                               csrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrilu02(cusparseContext handle,
                                               int m,
                                               int nnz,
                                               cusparseMatDescr descrA,
											   CUdeviceptr csrValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
											   CUdeviceptr csrRowPtrA,
											   CUdeviceptr csrColIndA,
                                               csrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrilu02(cusparseContext handle,
                                               int m,
                                               int nnz,
                                               cusparseMatDescr descrA,
											   CUdeviceptr csrValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
											   CUdeviceptr csrRowPtrA,
											   CUdeviceptr csrColIndA,
                                               csrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrilu02(cusparseContext handle,
                                               int m,
                                               int nnz,
                                               cusparseMatDescr descrA,
											   CUdeviceptr csrValA_valM,
                                               /* matrix A values are updated inplace 
                                                  to be the preconditioner M values */
											   CUdeviceptr csrRowPtrA,
											   CUdeviceptr csrColIndA,
                                               csrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);





		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrilu02_bufferSize(cusparseContext handle,
                                                          cusparseDirection dirA,
                                                          int mb,
                                                          int nnzb,
                                                          cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
                                                          int blockDim,
                                                          bsrilu02Info info,
                                                          ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrilu02_bufferSize(cusparseContext handle,
                                                          cusparseDirection dirA,
                                                          int mb,
                                                          int nnzb,
                                                          cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
                                                          int blockDim,
                                                          bsrilu02Info info,
                                                          ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrilu02_bufferSize(cusparseContext handle,
                                                          cusparseDirection dirA,
                                                          int mb,
                                                          int nnzb,
                                                          cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
                                                          int blockDim,
                                                          bsrilu02Info info,
                                                          ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrilu02_bufferSize(cusparseContext handle,
                                                          cusparseDirection dirA,
                                                          int mb,
                                                          int nnzb,
                                                          cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
                                                          int blockDim,
                                                          bsrilu02Info info,
														  ref int pBufferSize);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrilu02_bufferSizeExt(cusparseContext handle,
														  cusparseDirection dirA,
														  int mb,
														  int nnzb,
														  cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
														  int blockDim,
														  bsrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrilu02_bufferSizeExt(cusparseContext handle,
														  cusparseDirection dirA,
														  int mb,
														  int nnzb,
														  cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
														  int blockDim,
														  bsrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrilu02_bufferSizeExt(cusparseContext handle,
														  cusparseDirection dirA,
														  int mb,
														  int nnzb,
														  cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
														  int blockDim,
														  bsrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrilu02_bufferSizeExt(cusparseContext handle,
														  cusparseDirection dirA,
														  int mb,
														  int nnzb,
														  cusparseMatDescr descrA,
														  CUdeviceptr bsrVal,
														  CUdeviceptr bsrRowPtr,
														  CUdeviceptr bsrColInd,
														  int blockDim,
														  bsrilu02Info info,
														  ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrilu02_analysis(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        int mb,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockDim,
                                                        bsrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrilu02_analysis(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        int mb,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockDim,
                                                        bsrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrilu02_analysis(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        int mb,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockDim,
                                                        bsrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrilu02_analysis(cusparseContext handle,
                                                        cusparseDirection dirA,
                                                        int mb,
                                                        int nnzb,
                                                        cusparseMatDescr descrA,
														CUdeviceptr bsrVal,
														CUdeviceptr bsrRowPtr,
														CUdeviceptr bsrColInd,
                                                        int blockDim,
                                                        bsrilu02Info info,
                                                        cusparseSolvePolicy policy,
                                                        CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrilu02(cusparseContext handle,
                                               cusparseDirection dirA,
                                               int mb,
                                               int nnzb,
                                               cusparseMatDescr descra,
											   CUdeviceptr bsrVal,
											   CUdeviceptr bsrRowPtr,
											   CUdeviceptr bsrColInd,
                                               int blockDim,
                                               bsrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrilu02(cusparseContext handle,
                                               cusparseDirection dirA,
                                               int mb,
                                               int nnzb,
                                               cusparseMatDescr descra,
											   CUdeviceptr bsrVal,
											   CUdeviceptr bsrRowPtr,
											   CUdeviceptr bsrColInd,
                                               int blockDim,
                                               bsrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrilu02(cusparseContext handle,
                                               cusparseDirection dirA,
                                               int mb,
                                               int nnzb,
                                               cusparseMatDescr descra,
											   CUdeviceptr bsrVal,
											   CUdeviceptr bsrRowPtr,
											   CUdeviceptr bsrColInd,
                                               int blockDim,
                                               bsrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrilu02(cusparseContext handle,
                                               cusparseDirection dirA,
                                               int mb,
                                               int nnzb,
                                               cusparseMatDescr descra,
											   CUdeviceptr bsrVal,
											   CUdeviceptr bsrRowPtr,
											   CUdeviceptr bsrColInd,
                                               int blockDim,
                                               bsrilu02Info info,
                                               cusparseSolvePolicy policy,
                                               CUdeviceptr pBuffer);






		/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   based on the information in the opaque structure info that was obtained 
   from the analysis phase (csrsv_analysis). */

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsric0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m, 
                                              cusparseMatDescr descrA,
											  CUdeviceptr csrValA_ValM,
											/* matrix A values are updated inplace 
											   to be the preconditioner M values */ 
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsric0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m,
											  cusparseMatDescr descrA,
											  CUdeviceptr csrValA_ValM,
											/* matrix A values are updated inplace 
											   to be the preconditioner M values */
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsric0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m,
											  cusparseMatDescr descrA,
											  CUdeviceptr csrValA_ValM,
											/* matrix A values are updated inplace 
											   to be the preconditioner M values */
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsric0(cusparseContext handle, 
                                              cusparseOperation trans, 
                                              int m,
											  cusparseMatDescr descrA,
											  CUdeviceptr csrValA_ValM,
											/* matrix A values are updated inplace 
											   to be the preconditioner M values */
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
                                              cusparseSolveAnalysisInfo info);



		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsric02_zeroPivot(cusparseContext handle,
														csric02Info info,
														ref int position);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsric02_zeroPivot(cusparseContext handle,
														bsric02Info info,
														ref int position);
		#endregion

		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsric02_zeroPivot(cusparseContext handle,
														csric02Info info,
														CUdeviceptr position);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXbsric02_zeroPivot(cusparseContext handle,
														bsric02Info info,
														CUdeviceptr position);
		#endregion

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsric02_bufferSize(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsric02_bufferSize(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsric02_bufferSize(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsric02_bufferSize(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsric02_bufferSizeExt(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsric02_bufferSizeExt(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsric02_bufferSizeExt(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsric02_bufferSizeExt(cusparseContext handle,
														 int m,
														 int nnz,
														 cusparseMatDescr descrA,
														 CUdeviceptr csrValA,
														 CUdeviceptr csrRowPtrA,
														 CUdeviceptr csrColIndA,
														 csric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsric02_analysis(cusparseContext handle,
													   int m,
													   int nnz,
													   cusparseMatDescr descrA,
													   CUdeviceptr csrValA,
													   CUdeviceptr csrRowPtrA,
													   CUdeviceptr csrColIndA,
													   csric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsric02_analysis(cusparseContext handle,
													   int m,
													   int nnz,
													   cusparseMatDescr descrA,
													   CUdeviceptr csrValA,
													   CUdeviceptr csrRowPtrA,
													   CUdeviceptr csrColIndA,
													   csric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsric02_analysis(cusparseContext handle,
													   int m,
													   int nnz,
													   cusparseMatDescr descrA,
													   CUdeviceptr csrValA,
													   CUdeviceptr csrRowPtrA,
													   CUdeviceptr csrColIndA,
													   csric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsric02_analysis(cusparseContext handle,
													   int m,
													   int nnz,
													   cusparseMatDescr descrA,
													   CUdeviceptr csrValA,
													   CUdeviceptr csrRowPtrA,
													   CUdeviceptr csrColIndA,
													   csric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsric02(cusparseContext handle,
											  int m,
											  int nnz,
											  cusparseMatDescr descrA,
											  CUdeviceptr csrValA_valM,
			/* matrix A values are updated inplace 
			   to be the preconditioner M values */
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  csric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsric02(cusparseContext handle,
											  int m,
											  int nnz,
											  cusparseMatDescr descrA,
											  CUdeviceptr csrValA_valM,
			/* matrix A values are updated inplace 
			   to be the preconditioner M values */
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  csric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsric02(cusparseContext handle,
											  int m,
											  int nnz,
											  cusparseMatDescr descrA,
											  CUdeviceptr csrValA_valM,
			/* matrix A values are updated inplace 
			   to be the preconditioner M values */
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  csric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsric02(cusparseContext handle,
											  int m,
											  int nnz,
											  cusparseMatDescr descrA,
											  CUdeviceptr csrValA_valM,
			/* matrix A values are updated inplace 
			   to be the preconditioner M values */
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  csric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);



		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsric02_bufferSize(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsric02_bufferSize(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsric02_bufferSize(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsric02_bufferSize(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref int pBufferSize);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsric02_bufferSizeExt(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsric02_bufferSizeExt(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsric02_bufferSizeExt(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsric02_bufferSizeExt(cusparseContext handle,
														 cusparseDirection dirA,
														 int mb,
														 int nnzb,
														 cusparseMatDescr descrA,
														 CUdeviceptr bsrVal,
														 CUdeviceptr bsrRowPtr,
														 CUdeviceptr bsrColInd,
														 int blockDim,
														 bsric02Info info,
														 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsric02_analysis(cusparseContext handle,
													   cusparseDirection dirA,
													   int mb,
													   int nnzb,
													   cusparseMatDescr descrA,
													   CUdeviceptr bsrVal,
													   CUdeviceptr bsrRowPtr,
													   CUdeviceptr bsrColInd,
													   int blockDim,
													   bsric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pInputBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsric02_analysis(cusparseContext handle,
													   cusparseDirection dirA,
													   int mb,
													   int nnzb,
													   cusparseMatDescr descrA,
													   CUdeviceptr bsrVal,
													   CUdeviceptr bsrRowPtr,
													   CUdeviceptr bsrColInd,
													   int blockDim,
													   bsric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pInputBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsric02_analysis(cusparseContext handle,
													   cusparseDirection dirA,
													   int mb,
													   int nnzb,
													   cusparseMatDescr descrA,
													   CUdeviceptr bsrVal,
													   CUdeviceptr bsrRowPtr,
													   CUdeviceptr bsrColInd,
													   int blockDim,
													   bsric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pInputBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsric02_analysis(cusparseContext handle,
													   cusparseDirection dirA,
													   int mb,
													   int nnzb,
													   cusparseMatDescr descrA,
													   CUdeviceptr bsrVal,
													   CUdeviceptr bsrRowPtr,
													   CUdeviceptr bsrColInd,
													   int blockDim,
													   bsric02Info info,
													   cusparseSolvePolicy policy,
													   CUdeviceptr pInputBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsric02(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nnzb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int blockDim,
											  bsric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsric02(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nnzb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int blockDim,
											  bsric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsric02(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nnzb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int blockDim,
											  bsric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsric02(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nnzb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int blockDim,
											  bsric02Info info,
											  cusparseSolvePolicy policy,
											  CUdeviceptr pBuffer);










		/* Description: Solution of tridiagonal linear system A * B = B, 
		   with multiple right-hand-sides. The coefficient matrix A is 
		   composed of lower (dl), main (d) and upper (du) diagonals, and 
		   the right-hand-sides B are overwritten with the solution. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgtsv(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgtsv(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgtsv(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgtsv(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsv2_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsv2_bufferSizeExt( cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsv2_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsv2_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsv2(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsv2(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer);
        
        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsv2(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsv2(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer);

        /* Description: Solution of tridiagonal linear system A * B = B, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (dl), main (d) and upper (du) diagonals, and 
   the right-hand-sides B are overwritten with the solution. 
   These routines do not use pivoting, using a combination of PCR and CR algorithm */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgtsv_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);
                                 
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgtsv_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);
                                                                 
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgtsv_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgtsv_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsv2_nopivot_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsv2_nopivot_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes); 

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsv2_nopivot_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsv2_nopivot_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes); 

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsv2_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer); 

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsv2_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer); 

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsv2_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsv2_nopivot(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, CUdeviceptr pBuffer); 

        /* Description: Solution of a set of tridiagonal linear systems 
		   A * x = x, each with a single right-hand-side. The coefficient 
		   matrices A are composed of lower (dl), main (d) and upper (du) 
		   diagonals and stored separated by a batchStride, while the 
		   right-hand-sides x are also separated by a batchStride. */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgtsvStridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgtsvStridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgtsvStridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgtsvStridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride);



        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, ref SizeT bufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, ref SizeT bufferSizeInBytes); 

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, ref SizeT bufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, ref SizeT bufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsv2StridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, CUdeviceptr pBuffer); 

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsv2StridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsv2StridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsv2StridedBatch(cusparseContext handle, int m, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr x, int batchCount, int batchStride, CUdeviceptr pBuffer);


        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);


        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgtsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgtsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);


        /* Description: Solution of pentadiagonal linear system A * X = B, 
           with multiple right-hand-sides. The coefficient matrix A is 
           composed of lower (ds, dl), main (d) and upper (du, dw) diagonals, and 
           the right-hand-sides B are overwritten with the solution X. 
         */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgpsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgpsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgpsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgpsvInterleavedBatch_bufferSizeExt(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgpsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgpsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCgpsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgpsvInterleavedBatch(
            cusparseContext handle,
            int algo,
            int m,
            CUdeviceptr ds,
            CUdeviceptr dl,
            CUdeviceptr d,
            CUdeviceptr du,
            CUdeviceptr dw,
            CUdeviceptr x,
            int batchCount,
            CUdeviceptr pBuffer);



        /* --- Sparse Format Conversion --- */


        /* Description: This routine finds the total number of non-zero elements and 
		   the number of non-zero elements per row in a noncompressed csr matrix A. */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSnnz_compress(cusparseContext handle, 
                                          int m, 
                                          cusparseMatDescr descr,
                                          CUdeviceptr values, 
                                          CUdeviceptr rowPtr, 
                                          CUdeviceptr nnzPerRow, 
                                          CUdeviceptr nnzTotal,
                                          float tol);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDnnz_compress(cusparseContext handle, 
                                          int m, 
                                          cusparseMatDescr descr,
                                          CUdeviceptr values, 
                                          CUdeviceptr rowPtr, 
                                          CUdeviceptr nnzPerRow, 
                                          CUdeviceptr nnzTotal,
                                          double tol);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCnnz_compress(cusparseContext handle, 
                                          int m, 
                                          cusparseMatDescr descr,
                                          CUdeviceptr values, 
                                          CUdeviceptr rowPtr, 
                                          CUdeviceptr nnzPerRow, 
                                          CUdeviceptr nnzTotal,
                                          cuFloatComplex tol);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZnnz_compress(cusparseContext handle, 
                                          int m, 
                                          cusparseMatDescr descr,
                                          CUdeviceptr values, 
                                          CUdeviceptr rowPtr, 
                                          CUdeviceptr nnzPerRow, 
                                          CUdeviceptr nnzTotal,
                                          cuDoubleComplex tol);
		/* Description: This routine takes as input a csr form where the values may have 0 elements
		   and compresses it to return a csr form with no zeros. */

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2csr_compress(cusparseContext handle,
                                                      int m, 
                                                      int n,
                                                      cusparseMatDescr descra,
                                                      CUdeviceptr inVal,
                                                      CUdeviceptr inColInd,
													  CUdeviceptr inRowPtr, 
                                                      int inNnz,
                                                      CUdeviceptr nnzPerRow, 
                                                      CUdeviceptr outVal,
                                                      CUdeviceptr outColInd,
                                                      CUdeviceptr outRowPtr,
                                                      float tol);        

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2csr_compress(cusparseContext handle,
                                                      int m, //number of rows
                                                      int n,
                                                      cusparseMatDescr descra,
                                                      CUdeviceptr inVal, //csr values array-the elements which are below a certain tolerance will be remvoed
                                                      CUdeviceptr inColInd,
                                                      CUdeviceptr inRowPtr,  //corresponding input noncompressed row pointer
                                                      int inNnz,
                                                      CUdeviceptr nnzPerRow, //output: returns number of nonzeros per row 
                                                      CUdeviceptr outVal,
                                                      CUdeviceptr outColInd,
                                                      CUdeviceptr outRowPtr,
                                                      double tol);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2csr_compress(cusparseContext handle,
                                                        int m, //number of rows
                                                        int n,
                                                        cusparseMatDescr descra,
                                                        CUdeviceptr inVal, //csr values array-the elements which are below a certain tolerance will be remvoed
                                                        CUdeviceptr inColInd,
														CUdeviceptr inRowPtr,  //corresponding input noncompressed row pointer
                                                        int inNnz,
                                                        CUdeviceptr nnzPerRow, //output: returns number of nonzeros per row 
														CUdeviceptr outVal,
                                                        CUdeviceptr outColInd,
                                                        CUdeviceptr outRowPtr,
                                                        cuFloatComplex tol);                       

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2csr_compress(cusparseContext handle,
                                                      int m, //number of rows
                                                      int n,
                                                      cusparseMatDescr descra,
                                                      CUdeviceptr inVal, //csr values array-the elements which are below a certain tolerance will be remvoed
                                                      CUdeviceptr inColInd,
													  CUdeviceptr inRowPtr,  //corresponding input noncompressed row pointer
                                                      int inNnz,
                                                      CUdeviceptr nnzPerRow, //output: returns number of nonzeros per row 
                                                      CUdeviceptr outVal,
                                                      CUdeviceptr outColInd,
                                                      CUdeviceptr outRowPtr,
                                                      cuDoubleComplex tol);                        


		/* Description: This routine finds the total number of non-zero elements and 
		   the number of non-zero elements per row or column in the dense matrix A. */
		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, ref int nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, ref int nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, ref int nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, ref int nnzTotalDevHostPtr);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, CUdeviceptr nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, CUdeviceptr nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, CUdeviceptr nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZnnz(cusparseContext handle, cusparseDirection dirA, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRowCol, CUdeviceptr nnzTotalDevHostPtr);
		#endregion

		/* Description: This routine converts a dense matrix to a sparse matrix 
		   in the CSR storage format, using the information computed by the 
		   nnz routine. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSdense2csr(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDdense2csr(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCdense2csr(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZdense2csr(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA);

		/* Description: This routine converts a sparse matrix in CSR storage format
		   to a dense matrix. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, CUdeviceptr A, int lda);

		/* Description: This routine converts a dense matrix to a sparse matrix 
		   in the CSC storage format, using the information computed by the 
		   nnz routine. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSdense2csc(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerCol, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDdense2csc(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerCol, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCdense2csc(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerCol, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZdense2csc(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerCol, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA);

		/* Description: This routine converts a sparse matrix in CSC storage format
		   to a dense matrix. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsc2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsc2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsc2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsc2dense(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr cscValA, CUdeviceptr cscRowIndA, CUdeviceptr cscColPtrA, CUdeviceptr A, int lda);

		/* Description: This routine compresses the indecis of rows or columns.
		   It can be interpreted as a conversion from COO to CSR sparse storage
		   format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcoo2csr(cusparseContext handle, CUdeviceptr cooRowInd, int nnz, int m, CUdeviceptr csrRowPtr, cusparseIndexBase idxBase);

		/* Description: This routine uncompresses the indecis of rows or columns.
		   It can be interpreted as a conversion from CSR to COO sparse storage
		   format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsr2coo(cusparseContext handle, CUdeviceptr csrRowPtr, int nnz, int m, CUdeviceptr cooRowInd, cusparseIndexBase idxBase);

		/* Description: This routine converts a matrix from CSR to CSC sparse 
		   storage format. The resulting matrix can be re-interpreted as a 
		   transpose of the original matrix in CSR storage format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsr2cscEx(cusparseContext handle,
                                              int m, 
                                              int n, 
                                              int nnz,
                                              CUdeviceptr csrSortedVal, 
                                              cudaDataType csrSortedValtype,
                                              CUdeviceptr csrSortedRowPtr, 
                                              CUdeviceptr csrSortedColInd, 
                                              CUdeviceptr cscSortedVal, 
                                              cudaDataType cscSortedValtype,
                                              CUdeviceptr cscSortedRowInd, 
                                              CUdeviceptr cscSortedColPtr, 
                                              cusparseAction copyValues, 
                                              cusparseIndexBase idxBase,
                                              cudaDataType executiontype);
    
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2csc(cusparseContext handle, int m, int n, int nnz, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr cscVal, CUdeviceptr cscRowInd, CUdeviceptr cscColPtr, cusparseAction copyValues, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2csc(cusparseContext handle, int m, int n, int nnz, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr cscVal, CUdeviceptr cscRowInd, CUdeviceptr cscColPtr, cusparseAction copyValues, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2csc(cusparseContext handle, int m, int n, int nnz, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr cscVal, CUdeviceptr cscRowInd, CUdeviceptr cscColPtr, cusparseAction copyValues, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2csc(cusparseContext handle, int m, int n, int nnz, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd, CUdeviceptr cscVal, CUdeviceptr cscRowInd, CUdeviceptr cscColPtr, cusparseAction copyValues, cusparseIndexBase idxBase);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsr2cscEx2(cusparseContext handle,
				   int m, int n, int nnz,
				   CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd,
				   CUdeviceptr cscVal, CUdeviceptr cscColPtr, CUdeviceptr cscRowInd,
				   cudaDataType valType,
                   cusparseAction copyValues,
				   cusparseIndexBase  idxBase,
                   cusparseCsr2CscAlg alg,
				   CUdeviceptr buffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCsr2cscEx2_bufferSize(cusparseContext handle,
							  int m, int n, int nnz, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd,
							  CUdeviceptr cscVal, CUdeviceptr cscColPtr, CUdeviceptr cscRowInd, cudaDataType valType,
                              cusparseAction copyValues, cusparseIndexBase  idxBase, cusparseCsr2CscAlg alg, ref SizeT bufferSize);

		/* Description: This routine converts a dense matrix to a sparse matrix 
		   in HYB storage format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSdense2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDdense2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCdense2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZdense2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr A, int lda, CUdeviceptr nnzPerRow, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		/* Description: This routine converts a sparse matrix in HYB storage format
		   to a dense matrix. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShyb2dense(cusparseContext handle, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhyb2dense(cusparseContext handle, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChyb2dense(cusparseContext handle, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr A, int lda);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhyb2dense(cusparseContext handle, cusparseMatDescr descrA, cusparseHybMat hybA, CUdeviceptr A, int lda);

		/* Description: This routine converts a sparse matrix in CSR storage format
		   to a sparse matrix in HYB storage format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2hyb(cusparseContext handle, int m, int n, cusparseMatDescr descrA, CUdeviceptr csrValA, CUdeviceptr csrRowPtrA, CUdeviceptr csrColIndA, cusparseHybMat hybA, int userEllWidth, cusparseHybPartition partitionType);

		#endregion

		#region Sparse Level 4 routines

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrgemmNnz(cusparseContext handle,
												  cusparseOperation transA,
												  cusparseOperation transB,
												  int m,
												  int n,
												  int k,
												  cusparseMatDescr descrA,
												  int nnzA,
												  CUdeviceptr csrRowPtrA,
												  CUdeviceptr csrColIndA,
												  cusparseMatDescr descrB,
												  int nnzB,
												  CUdeviceptr csrRowPtrB,
												  CUdeviceptr csrColIndB,
												  cusparseMatDescr descrC,
												  CUdeviceptr csrRowPtrC,
												  ref int nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrgemmNnz(cusparseContext handle,
												  cusparseOperation transA,
												  cusparseOperation transB,
												  int m,
												  int n,
												  int k,
												  cusparseMatDescr descrA,
												  int nnzA,
												  CUdeviceptr csrRowPtrA,
												  CUdeviceptr csrColIndA,
												  cusparseMatDescr descrB,
												  int nnzB,
												  CUdeviceptr csrRowPtrB,
												  CUdeviceptr csrColIndB,
												  cusparseMatDescr descrC,
												  CUdeviceptr csrRowPtrC,
												  CUdeviceptr nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrgemm(cusparseContext handle,
											  cusparseOperation transA,
											  cusparseOperation transB,
											  int m,
											  int n,
											  int k,
											  cusparseMatDescr descrA,
											  int nnzA,
											  CUdeviceptr csrValA,
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  cusparseMatDescr descrB,
											  int nnzB,
											  CUdeviceptr csrValB,
											  CUdeviceptr csrRowPtrB,
											  CUdeviceptr csrColIndB,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);



		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrgemm(cusparseContext handle,
											  cusparseOperation transA,
											  cusparseOperation transB,
											  int m,
											  int n,
											  int k,
											  cusparseMatDescr descrA,
											  int nnzA,
											  CUdeviceptr csrValA,
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  cusparseMatDescr descrB,
											  int nnzB,
											  CUdeviceptr csrValB,
											  CUdeviceptr csrRowPtrB,
											  CUdeviceptr csrColIndB,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrgemm(cusparseContext handle,
											  cusparseOperation transA,
											  cusparseOperation transB,
											  int m,
											  int n,
											  int k,
											  cusparseMatDescr descrA,
											  int nnzA,
											  CUdeviceptr csrValA,
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  cusparseMatDescr descrB,
											  int nnzB,
											  CUdeviceptr csrValB,
											  CUdeviceptr csrRowPtrB,
											  CUdeviceptr csrColIndB,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrgemm(cusparseContext handle,
											  cusparseOperation transA,
											  cusparseOperation transB,
											  int m,
											  int n,
											  int k,
											  cusparseMatDescr descrA,
											  int nnzA,
											  CUdeviceptr csrValA,
											  CUdeviceptr csrRowPtrA,
											  CUdeviceptr csrColIndA,
											  cusparseMatDescr descrB,
											  int nnzB,
											  CUdeviceptr csrValB,
											  CUdeviceptr csrRowPtrB,
											  CUdeviceptr csrColIndB,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);


		/* Description: Compute sparse - sparse matrix multiplication for matrices 
   stored in CSR format. */

		#region ref host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 ref float alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 ref float beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 ref double alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 ref double beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 ref cuFloatComplex alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 ref cuFloatComplex beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 ref cuDoubleComplex alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 ref cuDoubleComplex beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);



		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   ref float alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   ref float beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   ref double alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   ref double beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   ref cuFloatComplex alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   ref cuFloatComplex beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   ref cuDoubleComplex alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   ref cuDoubleComplex beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrgemm2Nnz(cusparseContext handle,
												  int m,
												  int n,
												  int k,
												  cusparseMatDescr descrA,
												  int nnzA,
												  CUdeviceptr csrSortedRowPtrA,
												  CUdeviceptr csrSortedColIndA,
												  cusparseMatDescr descrB,
												  int nnzB,
												  CUdeviceptr csrSortedRowPtrB,
												  CUdeviceptr csrSortedColIndB,
												  cusparseMatDescr descrD,
												  int nnzD,
												  CUdeviceptr csrSortedRowPtrD,
												  CUdeviceptr csrSortedColIndD,
												  cusparseMatDescr descrC,
												  CUdeviceptr csrSortedRowPtrC,
												  ref int nnzTotalDevHostPtr,
												  csrgemm2Info info,
												  CUdeviceptr pBuffer);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 CUdeviceptr alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 CUdeviceptr beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 CUdeviceptr alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 CUdeviceptr beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 CUdeviceptr alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 CUdeviceptr beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrgemm2_bufferSizeExt(cusparseContext handle,
															 int m,
															 int n,
															 int k,
															 CUdeviceptr alpha,
															 cusparseMatDescr descrA,
															 int nnzA,
															 CUdeviceptr csrSortedRowPtrA,
															 CUdeviceptr csrSortedColIndA,
															 cusparseMatDescr descrB,
															 int nnzB,
															 CUdeviceptr csrSortedRowPtrB,
															 CUdeviceptr csrSortedColIndB,
															 CUdeviceptr beta,
															 cusparseMatDescr descrD,
															 int nnzD,
															 CUdeviceptr csrSortedRowPtrD,
															 CUdeviceptr csrSortedColIndD,
															 csrgemm2Info info,
															 ref SizeT pBufferSizeInBytes);



		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   CUdeviceptr alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   CUdeviceptr beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   CUdeviceptr alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   CUdeviceptr beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   CUdeviceptr alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   CUdeviceptr beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrgemm2(cusparseContext handle,
											   int m,
											   int n,
											   int k,
											   CUdeviceptr alpha,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrSortedValA,
											   CUdeviceptr csrSortedRowPtrA,
											   CUdeviceptr csrSortedColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrSortedValB,
											   CUdeviceptr csrSortedRowPtrB,
											   CUdeviceptr csrSortedColIndB,
											   CUdeviceptr beta,
											   cusparseMatDescr descrD,
											   int nnzD,
											   CUdeviceptr csrSortedValD,
											   CUdeviceptr csrSortedRowPtrD,
											   CUdeviceptr csrSortedColIndD,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrSortedValC,
											   CUdeviceptr csrSortedRowPtrC,
											   CUdeviceptr csrSortedColIndC,
											   csrgemm2Info info,
											   CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrgemm2Nnz(cusparseContext handle,
												  int m,
												  int n,
												  int k,
												  cusparseMatDescr descrA,
												  int nnzA,
												  CUdeviceptr csrSortedRowPtrA,
												  CUdeviceptr csrSortedColIndA,
												  cusparseMatDescr descrB,
												  int nnzB,
												  CUdeviceptr csrSortedRowPtrB,
												  CUdeviceptr csrSortedColIndB,
												  cusparseMatDescr descrD,
												  int nnzD,
												  CUdeviceptr csrSortedRowPtrD,
												  CUdeviceptr csrSortedColIndD,
												  cusparseMatDescr descrC,
												  CUdeviceptr csrSortedRowPtrC,
												  CUdeviceptr nnzTotalDevHostPtr,
												  csrgemm2Info info,
												  CUdeviceptr pBuffer);
		#endregion







		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrgeamNnz(cusparseContext handle,
											   int m,
											   int n,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrRowPtrA,
											   CUdeviceptr csrColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrRowPtrB,
											   CUdeviceptr csrColIndB,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrRowPtrC,
											   ref int nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrgeamNnz(cusparseContext handle,
											   int m,
											   int n,
											   cusparseMatDescr descrA,
											   int nnzA,
											   CUdeviceptr csrRowPtrA,
											   CUdeviceptr csrColIndA,
											   cusparseMatDescr descrB,
											   int nnzB,
											   CUdeviceptr csrRowPtrB,
											   CUdeviceptr csrColIndB,
											   cusparseMatDescr descrC,
											   CUdeviceptr csrRowPtrC,
											   CUdeviceptr nnzTotalDevHostPtr);

		#region host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrgeam(cusparseContext handle,
											int m,
											int n,
											ref float alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											ref float beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrgeam(cusparseContext handle,
											int m,
											int n,
											ref double alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											ref double beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrgeam(cusparseContext handle,
											int m,
											int n,
											ref cuFloatComplex alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											ref cuFloatComplex beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrgeam(cusparseContext handle,
											int m,
											int n,
											ref cuDoubleComplex alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											ref cuDoubleComplex beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);
		#endregion
		#region device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrgeam(cusparseContext handle,
											int m,
											int n,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											CUdeviceptr beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrgeam(cusparseContext handle,
											int m,
											int n,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											CUdeviceptr beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrgeam(cusparseContext handle,
											int m,
											int n,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											CUdeviceptr beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrgeam(cusparseContext handle,
											int m,
											int n,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											int nnzA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											CUdeviceptr beta,
											cusparseMatDescr descrB,
											int nnzB,
											CUdeviceptr csrValB,
											CUdeviceptr csrRowPtrB,
											CUdeviceptr csrColIndB,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);
        #endregion





        #region host

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseXcsrgeam2Nnz(
            cusparseContext handle,
            int m,
            int n,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedRowPtrC,
            ref int nnzTotalDevHostPtr,
            CUdeviceptr workspace);
        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            ref float alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref float beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes );

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            ref double alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref double beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            ref cuFloatComplex alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref cuFloatComplex beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            ref cuDoubleComplex alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref cuDoubleComplex beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            ref float alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref float beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer );

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            ref double alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref double beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer );

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            ref cuFloatComplex alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref cuFloatComplex beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer );

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            ref cuDoubleComplex alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            ref cuDoubleComplex beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer );
        #endregion

        #region device

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseXcsrgeam2Nnz(
            cusparseContext handle,
            int m,
            int n,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr nnzTotalDevHostPtr,
            CUdeviceptr workspace);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrgeam2_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            ref SizeT pBufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDcsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCcsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZcsrgeam2(
            cusparseContext handle,
            int m,
            int n,
            CUdeviceptr alpha,
            cusparseMatDescr descrA,
            int nnzA,
            CUdeviceptr csrSortedValA,
            CUdeviceptr csrSortedRowPtrA,
            CUdeviceptr csrSortedColIndA,
            CUdeviceptr beta,
            cusparseMatDescr descrB,
            int nnzB,
            CUdeviceptr csrSortedValB,
            CUdeviceptr csrSortedRowPtrB,
            CUdeviceptr csrSortedColIndB,
            cusparseMatDescr descrC,
            CUdeviceptr csrSortedValC,
            CUdeviceptr csrSortedRowPtrC,
            CUdeviceptr csrSortedColIndC,
            CUdeviceptr pBuffer);
        #endregion




        /* --- Sparse Matrix Reorderings --- */

        /* Description: Find an approximate coloring of a matrix stored in CSR format. */
        #region ref host
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               ref float fractionToColor,
											   ref int ncolors,
                                               CUdeviceptr coloring,
                                               CUdeviceptr reordering,
                                               cusparseColorInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               ref double fractionToColor,
											   ref int ncolors,
                                               CUdeviceptr coloring,
                                               CUdeviceptr reordering,
                                               cusparseColorInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               ref float fractionToColor,
											   ref int ncolors,
                                               CUdeviceptr coloring,
                                               CUdeviceptr reordering,
                                               cusparseColorInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               ref double fractionToColor,
											   ref int ncolors,
											   CUdeviceptr coloring,
											   CUdeviceptr reordering,
                                               cusparseColorInfo info);
		#endregion
		#region ref device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               CUdeviceptr fractionToColor,
											   CUdeviceptr ncolors,
                                               CUdeviceptr coloring,
                                               CUdeviceptr reordering,
                                               cusparseColorInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               CUdeviceptr fractionToColor,
											   CUdeviceptr ncolors,
                                               CUdeviceptr coloring,
                                               CUdeviceptr reordering,
                                               cusparseColorInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               CUdeviceptr fractionToColor,
											   CUdeviceptr ncolors,
                                               CUdeviceptr coloring,
                                               CUdeviceptr reordering,
                                               cusparseColorInfo info);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsrcolor(cusparseContext handle,
                                               int m, 
                                               int nnz,
                                               cusparseMatDescr descrA, 
                                               CUdeviceptr csrSortedValA, 
                                               CUdeviceptr csrSortedRowPtrA, 
                                               CUdeviceptr csrSortedColIndA,
                                               CUdeviceptr fractionToColor,
											   CUdeviceptr ncolors,
											   CUdeviceptr coloring,
											   CUdeviceptr reordering,
                                               cusparseColorInfo info);
		#endregion


		/* Description: This routine converts a sparse matrix in HYB storage format
		   to a sparse matrix in CSR storage format. */

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShyb2csr(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
                                              CUdeviceptr csrValA,
                                              CUdeviceptr csrRowPtrA,
                                              CUdeviceptr csrColIndA);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhyb2csr(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
                                              CUdeviceptr csrValA,
                                              CUdeviceptr csrRowPtrA,
                                              CUdeviceptr csrColIndA);              


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChyb2csr(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
                                              CUdeviceptr csrValA,
                                              CUdeviceptr csrRowPtrA,
                                              CUdeviceptr csrColIndA);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhyb2csr(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
                                              CUdeviceptr csrValA,
                                              CUdeviceptr csrRowPtrA,
                                              CUdeviceptr csrColIndA);


		/* Description: This routine converts a sparse matrix in CSC storage format
   to a sparse matrix in HYB storage format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsc2hyb(cusparseContext handle,
                                              int m,
                                              int n,
                                              cusparseMatDescr descrA,
                                              CUdeviceptr cscValA,
                                              CUdeviceptr cscRowIndA,
                                              CUdeviceptr cscColPtrA,
                                              cusparseHybMat hybA,
                                              int userEllWidth,
                                              cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsc2hyb(cusparseContext handle,
                                              int m,
                                              int n,
                                              cusparseMatDescr descrA,
                                              CUdeviceptr cscValA,
                                              CUdeviceptr cscRowIndA,
                                              CUdeviceptr cscColPtrA,
                                              cusparseHybMat hybA,
                                              int userEllWidth,
                                              cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsc2hyb(cusparseContext handle,
                                              int m,
                                              int n,
                                              cusparseMatDescr descrA,
                                              CUdeviceptr cscValA,
                                              CUdeviceptr cscRowIndA,
                                              CUdeviceptr cscColPtrA,
                                              cusparseHybMat hybA,
                                              int userEllWidth,
                                              cusparseHybPartition partitionType);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsc2hyb(cusparseContext handle,
                                              int m,
                                              int n,
                                              cusparseMatDescr descrA,
                                              CUdeviceptr cscValA,
                                              CUdeviceptr cscRowIndA,
                                              CUdeviceptr cscColPtrA,
                                              cusparseHybMat hybA,
                                              int userEllWidth,
                                              cusparseHybPartition partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a sparse matrix in CSC storage format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseShyb2csc(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
                                              CUdeviceptr cscVal,
											  CUdeviceptr cscRowInd,
											  CUdeviceptr cscColPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDhyb2csc(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
											  CUdeviceptr cscVal,
											  CUdeviceptr cscRowInd,
											  CUdeviceptr cscColPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseChyb2csc(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
											  CUdeviceptr cscVal,
											  CUdeviceptr cscRowInd,
											  CUdeviceptr cscColPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZhyb2csc(cusparseContext handle,
                                              cusparseMatDescr descrA,
                                              cusparseHybMat hybA,
											  CUdeviceptr cscVal,
											  CUdeviceptr cscRowInd,
											  CUdeviceptr cscColPtr);





		/* Description: This routine converts a sparse matrix in CSR storage format
		to a sparse matrix in BSR storage format. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsr2bsrNnz(cusparseContext handle,
											cusparseDirection dirA,
											int m,
											int n,
											cusparseMatDescr descrA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr bsrRowPtrC,
											CUdeviceptr nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsr2bsrNnz(cusparseContext handle,
											cusparseDirection dirA,
											int m,
											int n,
											cusparseMatDescr descrA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr bsrRowPtrC,
											ref int nnzTotalDevHostPtr);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2bsr(cusparseContext handle,
											cusparseDirection dirA,
											int m,
											int n,
											cusparseMatDescr descrA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr bsrValC,
											CUdeviceptr bsrRowPtrC,
											CUdeviceptr bsrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2bsr(cusparseContext handle,
											cusparseDirection dirA,
											int m,
											int n,
											cusparseMatDescr descrA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr bsrValC,
											CUdeviceptr bsrRowPtrC,
											CUdeviceptr bsrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2bsr(cusparseContext handle,
											cusparseDirection dirA,
											int m,
											int n,
											cusparseMatDescr descrA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr bsrValC,
											CUdeviceptr bsrRowPtrC,
											CUdeviceptr bsrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2bsr(cusparseContext handle,
											cusparseDirection dirA,
											int m,
											int n,
											cusparseMatDescr descrA,
											CUdeviceptr csrValA,
											CUdeviceptr csrRowPtrA,
											CUdeviceptr csrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr bsrValC,
											CUdeviceptr bsrRowPtrC,
											CUdeviceptr bsrColIndC);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsr2csr(cusparseContext handle,
											cusparseDirection dirA,
											int mb,
											int nb,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsr2csr(cusparseContext handle,
											cusparseDirection dirA,
											int mb,
											int nb,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsr2csr(cusparseContext handle,
											cusparseDirection dirA,
											int mb,
											int nb,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsr2csr(cusparseContext handle,
											cusparseDirection dirA,
											int mb,
											int nb,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											cusparseMatDescr descrC,
											CUdeviceptr csrValC,
											CUdeviceptr csrRowPtrC,
											CUdeviceptr csrColIndC);



		#region Removed in Cuda 5.5 production release

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseSgebsr2gebsc(cusparseContext handle,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      CUdeviceptr bsrVal,
		//                                      CUdeviceptr bsrRowPtr,
		//                                      CUdeviceptr bsrColInd,
		//                                      int rowBlockDim,
		//                                      int colBlockDim,
		//                                      CUdeviceptr bscVal,
		//                                      CUdeviceptr bscRowInd,
		//                                      CUdeviceptr bscColPtr,
		//                                      cusparseAction copyValues,
		//                                      cusparseIndexBase baseIdx);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseDgebsr2gebsc(cusparseContext handle,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      CUdeviceptr bsrVal,
		//                                      CUdeviceptr bsrRowPtr,
		//                                      CUdeviceptr bsrColInd,
		//                                      int rowBlockDim,
		//                                      int colBlockDim,
		//                                      CUdeviceptr bscVal,
		//                                      CUdeviceptr bscRowInd,
		//                                      CUdeviceptr bscColPtr,
		//                                      cusparseAction copyValues,
		//                                      cusparseIndexBase baseIdx);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseCgebsr2gebsc(cusparseContext handle,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      CUdeviceptr bsrVal,
		//                                      CUdeviceptr bsrRowPtr,
		//                                      CUdeviceptr bsrColInd,
		//                                      int rowBlockDim,
		//                                      int colBlockDim,
		//                                      CUdeviceptr bscVal,
		//                                      CUdeviceptr bscRowInd,
		//                                      CUdeviceptr bscColPtr,
		//                                      cusparseAction copyValues,
		//                                      cusparseIndexBase baseIdx);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseZgebsr2gebsc(cusparseContext handle,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      CUdeviceptr bsrVal,
		//                                      CUdeviceptr bsrRowPtr,
		//                                      CUdeviceptr bsrColInd,
		//                                      int rowBlockDim,
		//                                      int colBlockDim,
		//                                      CUdeviceptr bscVal,
		//                                      CUdeviceptr bscRowInd,
		//                                      CUdeviceptr bscColPtr,
		//                                      cusparseAction copyValues,
		//                                      cusparseIndexBase baseIdx);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseXgebsr2csr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int   rowBlockDim,
		//                                      int   colBlockDim,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr csrRowPtrC,
		//                                      CUdeviceptr csrColIndC );

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseSgebsr2csr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int   rowBlockDim,
		//                                      int   colBlockDim,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr csrValC,
		//                                      CUdeviceptr csrRowPtrC,
		//                                      CUdeviceptr csrColIndC );


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseDgebsr2csr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int   rowBlockDim,
		//                                      int   colBlockDim,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr csrValC,
		//                                      CUdeviceptr csrRowPtrC,
		//                                      CUdeviceptr csrColIndC );


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseCgebsr2csr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int   rowBlockDim,
		//                                      int   colBlockDim,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr csrValC,
		//                                      CUdeviceptr csrRowPtrC,
		//                                      CUdeviceptr csrColIndC );


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseZgebsr2csr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int   rowBlockDim,
		//                                      int   colBlockDim,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr csrValC,
		//                                      CUdeviceptr csrRowPtrC,
		//                                      CUdeviceptr csrColIndC );


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseXcsr2gebsrNnz(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int m,
		//                                      int n,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr csrRowPtrA,
		//                                      CUdeviceptr csrColIndA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      int rowBlockDim,
		//                                      int colBlockDim,
		//                                      CUdeviceptr nnzTotalDevHostPtr);
		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseXcsr2gebsrNnz(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int m,
		//                                      int n,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr csrRowPtrA,
		//                                      CUdeviceptr csrColIndA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      int rowBlockDim,
		//                                      int colBlockDim,
		//                                      ref int nnzTotalDevHostPtr );

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseScsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int m,
		//                                      int n,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr csrValA,
		//                                      CUdeviceptr csrRowPtrA,
		//                                      CUdeviceptr csrColIndA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDim,
		//                                      int colBlockDim);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseDcsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int m,
		//                                      int n,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr csrValA,
		//                                      CUdeviceptr csrRowPtrA,
		//                                      CUdeviceptr csrColIndA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDim,
		//                                      int colBlockDim);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseCcsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int m,
		//                                      int n,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr csrValA,
		//                                      CUdeviceptr csrRowPtrA,
		//                                      CUdeviceptr csrColIndA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDim,
		//                                      int colBlockDim);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseZcsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int m,
		//                                      int n,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr csrValA,
		//                                      CUdeviceptr csrRowPtrA,
		//                                      CUdeviceptr csrColIndA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDim,
		//                                      int colBlockDim);


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseXgebsr2gebsrNnz(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int rowBlockDimA,
		//                                      int colBlockDimA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      int rowBlockDimC,
		//                                      int colBlockDimC,
		//                                      CUdeviceptr nnzTotalDevHostPtr);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseXgebsr2gebsrNnz(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int rowBlockDimA,
		//                                      int colBlockDimA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      int rowBlockDimC,
		//                                      int colBlockDimC,
		//                                      ref int nnzTotalDevHostPtr);

		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseSgebsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int rowBlockDimA,
		//                                      int colBlockDimA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDimC,
		//                                      int colBlockDimC);


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseDgebsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int rowBlockDimA,
		//                                      int colBlockDimA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDimC,
		//                                      int colBlockDimC);


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseCgebsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int rowBlockDimA,
		//                                      int colBlockDimA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDimC,
		//                                      int colBlockDimC);


		///// <summary/>
		//[DllImport(CUSPARSE_API_DLL_NAME)]
		//public static extern cusparseStatus cusparseZgebsr2gebsr(cusparseContext handle,
		//                                      cusparseDirection dirA,
		//                                      int mb,
		//                                      int nb,
		//                                      int nnzb,
		//                                      cusparseMatDescr descrA,
		//                                      CUdeviceptr bsrValA,
		//                                      CUdeviceptr bsrRowPtrA,
		//                                      CUdeviceptr bsrColIndA,
		//                                      int rowBlockDimA,
		//                                      int colBlockDimA,
		//                                      cusparseMatDescr descrC,
		//                                      CUdeviceptr bsrValC,
		//                                      CUdeviceptr bsrRowPtrC,
		//                                      CUdeviceptr bsrColIndC,
		//                                      int rowBlockDimC,
		//                                      int colBlockDimC);

		#endregion


		#region host
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											ref float alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref float beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											ref double alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref double beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											ref cuFloatComplex alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref cuFloatComplex beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											ref cuDoubleComplex alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref cuDoubleComplex beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											ref float alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref float beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											ref double alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref double beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											ref cuFloatComplex alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref cuFloatComplex beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											ref cuDoubleComplex alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											ref cuDoubleComplex beta,
											CUdeviceptr y);

		#endregion
		#region device
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZbsrxmv(cusparseContext handle,
											cusparseDirection dirA,
											cusparseOperation transA,
											int sizeOfMask,
											int mb,
											int nb,
											int nnzb,
											CUdeviceptr alpha,
											cusparseMatDescr descrA,
											CUdeviceptr bsrValA,
											CUdeviceptr bsrMaskPtrA,
											CUdeviceptr bsrRowPtrA,
											CUdeviceptr bsrEndPtrA,
											CUdeviceptr bsrColIndA,
											int blockDim,
											CUdeviceptr x,
											CUdeviceptr beta,
											CUdeviceptr y);

		#endregion



		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgebsr2gebsc_bufferSize(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgebsr2gebsc_bufferSize(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgebsr2gebsc_bufferSize(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgebsr2gebsc_bufferSize(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgebsr2gebsc_bufferSizeExt(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgebsr2gebsc_bufferSizeExt(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgebsr2gebsc_bufferSizeExt(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgebsr2gebsc_bufferSizeExt(cusparseContext handle,
															 int mb,
															 int nb,
															 int nnzb,
															 CUdeviceptr bsrVal,
															 CUdeviceptr bsrRowPtr,
															 CUdeviceptr bsrColInd,
															 int rowBlockDim,
															 int colBlockDim,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgebsr2gebsc(cusparseContext handle,
											  int mb,
											  int nb,
											  int nnzb,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int rowBlockDim,
											  int colBlockDim,
											  CUdeviceptr bscVal,
											  CUdeviceptr bscRowInd,
											  CUdeviceptr bscColPtr,
											  cusparseAction copyValues,
											  cusparseIndexBase baseIdx,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgebsr2gebsc(cusparseContext handle,
											  int mb,
											  int nb,
											  int nnzb,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int rowBlockDim,
											  int colBlockDim,
											  CUdeviceptr bscVal,
											  CUdeviceptr bscRowInd,
											  CUdeviceptr bscColPtr,
											  cusparseAction copyValues,
											  cusparseIndexBase baseIdx,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgebsr2gebsc(cusparseContext handle,
											  int mb,
											  int nb,
											  int nnzb,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int rowBlockDim,
											  int colBlockDim,
											  CUdeviceptr bscVal,
											  CUdeviceptr bscRowInd,
											  CUdeviceptr bscColPtr,
											  cusparseAction copyValues,
											  cusparseIndexBase baseIdx,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgebsr2gebsc(cusparseContext handle,
											  int mb,
											  int nb,
											  int nnzb,
											  CUdeviceptr bsrVal,
											  CUdeviceptr bsrRowPtr,
											  CUdeviceptr bsrColInd,
											  int rowBlockDim,
											  int colBlockDim,
											  CUdeviceptr bscVal,
											  CUdeviceptr bscRowInd,
											  CUdeviceptr bscColPtr,
											  cusparseAction copyValues,
											  cusparseIndexBase baseIdx,
											  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXgebsr2csr(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrRowPtrA,
											  CUdeviceptr bsrColIndA,
											  int rowBlockDim,
											  int colBlockDim,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgebsr2csr(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrValA,
											  CUdeviceptr bsrRowPtrA,
											  CUdeviceptr bsrColIndA,
											  int rowBlockDim,
											  int colBlockDim,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgebsr2csr(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrValA,
											  CUdeviceptr bsrRowPtrA,
											  CUdeviceptr bsrColIndA,
											  int rowBlockDim,
											  int colBlockDim,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgebsr2csr(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrValA,
											  CUdeviceptr bsrRowPtrA,
											  CUdeviceptr bsrColIndA,
											  int rowBlockDim,
											  int colBlockDim,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgebsr2csr(cusparseContext handle,
											  cusparseDirection dirA,
											  int mb,
											  int nb,
											  cusparseMatDescr descrA,
											  CUdeviceptr bsrValA,
											  CUdeviceptr bsrRowPtrA,
											  CUdeviceptr bsrColIndA,
											  int rowBlockDim,
											  int colBlockDim,
											  cusparseMatDescr descrC,
											  CUdeviceptr csrValC,
											  CUdeviceptr csrRowPtrC,
											  CUdeviceptr csrColIndC);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2gebsr_bufferSize(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2gebsr_bufferSize(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2gebsr_bufferSize(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2gebsr_bufferSize(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2gebsr_bufferSizeExt(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2gebsr_bufferSizeExt(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2gebsr_bufferSizeExt(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2gebsr_bufferSizeExt(cusparseContext handle,
														   cusparseDirection dirA,
														   int m,
														   int n,
														   cusparseMatDescr descrA,
														   CUdeviceptr csrValA,
														   CUdeviceptr csrRowPtrA,
														   CUdeviceptr csrColIndA,
														   int rowBlockDim,
														   int colBlockDim,
														   ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsr2gebsrNnz(cusparseContext handle,
												   cusparseDirection dirA,
												   int m,
												   int n,
												   cusparseMatDescr descrA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   cusparseMatDescr descrC,
												   CUdeviceptr bsrRowPtrC,
												   int rowBlockDim,
												   int colBlockDim,
												   ref int nnzTotalDevHostPtr,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsr2gebsrNnz(cusparseContext handle,
												   cusparseDirection dirA,
												   int m,
												   int n,
												   cusparseMatDescr descrA,
												   CUdeviceptr csrRowPtrA,
												   CUdeviceptr csrColIndA,
												   cusparseMatDescr descrC,
												   CUdeviceptr bsrRowPtrC,
												   int rowBlockDim,
												   int colBlockDim,
												   CUdeviceptr nnzTotalDevHostPtr,
												   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2gebsr(cusparseContext handle,
												cusparseDirection dirA,
												int m,
												int n,
												cusparseMatDescr descrA,
												CUdeviceptr csrValA,
												CUdeviceptr csrRowPtrA,
												CUdeviceptr csrColIndA,
												cusparseMatDescr descrC,
												CUdeviceptr bsrValC,
												CUdeviceptr bsrRowPtrC,
												CUdeviceptr bsrColIndC,
												int rowBlockDim,
												int colBlockDim,
												CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2gebsr(cusparseContext handle,
												cusparseDirection dirA,
												int m,
												int n,
												cusparseMatDescr descrA,
												CUdeviceptr csrValA,
												CUdeviceptr csrRowPtrA,
												CUdeviceptr csrColIndA,
												cusparseMatDescr descrC,
												CUdeviceptr bsrValC,
												CUdeviceptr bsrRowPtrC,
												CUdeviceptr bsrColIndC,
												int rowBlockDim,
												int colBlockDim,
												CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2gebsr(cusparseContext handle,
												cusparseDirection dirA,
												int m,
												int n,
												cusparseMatDescr descrA,
												CUdeviceptr csrValA,
												CUdeviceptr csrRowPtrA,
												CUdeviceptr csrColIndA,
												cusparseMatDescr descrC,
												CUdeviceptr bsrValC,
												CUdeviceptr bsrRowPtrC,
												CUdeviceptr bsrColIndC,
												int rowBlockDim,
												int colBlockDim,
												CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2gebsr(cusparseContext handle,
												cusparseDirection dirA,
												int m,
												int n,
												cusparseMatDescr descrA,
												CUdeviceptr csrValA,
												CUdeviceptr csrRowPtrA,
												CUdeviceptr csrColIndA,
												cusparseMatDescr descrC,
												CUdeviceptr bsrValC,
												CUdeviceptr bsrRowPtrC,
												CUdeviceptr bsrColIndC,
												int rowBlockDim,
												int colBlockDim,
												CUdeviceptr pBuffer);


		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgebsr2gebsr_bufferSize(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgebsr2gebsr_bufferSize(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgebsr2gebsr_bufferSize(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgebsr2gebsr_bufferSize(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref int pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgebsr2gebsr_bufferSizeExt(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgebsr2gebsr_bufferSizeExt(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgebsr2gebsr_bufferSizeExt(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgebsr2gebsr_bufferSizeExt(cusparseContext handle,
															 cusparseDirection dirA,
															 int mb,
															 int nb,
															 int nnzb,
															 cusparseMatDescr descrA,
															 CUdeviceptr bsrValA,
															 CUdeviceptr bsrRowPtrA,
															 CUdeviceptr bsrColIndA,
															 int rowBlockDimA,
															 int colBlockDimA,
															 int rowBlockDimC,
															 int colBlockDimC,
															 ref SizeT pBufferSize);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXgebsr2gebsrNnz(cusparseContext handle,
													 cusparseDirection dirA,
													 int mb,
													 int nb,
													 int nnzb,
													 cusparseMatDescr descrA,
													 CUdeviceptr bsrRowPtrA,
													 CUdeviceptr bsrColIndA,
													 int rowBlockDimA,
													 int colBlockDimA,
													 cusparseMatDescr descrC,
													 CUdeviceptr bsrRowPtrC,
													 int rowBlockDimC,
													 int colBlockDimC,
													 ref int nnzTotalDevHostPtr,
													 CUdeviceptr pBuffer);
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXgebsr2gebsrNnz(cusparseContext handle,
													 cusparseDirection dirA,
													 int mb,
													 int nb,
													 int nnzb,
													 cusparseMatDescr descrA,
													 CUdeviceptr bsrRowPtrA,
													 CUdeviceptr bsrColIndA,
													 int rowBlockDimA,
													 int colBlockDimA,
													 cusparseMatDescr descrC,
													 CUdeviceptr bsrRowPtrC,
													 int rowBlockDimC,
													 int colBlockDimC,
													 CUdeviceptr nnzTotalDevHostPtr,
													 CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseSgebsr2gebsr(cusparseContext handle,
												  cusparseDirection dirA,
												  int mb,
												  int nb,
												  int nnzb,
												  cusparseMatDescr descrA,
												  CUdeviceptr bsrValA,
												  CUdeviceptr bsrRowPtrA,
												  CUdeviceptr bsrColIndA,
												  int rowBlockDimA,
												  int colBlockDimA,
												  cusparseMatDescr descrC,
												  CUdeviceptr bsrValC,
												  CUdeviceptr bsrRowPtrC,
												  CUdeviceptr bsrColIndC,
												  int rowBlockDimC,
												  int colBlockDimC,
												  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDgebsr2gebsr(cusparseContext handle,
												  cusparseDirection dirA,
												  int mb,
												  int nb,
												  int nnzb,
												  cusparseMatDescr descrA,
												  CUdeviceptr bsrValA,
												  CUdeviceptr bsrRowPtrA,
												  CUdeviceptr bsrColIndA,
												  int rowBlockDimA,
												  int colBlockDimA,
												  cusparseMatDescr descrC,
												  CUdeviceptr bsrValC,
												  CUdeviceptr bsrRowPtrC,
												  CUdeviceptr bsrColIndC,
												  int rowBlockDimC,
												  int colBlockDimC,
												  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCgebsr2gebsr(cusparseContext handle,
												  cusparseDirection dirA,
												  int mb,
												  int nb,
												  int nnzb,
												  cusparseMatDescr descrA,
												  CUdeviceptr bsrValA,
												  CUdeviceptr bsrRowPtrA,
												  CUdeviceptr bsrColIndA,
												  int rowBlockDimA,
												  int colBlockDimA,
												  cusparseMatDescr descrC,
												  CUdeviceptr bsrValC,
												  CUdeviceptr bsrRowPtrC,
												  CUdeviceptr bsrColIndC,
												  int rowBlockDimC,
												  int colBlockDimC,
												  CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZgebsr2gebsr(cusparseContext handle,
												  cusparseDirection dirA,
												  int mb,
												  int nb,
												  int nnzb,
												  cusparseMatDescr descrA,
												  CUdeviceptr bsrValA,
												  CUdeviceptr bsrRowPtrA,
												  CUdeviceptr bsrColIndA,
												  int rowBlockDimA,
												  int colBlockDimA,
												  cusparseMatDescr descrC,
												  CUdeviceptr bsrValC,
												  CUdeviceptr bsrRowPtrC,
												  CUdeviceptr bsrColIndC,
												  int rowBlockDimC,
												  int colBlockDimC,
												  CUdeviceptr pBuffer);



		/* --- Sparse Matrix Sorting --- */

		/* Description: Create a identity sequence p=[0,1,...,n-1]. */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCreateIdentityPermutation(cusparseContext handle,
                                                               int n,
                                                               CUdeviceptr p);

		/* Description: Sort sparse matrix stored in COO format */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcoosort_bufferSizeExt(cusparseContext handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            CUdeviceptr cooRowsA,
                                                            CUdeviceptr cooColsA,
                                                            ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcoosortByRow(cusparseContext handle,
                                                   int m,
                                                   int n,
                                                   int nnz,
                                                   CUdeviceptr cooRowsA,
                                                   CUdeviceptr cooColsA,
                                                   CUdeviceptr P,
                                                   CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcoosortByColumn(cusparseContext handle,
                                                      int m,
                                                      int n, 
                                                      int nnz,
                                                      CUdeviceptr cooRowsA,
                                                      CUdeviceptr cooColsA,
                                                      CUdeviceptr P,
                                                      CUdeviceptr pBuffer);

		/* Description: Sort sparse matrix stored in CSR format */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrsort_bufferSizeExt(cusparseContext handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            CUdeviceptr csrRowPtrA,
                                                            CUdeviceptr csrColIndA,
                                                            ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcsrsort(cusparseContext handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              cusparseMatDescr descrA,
                                              CUdeviceptr csrRowPtrA,
                                              CUdeviceptr csrColIndA,
                                              CUdeviceptr P,
                                              CUdeviceptr pBuffer);
    
		/* Description: Sort sparse matrix stored in CSC format */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcscsort_bufferSizeExt(cusparseContext handle,
                                                            int m,
                                                            int n,
                                                            int nnz,
                                                            CUdeviceptr cscColPtrA,
                                                            CUdeviceptr cscRowIndA,
                                                            ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseXcscsort(cusparseContext handle,
                                              int m,
                                              int n,
                                              int nnz,
                                              cusparseMatDescr descrA,
                                              CUdeviceptr cscColPtrA,
                                              CUdeviceptr cscRowIndA,
                                              CUdeviceptr P,
                                              CUdeviceptr pBuffer);

		/* Description: Wrapper that sorts sparse matrix stored in CSR format 
		   (without exposing the permutation). */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsru2csr_bufferSizeExt(cusparseContext handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             CUdeviceptr csrVal,
                                                             CUdeviceptr csrRowPtr,
                                                             CUdeviceptr csrColInd,
                                                             csru2csrInfo  info,
                                                             ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsru2csr_bufferSizeExt(cusparseContext handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             CUdeviceptr csrVal,
                                                             CUdeviceptr csrRowPtr,
                                                             CUdeviceptr csrColInd,
                                                             csru2csrInfo  info,
                                                             ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsru2csr_bufferSizeExt(cusparseContext handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             CUdeviceptr csrVal,
                                                             CUdeviceptr csrRowPtr,
                                                             CUdeviceptr csrColInd,
                                                             csru2csrInfo  info,
                                                             ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsru2csr_bufferSizeExt(cusparseContext handle,
                                                             int m,
                                                             int n,
                                                             int nnz,
                                                             CUdeviceptr csrVal,
                                                             CUdeviceptr csrRowPtr,
                                                             CUdeviceptr csrColInd,
                                                             csru2csrInfo  info,
                                                             ref SizeT pBufferSizeInBytes);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsru2csr(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsru2csr(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsru2csr(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsru2csr(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);

		/* Description: Wrapper that un-sorts sparse matrix stored in CSR format 
		   (without exposing the permutation). */
		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseScsr2csru(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseDcsr2csru(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseCcsr2csru(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);

		/// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
		public static extern cusparseStatus cusparseZcsr2csru(cusparseContext handle,
                                               int m,
                                               int n,
                                               int nnz,
                                               cusparseMatDescr descrA,
                                               CUdeviceptr csrVal,
                                               CUdeviceptr csrRowPtr,
                                               CUdeviceptr csrColInd,
                                               csru2csrInfo  info,
                                               CUdeviceptr pBuffer);



        #endregion

        #region prune dense matrix to a sparse matrix with CSR format
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    ref half threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csr_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    ref float threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csr_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    ref double threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    ref half threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    ref float threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csrNnz(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    ref double threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csrNnz(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csr(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    ref half threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csr(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csr(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    ref float threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csr(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csr(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    ref double threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csr(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);

        /* Description: prune sparse matrix with CSR format to another sparse matrix with CSR format */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref half threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csr_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref float threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csr_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref double threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csr_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref half threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref float threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csrNnz(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csrNnz(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref double threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csrNnz(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csr(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref half threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csr(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csr(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref float threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csr(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csr(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    ref double threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csr(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    CUdeviceptr threshold,
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    CUdeviceptr pBuffer);

        /* Description: prune dense matrix to a sparse matrix with CSR format by percentage */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csrByPercentage_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csrByPercentage_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csrByPercentage_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csrNnzByPercentage(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csrNnzByPercentage(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneDense2csrByPercentage(
    cusparseContext handle,
    int m,
    int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneDense2csrByPercentage(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneDense2csrByPercentage(
            cusparseContext handle,
            int m,
            int n,
    CUdeviceptr A,
    int lda,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    CUdeviceptr pBuffer);


        /* Description: prune sparse matrix to a sparse matrix with CSR format by percentage*/
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    ref SizeT pBufferSizeInBytes);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csrNnzByPercentage(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csrNnzByPercentage(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    ref int nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csrNnzByPercentage(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr nnzTotalDevHostPtr, /* can be on host or device */
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseHpruneCsr2csrByPercentage(
    cusparseContext handle,
    int m,
    int n,
    int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpruneCsr2csrByPercentage(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDpruneCsr2csrByPercentage(
            cusparseContext handle,
            int m,
            int n,
            int nnzA,
    cusparseMatDescr descrA,
    CUdeviceptr csrValA,
    CUdeviceptr csrRowPtrA,
    CUdeviceptr csrColIndA,
    float percentage, /* between 0 to 100 */
    cusparseMatDescr descrC,
    CUdeviceptr csrValC,
    CUdeviceptr csrRowPtrC,
    CUdeviceptr csrColIndC,
    pruneInfo info,
    CUdeviceptr pBuffer);
        #endregion
    }
}
