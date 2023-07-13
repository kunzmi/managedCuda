﻿// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
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
        internal const string CUSPARSE_API_DLL_NAME = "cusparse64_12";

#if (NETCOREAPP)
        internal const string CUSPARSE_API_DLL_NAME_LINUX = "cusparse";

        static CudaSparseNativeMethods()
        {
            NativeLibrary.SetDllImportResolver(typeof(CudaSparseNativeMethods).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            IntPtr libHandle = IntPtr.Zero;

            if (libraryName == CUSPARSE_API_DLL_NAME)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    bool res = NativeLibrary.TryLoad(CUSPARSE_API_DLL_NAME_LINUX, assembly, DllImportSearchPath.SafeDirectories, out libHandle);
                    if (!res)
                    {
                        Debug.WriteLine("Failed to load '" + CUSPARSE_API_DLL_NAME_LINUX + "' shared library. Falling back to (Windows-) default library name '"
                            + CUSPARSE_API_DLL_NAME + "'. Check LD_LIBRARY_PATH environment variable for correct paths.");
                    }
                }
            }
            //On Windows, use the default library name
            return libHandle;
        }
#endif

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
        public static extern cusparseStatus cusparseSetMatIndexBase(cusparseMatDescr descrA, IndexBase indexBase);
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern IndexBase cusparseGetMatIndexBase(cusparseMatDescr descrA);
        #endregion


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
        ///// <summary/>
        //[DllImport(CUSPARSE_API_DLL_NAME)]
        //public static extern cusparseStatus cusparseSetColorAlgs(cusparseColorInfo info, cusparseColorAlg alg);
        ///// <summary/>
        //[DllImport(CUSPARSE_API_DLL_NAME)]
        //public static extern cusparseStatus cusparseGetColorAlgs(cusparseColorInfo info, ref cusparseColorAlg alg);
        #endregion

        #region prune information
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreatePruneInfo(ref pruneInfo info);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDestroyPruneInfo(pruneInfo info);
        #endregion

        #region Sparse Level 2 routines
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
                                    IndexBase idxBase,
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
                                    IndexBase idxBase,
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
                                    IndexBase idxBase,
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
                                    IndexBase idxBase,
                                    CUdeviceptr pBuffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseZgemvi_bufferSize(cusparseContext handle,
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
                                    IndexBase idxBase,
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
                                    IndexBase idxBase,
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
                                    IndexBase idxBase,
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
                                    IndexBase idxBase,
                                    CUdeviceptr pBuffer);

        #endregion



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



        #endregion

        #region Sparse Level 3 routines

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












        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSgtsv2_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

        /// <summary/>
		[DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDgtsv2_bufferSizeExt(cusparseContext handle, int m, int n, CUdeviceptr dl, CUdeviceptr d, CUdeviceptr du, CUdeviceptr B, int ldb, ref SizeT bufferSizeInBytes);

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


        /* Description: This routine compresses the indecis of rows or columns.
		   It can be interpreted as a conversion from COO to CSR sparse storage
		   format. */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseXcoo2csr(cusparseContext handle, CUdeviceptr cooRowInd, int nnz, int m, CUdeviceptr csrRowPtr, IndexBase idxBase);

        /* Description: This routine uncompresses the indecis of rows or columns.
		   It can be interpreted as a conversion from CSR to COO sparse storage
		   format. */
        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseXcsr2coo(cusparseContext handle, CUdeviceptr csrRowPtr, int nnz, int m, CUdeviceptr cooRowInd, IndexBase idxBase);


        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCsr2cscEx2(cusparseContext handle,
                   int m, int n, int nnz,
                   CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd,
                   CUdeviceptr cscVal, CUdeviceptr cscColPtr, CUdeviceptr cscRowInd,
                   cudaDataType valType,
                   cusparseAction copyValues,
                   IndexBase idxBase,
                   cusparseCsr2CscAlg alg,
                   CUdeviceptr buffer);

        /// <summary/>
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCsr2cscEx2_bufferSize(cusparseContext handle,
                              int m, int n, int nnz, CUdeviceptr csrVal, CUdeviceptr csrRowPtr, CUdeviceptr csrColInd,
                              CUdeviceptr cscVal, CUdeviceptr cscColPtr, CUdeviceptr cscRowInd, cudaDataType valType,
                              cusparseAction copyValues, IndexBase idxBase, cusparseCsr2CscAlg alg, ref SizeT bufferSize);


        #endregion

        #region Sparse Level 4 routines



        /* Description: Compute sparse - sparse matrix multiplication for matrices 
   stored in CSR format. */



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
            ref SizeT pBufferSizeInBytes);

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
            CUdeviceptr pBuffer);

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
            CUdeviceptr pBuffer);

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
            CUdeviceptr pBuffer);

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
            CUdeviceptr pBuffer);
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
        //                                      IndexBase baseIdx);

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
        //                                      IndexBase baseIdx);

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
        //                                      IndexBase baseIdx);

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
        //                                      IndexBase baseIdx);

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
                                              IndexBase baseIdx,
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
                                              IndexBase baseIdx,
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
                                              IndexBase baseIdx,
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
                                              IndexBase baseIdx,
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
                                                             csru2csrInfo info,
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
                                                             csru2csrInfo info,
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
                                                             csru2csrInfo info,
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
                                                             csru2csrInfo info,
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
                                               csru2csrInfo info,
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
                                               csru2csrInfo info,
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
                                               csru2csrInfo info,
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
                                               csru2csrInfo info,
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
                                               csru2csrInfo info,
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
                                               csru2csrInfo info,
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
                                               csru2csrInfo info,
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
                                               csru2csrInfo info,
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

        #region Generic API

        #region Sparse vector
        // #############################################################################
        // # SPARSE VECTOR DESCRIPTOR
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateSpVec(ref cusparseSpVecDescr spVecDescr,
                    long size,
                    long nnz,
                    CUdeviceptr indices,
                    CUdeviceptr values,
                    IndexType idxType,
                    IndexBase idxBase,
                    cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstSpVec(ref cusparseConstSpVecDescr spVecDescr,
                    long size,
                    long nnz,
                    CUdeviceptr indices,
                    CUdeviceptr values,
                    IndexType idxType,
                    IndexBase idxBase,
                    cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDestroySpVec(cusparseConstSpVecDescr spVecDescr);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVecGet(cusparseSpVecDescr spVecDescr,
                 ref long size,
                 ref long nnz,
                 ref CUdeviceptr indices,
                 ref CUdeviceptr values,
                 ref IndexType idxType,
                 ref IndexBase idxBase,
                 ref cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstSpVecGet(cusparseConstSpVecDescr spVecDescr,
                 ref long size,
                 ref long nnz,
                 ref CUdeviceptr indices,
                 ref CUdeviceptr values,
                 ref IndexType idxType,
                 ref IndexBase idxBase,
                 ref cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVecGetIndexBase(cusparseConstSpVecDescr spVecDescr,
                          ref IndexBase idxBase);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVecGetValues(cusparseSpVecDescr spVecDescr,
                        ref CUdeviceptr values);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstSpVecGetValues(cusparseConstSpVecDescr spVecDescr,
                        ref CUdeviceptr values);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVecSetValues(cusparseSpVecDescr spVecDescr,
                       CUdeviceptr values);

        #endregion
        #region Dense Vector

        // #############################################################################
        // # DENSE VECTOR DESCRIPTOR
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateDnVec(ref cusparseDnVecDescr dnVecDescr,
                    long size,
                    CUdeviceptr values,
                    cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstDnVec(ref cusparseConstDnVecDescr dnVecDescr,
                    long size,
                    CUdeviceptr values,
                    cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDestroyDnVec(cusparseConstDnVecDescr dnVecDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnVecGet(cusparseDnVecDescr dnVecDescr,
                 ref long size,
                 ref CUdeviceptr values,
                 ref cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstDnVecGet(cusparseConstDnVecDescr dnVecDescr,
                 ref long size,
                 ref CUdeviceptr values,
                 ref cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnVecGetValues(cusparseDnVecDescr dnVecDescr,
                       ref CUdeviceptr values);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstDnVecGetValues(cusparseConstDnVecDescr dnVecDescr,
                       ref CUdeviceptr values);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnVecSetValues(cusparseDnVecDescr dnVecDescr,
                       CUdeviceptr values);

        #endregion
        #region Sparse Matrix
        // #############################################################################
        // # SPARSE MATRIX DESCRIPTOR
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDestroySpMat(cusparseConstSpMatDescr spMatDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatGetFormat(cusparseConstSpMatDescr spMatDescr,
                       ref Format format);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatGetIndexBase(cusparseConstSpMatDescr spMatDescr,
                          ref IndexBase idxBase);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatGetValues(cusparseSpMatDescr spMatDescr,
                       ref CUdeviceptr values);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstSpMatGetValues(cusparseConstSpMatDescr spMatDescr,
                       ref CUdeviceptr values);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatSetValues(cusparseSpMatDescr spMatDescr,
                       CUdeviceptr values);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatGetSize(cusparseConstSpMatDescr spMatDescr,
                     ref long rows,
                     ref long cols,
                     ref long nnz);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr spMatDescr,
                             ref int batchCount);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatSetStridedBatch(cusparseSpMatDescr spMatDescr,
                             int batchCount);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCooSetStridedBatch(cusparseSpMatDescr spMatDescr,
                            int batchCount,
                            long batchStride);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCsrSetStridedBatch(cusparseSpMatDescr spMatDescr,
                            int batchCount,
                            long offsetsBatchStride,
                            long columnsValuesBatchStride);

        [DllImport(CUSPARSE_API_DLL_NAME)]

        public static extern cusparseStatus cusparseBsrSetStridedBatch(cusparseSpMatDescr spMatDescr,
                           int batchCount,
                           long offsetsBatchStride,
                           long columnsValuesBatchStride,
                           long ValuesBatchStride);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatGetAttribute(cusparseConstSpMatDescr spMatDescr,
                          cusparseSpMatAttribute attribute,
                          ref cusparseDiagType data,
                          SizeT dataSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpMatSetAttribute(cusparseSpMatDescr spMatDescr,
                          cusparseSpMatAttribute attribute,
                          ref cusparseDiagType data,
                          SizeT dataSize);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMatGetAttribute(cusparseConstSpMatDescr spMatDescr,
                          cusparseSpMatAttribute attribute,
                          ref cusparseFillMode data,
                          SizeT dataSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpMatSetAttribute(cusparseSpMatDescr spMatDescr,
                          cusparseSpMatAttribute attribute,
                          ref cusparseFillMode data,
                          SizeT dataSize);
        //------------------------------------------------------------------------------
        // ### CSR ###

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateCsr(ref cusparseSpMatDescr spMatDescr,
                  long rows,
                  long cols,
                  long nnz,
                  CUdeviceptr csrRowOffsets,
                  CUdeviceptr csrColInd,
                  CUdeviceptr csrValues,
                  IndexType csrRowOffsetsType,
                  IndexType csrColIndType,
                  IndexBase idxBase,
                  cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstCsr(ref cusparseConstSpMatDescr spMatDescr,
                  long rows,
                  long cols,
                  long nnz,
                  CUdeviceptr csrRowOffsets,
                  CUdeviceptr csrColInd,
                  CUdeviceptr csrValues,
                  IndexType csrRowOffsetsType,
                  IndexType csrColIndType,
                  IndexBase idxBase,
                  cudaDataType valueType);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateCsc(ref cusparseSpMatDescr spMatDescr,
                  long rows,
                  long cols,
                  long nnz,
                  CUdeviceptr cscColOffsets,
                  CUdeviceptr cscRowInd,
                  CUdeviceptr cscValues,
                  IndexType cscColOffsetsType,
                  IndexType cscRowIndType,
                  IndexBase idxBase,
                  cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstCsc(ref cusparseConstSpMatDescr spMatDescr,
                  long rows,
                  long cols,
                  long nnz,
                  CUdeviceptr cscColOffsets,
                  CUdeviceptr cscRowInd,
                  CUdeviceptr cscValues,
                  IndexType cscColOffsetsType,
                  IndexType cscRowIndType,
                  IndexBase idxBase,
                  cudaDataType valueType);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCsrGet(cusparseSpMatDescr spMatDescr,
               ref long rows,
               ref long cols,
               ref long nnz,
               ref CUdeviceptr csrRowOffsets,
               ref CUdeviceptr csrColInd,
               ref CUdeviceptr csrValues,
               ref IndexType csrRowOffsetsType,
               ref IndexType csrColIndType,
               ref IndexBase idxBase,
               ref cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstCsrGet(cusparseConstSpMatDescr spMatDescr,
               ref long rows,
               ref long cols,
               ref long nnz,
               ref CUdeviceptr csrRowOffsets,
               ref CUdeviceptr csrColInd,
               ref CUdeviceptr csrValues,
               ref IndexType csrRowOffsetsType,
               ref IndexType csrColIndType,
               ref IndexBase idxBase,
               ref cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCscGet(cusparseSpMatDescr spMatDescr,
               ref long rows,
               ref long cols,
               ref long nnz,
               ref CUdeviceptr cscColOffsets,
               ref CUdeviceptr cscRowInd,
               ref CUdeviceptr cscValues,
               ref IndexType cscColOffsetsType,
               ref IndexType cscRowIndType,
               ref IndexBase idxBase,
               ref cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstCscGet(cusparseConstSpMatDescr spMatDescr,
                    ref long rows,
                    ref long cols,
                    ref long nnz,
                    ref CUdeviceptr cscColOffsets,
                    ref CUdeviceptr cscRowInd,
                    ref CUdeviceptr cscValues,
                    ref IndexType cscColOffsetsType,
                    ref IndexType cscRowIndType,
                    ref IndexBase idxBase,
                    ref cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCsrSetPointers(cusparseSpMatDescr spMatDescr,
                       CUdeviceptr csrRowOffsets,
                       CUdeviceptr csrColInd,
                       CUdeviceptr csrValues);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCscSetPointers(cusparseSpMatDescr spMatDescr,
                       CUdeviceptr cscColOffsets,
                       CUdeviceptr cscRowInd,
                       CUdeviceptr cscValues);
        //------------------------------------------------------------------------------
        // ### BSR ###

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateBsr(ref cusparseSpMatDescr spMatDescr,
                  long brows,
                  long bcols,
                  long bnnz,
                  long rowBlockDim,
                  long colBlockDim,
                  CUdeviceptr bsrRowOffsets,
                  CUdeviceptr bsrColInd,
                  CUdeviceptr bsrValues,
                  IndexType bsrRowOffsetsType,
                  IndexType bsrColIndType,
                  IndexBase idxBase,
                  cudaDataType valueType,
                  Order order);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstBsr(ref cusparseConstSpMatDescr spMatDescr,
                       long brows,
                       long bcols,
                       long bnnz,
                       long rowBlockDim,
                       long colBlockDim,
                       CUdeviceptr bsrRowOffsets,
                       CUdeviceptr bsrColInd,
                       CUdeviceptr bsrValues,
                       IndexType bsrRowOffsetsType,
                       IndexType bsrColIndType,
                       IndexBase idxBase,
                       cudaDataType valueType,
                       Order order);

        //------------------------------------------------------------------------------

        //------------------------------------------------------------------------------
        // ### COO ###

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateCoo(ref cusparseSpMatDescr spMatDescr,
                  long rows,
                  long cols,
                  long nnz,
                  CUdeviceptr cooRowInd,
                  CUdeviceptr cooColInd,
                  CUdeviceptr cooValues,
                  IndexType cooIdxType,
                  IndexBase idxBase,
                  cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstCoo(ref cusparseConstSpMatDescr spMatDescr,
                  long rows,
                  long cols,
                  long nnz,
                  CUdeviceptr cooRowInd,
                  CUdeviceptr cooColInd,
                  CUdeviceptr cooValues,
                  IndexType cooIdxType,
                  IndexBase idxBase,
                  cudaDataType valueType);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCooGet(cusparseSpMatDescr spMatDescr,
               ref long rows,
               ref long cols,
               ref long nnz,
               ref CUdeviceptr cooRowInd,  // COO row indices
               ref CUdeviceptr cooColInd,  // COO column indices
               ref CUdeviceptr cooValues,  // COO values
               ref IndexType idxType,
               ref IndexBase idxBase,
               ref cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstCooGet(cusparseConstSpMatDescr spMatDescr,
               ref long rows,
               ref long cols,
               ref long nnz,
               ref CUdeviceptr cooRowInd,  // COO row indices
               ref CUdeviceptr cooColInd,  // COO column indices
               ref CUdeviceptr cooValues,  // COO values
               ref IndexType idxType,
               ref IndexBase idxBase,
               ref cudaDataType valueType);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCooSetPointers(cusparseSpMatDescr spMatDescr,
                       CUdeviceptr cooRows,
                       CUdeviceptr cooColumns,
                       CUdeviceptr cooValues);
        //------------------------------------------------------------------------------
        // ### Sliced ELLPACK ###

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateSlicedEll(ref cusparseSpMatDescr spMatDescr,
                        long rows,
                        long cols,
                        long nnz,
                        long sellValuesSize,
                        long sliceSize,
                        CUdeviceptr sellSliceOffsets,
                        CUdeviceptr sellColInd,
                        CUdeviceptr sellValues,
                        IndexType sellSliceOffsetsType,
                        IndexType sellColIndType,
                        IndexBase idxBase,
                        cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstSlicedEll(ref cusparseConstSpMatDescr spMatDescr,
                             long rows,
                             long cols,
                             long nnz,
                             long sellValuesSize,
                             long sliceSize,
                             CUdeviceptr sellSliceOffsets,
                             CUdeviceptr sellColInd,
                             CUdeviceptr sellValues,
                             IndexType sellSliceOffsetsType,
                             IndexType sellColIndType,
                             IndexBase idxBase,
                             cudaDataType valueType);

        #endregion

        #region BLOCKED ELL

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseCreateBlockedEll(ref cusparseSpMatDescr spMatDescr,
                         long rows,
                         long cols,
                         long ellBlockSize,
                         long ellCols,
                         CUdeviceptr ellColInd,
                         CUdeviceptr ellValue,
                         IndexType ellIdxType,
                         IndexBase idxBase,
                         cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseCreateConstBlockedEll(ref cusparseConstSpMatDescr spMatDescr,
                         long rows,
                         long cols,
                         long ellBlockSize,
                         long ellCols,
                         CUdeviceptr ellColInd,
                         CUdeviceptr ellValue,
                         IndexType ellIdxType,
                         IndexBase idxBase,
                         cudaDataType valueType);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseBlockedEllGet(cusparseSpMatDescr spMatDescr,
                      ref long rows,
                      ref long cols,
                      ref long ellBlockSize,
                      ref long ellCols,
                      ref CUdeviceptr ellColInd,
                      ref CUdeviceptr ellValue,
                      ref IndexType ellIdxType,
                      ref IndexBase idxBase,
                      ref cudaDataType valueType);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseConstBlockedEllGet(cusparseConstSpMatDescr spMatDescr,
                      ref long rows,
                      ref long cols,
                      ref long ellBlockSize,
                      ref long ellCols,
                      ref CUdeviceptr ellColInd,
                      ref CUdeviceptr ellValue,
                      ref IndexType ellIdxType,
                      ref IndexBase idxBase,
                      ref cudaDataType valueType);
        #endregion

        #region SPARSE TO DENSE

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSparseToDense_bufferSize(cusparseContext handle,
                                 cusparseSpMatDescr matA,
                                 cusparseDnMatDescr matB,
                                 cusparseSparseToDenseAlg alg,
                                 ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSparseToDense(cusparseContext handle,
                      cusparseSpMatDescr matA,
                      cusparseDnMatDescr matB,
                      cusparseSparseToDenseAlg alg,
                      CUdeviceptr buffer);

        #endregion
        #region DENSE TO SPARSE

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDenseToSparse_bufferSize(cusparseContext handle,
                                 cusparseConstDnMatDescr matA,
                                 cusparseSpMatDescr matB,
                                 cusparseDenseToSparseAlg alg,
                                 ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDenseToSparse_analysis(cusparseContext handle,
                               cusparseConstDnMatDescr matA,
                               cusparseSpMatDescr matB,
                               cusparseDenseToSparseAlg alg,
                               CUdeviceptr buffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDenseToSparse_convert(cusparseContext handle,
                              cusparseConstDnMatDescr matA,
                              cusparseSpMatDescr matB,
                              cusparseDenseToSparseAlg alg,
                              CUdeviceptr buffer);

        #endregion
        #region Dense Matrix
        // #############################################################################
        // # DENSE MATRIX DESCRIPTOR
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateDnMat(ref cusparseDnMatDescr dnMatDescr,
                    long rows,
                    long cols,
                    long ld,
                    CUdeviceptr values,
                    cudaDataType valueType,
                    Order order);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseCreateConstDnMat(ref cusparseConstDnMatDescr dnMatDescr,
                    long rows,
                    long cols,
                    long ld,
                    CUdeviceptr values,
                    cudaDataType valueType,
                    Order order);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDestroyDnMat(cusparseConstDnMatDescr dnMatDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnMatGet(cusparseDnMatDescr dnMatDescr,
                 ref long rows,
                 ref long cols,
                 ref long ld,
                 ref CUdeviceptr values,
                 ref cudaDataType type,
                 ref Order order);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstDnMatGet(cusparseConstDnMatDescr dnMatDescr,
                 ref long rows,
                 ref long cols,
                 ref long ld,
                 ref CUdeviceptr values,
                 ref cudaDataType type,
                 ref Order order);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnMatGetValues(cusparseDnMatDescr dnMatDescr,
                       ref CUdeviceptr values);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseConstDnMatGetValues(cusparseConstDnMatDescr dnMatDescr,
                       ref CUdeviceptr values);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnMatSetValues(cusparseDnMatDescr dnMatDescr,
                       CUdeviceptr values);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnMatSetStridedBatch(cusparseDnMatDescr dnMatDescr,
                             int batchCount,
                             long batchStride);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr dnMatDescr,
                             ref int batchCount,
                             ref long batchStride);

        #endregion
        #region SPARSE VECTOR-VECTOR MULTIPLICATION
        // #############################################################################
        // # SPARSE VECTOR-VECTOR MULTIPLICATION
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseAxpby(cusparseContext handle,
              IntPtr alpha,
              cusparseConstSpVecDescr vecX,
              IntPtr beta,
              cusparseDnVecDescr vecY);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseAxpby(cusparseContext handle,
              CUdeviceptr alpha,
              cusparseConstSpVecDescr vecX,
              CUdeviceptr beta,
              cusparseDnVecDescr vecY);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseGather(cusparseContext handle,
               cusparseConstDnVecDescr vecY,
               cusparseSpVecDescr vecX);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseScatter(cusparseContext handle,
                cusparseConstSpVecDescr vecX,
                cusparseDnVecDescr vecY);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseRot(cusparseContext handle,
            IntPtr c_coeff,
            IntPtr s_coeff,
            cusparseSpVecDescr vecX,
            cusparseDnVecDescr vecY);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseRot(cusparseContext handle,
            CUdeviceptr c_coeff,
            CUdeviceptr s_coeff,
            cusparseSpVecDescr vecX,
            cusparseDnVecDescr vecY);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVV_bufferSize(cusparseContext handle,
                        cusparseOperation opX,
                        cusparseConstSpVecDescr vecX,
                        cusparseConstDnVecDescr vecY,

                        IntPtr result,
                        cudaDataType computeType,
                        ref SizeT bufferSize);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVV_bufferSize(cusparseContext handle,
                        cusparseOperation opX,
                        cusparseConstSpVecDescr vecX,
                        cusparseConstDnVecDescr vecY,

                        CUdeviceptr result,
                        cudaDataType computeType,
                        ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVV(cusparseContext handle,
             cusparseOperation opX,
             cusparseConstSpVecDescr vecX,
             cusparseConstDnVecDescr vecY,
             IntPtr result,
             cudaDataType computeType,
             CUdeviceptr externalBuffer);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpVV(cusparseContext handle,
             cusparseOperation opX,
             cusparseConstSpVecDescr vecX,
             cusparseConstDnVecDescr vecY,
             CUdeviceptr result,
             cudaDataType computeType,
             CUdeviceptr externalBuffer);
        #endregion
        #region SPARSE MATRIX-VECTOR MULTIPLICATION
        // #############################################################################
        // # SPARSE MATRIX-VECTOR MULTIPLICATION
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMV(cusparseContext handle,
             cusparseOperation opA,

             IntPtr alpha,
             cusparseConstSpMatDescr matA,
             cusparseConstDnVecDescr vecX,

             IntPtr beta,
             cusparseDnVecDescr vecY,
             cudaDataType computeType,
             SpMVAlg alg,
             CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMV_bufferSize(cusparseContext handle,
                        cusparseOperation opA,

                        IntPtr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnVecDescr vecX,

                        IntPtr beta,
                        cusparseDnVecDescr vecY,
                        cudaDataType computeType,
                        SpMVAlg alg,
                        ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMV(cusparseContext handle,
             cusparseOperation opA,

             CUdeviceptr alpha,
             cusparseConstSpMatDescr matA,
             cusparseConstDnVecDescr vecX,

             CUdeviceptr beta,
             cusparseDnVecDescr vecY,
             cudaDataType computeType,
             SpMVAlg alg,
             CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMV_bufferSize(cusparseContext handle,
                        cusparseOperation opA,

                        CUdeviceptr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnVecDescr vecX,

                        CUdeviceptr beta,
                        cusparseDnVecDescr vecY,
                        cudaDataType computeType,
                        SpMVAlg alg,
                        ref SizeT bufferSize);

        #endregion

        #region SPARSE TRIANGULAR MATRIX SOLVE

        // #############################################################################
        // # SPARSE TRIANGULAR MATRIX SOLVE
        // #############################################################################




        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_createDescr(ref cusparseSpSMDescr descr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_destroyDescr(cusparseSpSMDescr descr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_bufferSize(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,
                        IntPtr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        cusparseSpSMAlg alg,
                        cusparseSpSMDescr spsmDescr,
                        ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_analysis(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,
                        IntPtr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        cusparseSpSMAlg alg,
                        cusparseSpSMDescr spsmDescr,
                        CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_solve(cusparseContext handle,
                    cusparseOperation opA,
                    cusparseOperation opB,
                    IntPtr alpha,
                    cusparseConstSpMatDescr matA,
                    cusparseConstDnMatDescr matB,
                    cusparseDnMatDescr matC,
                    cudaDataType computeType,
                    cusparseSpSMAlg alg,
                    cusparseSpSMDescr spsmDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_bufferSize(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,
                        CUdeviceptr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        cusparseSpSMAlg alg,
                        cusparseSpSMDescr spsmDescr,
                        ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_analysis(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,
                        CUdeviceptr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        cusparseSpSMAlg alg,
                        cusparseSpSMDescr spsmDescr,
                        CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpSM_solve(cusparseContext handle,
                    cusparseOperation opA,
                    cusparseOperation opB,
                    CUdeviceptr alpha,
                    cusparseConstSpMatDescr matA,
                    cusparseConstDnMatDescr matB,
                    cusparseDnMatDescr matC,
                    cudaDataType computeType,
                    cusparseSpSMAlg alg,
                    cusparseSpSMDescr spsmDescr);

        #endregion

        #region SPARSE TRIANGULAR VECTOR SOLVE



        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_createDescr(ref cusparseSpSVDescr descr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_destroyDescr(cusparseSpSVDescr descr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_bufferSize(cusparseContext handle,
                        cusparseOperation opA,
                        IntPtr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnVecDescr vecX,
                        cusparseDnVecDescr vecY,
                        cudaDataType computeType,
                        cusparseSpSVAlg alg,
                        cusparseSpSVDescr spsvDescr,
                        ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_bufferSize(cusparseContext handle,
                        cusparseOperation opA,
                        CUdeviceptr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnVecDescr vecX,
                        cusparseDnVecDescr vecY,
                        cudaDataType computeType,
                        cusparseSpSVAlg alg,
                        cusparseSpSVDescr spsvDescr,
                        ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_analysis(cusparseContext handle,
                      cusparseOperation opA,
                      IntPtr alpha,
                      cusparseConstSpMatDescr matA,
                      cusparseConstDnVecDescr vecX,
                      cusparseDnVecDescr vecY,
                      cudaDataType computeType,
                      cusparseSpSVAlg alg,
                      cusparseSpSVDescr spsvDescr,
                      CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_analysis(cusparseContext handle,
                      cusparseOperation opA,
                      CUdeviceptr alpha,
                      cusparseConstSpMatDescr matA,
                      cusparseConstDnVecDescr vecX,
                      cusparseDnVecDescr vecY,
                      cudaDataType computeType,
                      cusparseSpSVAlg alg,
                      cusparseSpSVDescr spsvDescr,
                      CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_solve(cusparseContext handle,
                   cusparseOperation opA,
                   IntPtr alpha,
                   cusparseConstSpMatDescr matA,
                   cusparseConstDnVecDescr vecX,
                   cusparseDnVecDescr vecY,
                   cudaDataType computeType,
                   cusparseSpSVAlg alg,
                   cusparseSpSVDescr spsvDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_solve(cusparseContext handle,
                   cusparseOperation opA,
                   CUdeviceptr alpha,
                   cusparseConstSpMatDescr matA,
                   cusparseConstDnVecDescr vecX,
                   cusparseDnVecDescr vecY,
                   cudaDataType computeType,
                   cusparseSpSVAlg alg,
                   cusparseSpSVDescr spsvDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpSV_updateMatrix(cusparseContext handle,
                          cusparseSpSVDescr spsvDescr,
                          CUdeviceptr newValues,
                          cusparseSpSVUpdate updatePart);
        #endregion

        #region SPARSE MATRIX-MATRIX MULTIPLICATION
        // #############################################################################
        // # SPARSE MATRIX-MATRIX MULTIPLICATION
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMM(cusparseContext handle,
             cusparseOperation opA,
             cusparseOperation opB,

             IntPtr alpha,
             cusparseConstSpMatDescr matA,
             cusparseConstDnMatDescr matB,

             IntPtr beta,
             cusparseDnMatDescr matC,
             cudaDataType computeType,
             SpMMAlg alg,
             CUdeviceptr externalBuffer);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpMM_preprocess(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,
                        IntPtr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,
                        IntPtr beta,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        SpMMAlg alg,
                        CUdeviceptr externalBuffer);


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMM_bufferSize(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,

                        IntPtr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,

                        IntPtr beta,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        SpMMAlg alg,
                        ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMM(cusparseContext handle,
             cusparseOperation opA,
             cusparseOperation opB,

             CUdeviceptr alpha,
             cusparseConstSpMatDescr matA,
             cusparseConstDnMatDescr matB,

             CUdeviceptr beta,
             cusparseDnMatDescr matC,
             cudaDataType computeType,
             SpMMAlg alg,
             CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSpMM_preprocess(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,
                        CUdeviceptr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,
                        CUdeviceptr beta,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        SpMMAlg alg,
                        CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMM_bufferSize(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,

                        CUdeviceptr alpha,
                        cusparseConstSpMatDescr matA,
                        cusparseConstDnMatDescr matB,

                        CUdeviceptr beta,
                        cusparseDnMatDescr matC,
                        cudaDataType computeType,
                        SpMMAlg alg,
                        ref SizeT bufferSize);
        #endregion
        #region SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM) 
        // #############################################################################
        // # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM)
        // #############################################################################


        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_createDescr(ref cusparseSpGEMMDescr descr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr descr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_getNumProducts(cusparseSpGEMMDescr spgemmDescr,
                              ref long num_prods);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_estimateMemory(cusparseContext handle,
                              cusparseOperation opA,
                              cusparseOperation opB,
                              IntPtr alpha,
                              cusparseConstSpMatDescr matA,
                              cusparseConstSpMatDescr matB,
                              IntPtr beta,
                              cusparseSpMatDescr matC,
                              cudaDataType computeType,
                              cusparseSpGEMMAlg alg,
                              cusparseSpGEMMDescr spgemmDescr,
                              float chunk_fraction,
                              ref SizeT bufferSize3,
                              CUdeviceptr externalBuffer3,
                              ref SizeT bufferSize2);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_estimateMemory(cusparseContext handle,
                              cusparseOperation opA,
                              cusparseOperation opB,
                              CUdeviceptr alpha,
                              cusparseConstSpMatDescr matA,
                              cusparseConstSpMatDescr matB,
                              CUdeviceptr beta,
                              cusparseSpMatDescr matC,
                              cudaDataType computeType,
                              cusparseSpGEMMAlg alg,
                              cusparseSpGEMMDescr spgemmDescr,
                              float chunk_fraction,
                              ref SizeT bufferSize3,
                              CUdeviceptr externalBuffer3,
                              ref SizeT bufferSize2);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_workEstimation(cusparseContext handle,
                              cusparseOperation opA,
                              cusparseOperation opB,

                              IntPtr alpha,
                              cusparseConstSpMatDescr matA,
                              cusparseConstSpMatDescr matB,

                              IntPtr beta,
                              cusparseSpMatDescr matC,
                              cudaDataType computeType,
                              cusparseSpGEMMAlg alg,
                              cusparseSpGEMMDescr spgemmDescr,
                              ref SizeT bufferSize1,
                              CUdeviceptr externalBuffer1);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_compute(cusparseContext handle,
                       cusparseOperation opA,
                       cusparseOperation opB,

                       IntPtr alpha,
                       cusparseConstSpMatDescr matA,
                       cusparseConstSpMatDescr matB,

                       IntPtr beta,
                       cusparseSpMatDescr matC,
                       cudaDataType computeType,
                       cusparseSpGEMMAlg alg,
                       cusparseSpGEMMDescr spgemmDescr,
                       ref SizeT bufferSize2,
                       CUdeviceptr externalBuffer2);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_copy(cusparseContext handle,
                    cusparseOperation opA,
                    cusparseOperation opB,

                    IntPtr alpha,
                    cusparseConstSpMatDescr matA,
                    cusparseConstSpMatDescr matB,

                    IntPtr beta,
                    cusparseSpMatDescr matC,
                    cudaDataType computeType,
                    cusparseSpGEMMAlg alg,
                    cusparseSpGEMMDescr spgemmDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_workEstimation(cusparseContext handle,
                              cusparseOperation opA,
                              cusparseOperation opB,

                              CUdeviceptr alpha,
                              cusparseConstSpMatDescr matA,
                              cusparseConstSpMatDescr matB,

                              CUdeviceptr beta,
                              cusparseSpMatDescr matC,
                              cudaDataType computeType,
                              cusparseSpGEMMAlg alg,
                              cusparseSpGEMMDescr spgemmDescr,
                              ref SizeT bufferSize1,
                              CUdeviceptr externalBuffer1);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_compute(cusparseContext handle,
                       cusparseOperation opA,
                       cusparseOperation opB,

                       CUdeviceptr alpha,
                       cusparseConstSpMatDescr matA,
                       cusparseConstSpMatDescr matB,

                       CUdeviceptr beta,
                       cusparseSpMatDescr matC,
                       cudaDataType computeType,
                       cusparseSpGEMMAlg alg,
                       cusparseSpGEMMDescr spgemmDescr,
                       ref SizeT bufferSize2,
                       CUdeviceptr externalBuffer2);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMM_copy(cusparseContext handle,
                    cusparseOperation opA,
                    cusparseOperation opB,

                    CUdeviceptr alpha,
                    cusparseConstSpMatDescr matA,
                    cusparseConstSpMatDescr matB,

                    CUdeviceptr beta,
                    cusparseSpMatDescr matC,
                    cudaDataType computeType,
                    cusparseSpGEMMAlg alg,
                    cusparseSpGEMMDescr spgemmDescr);
        #endregion

        #region SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM) STRUCTURE REUSE

        // #############################################################################
        // # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM) STRUCTURE REUSE
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMMreuse_workEstimation(cusparseContext handle,
                                   cusparseOperation opA,
                                   cusparseOperation opB,
                                   cusparseConstSpMatDescr matA,
                                   cusparseConstSpMatDescr matB,
                                   cusparseSpMatDescr matC,
                                   cusparseSpGEMMAlg alg,
                                   cusparseSpGEMMDescr spgemmDescr,
                                   ref SizeT bufferSize1,
                                   CUdeviceptr externalBuffer1);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMMreuse_nnz(cusparseContext handle,
                        cusparseOperation opA,
                        cusparseOperation opB,
                        cusparseConstSpMatDescr matA,
                        cusparseConstSpMatDescr matB,
                        cusparseSpMatDescr matC,
                        cusparseSpGEMMAlg alg,
                        cusparseSpGEMMDescr spgemmDescr,
                        ref SizeT bufferSize2,
                        CUdeviceptr externalBuffer2,
                        ref SizeT bufferSize3,
                        CUdeviceptr externalBuffer3,
                        ref SizeT bufferSize4,
                        CUdeviceptr externalBuffer4);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMMreuse_copy(cusparseContext handle,
                         cusparseOperation opA,
                         cusparseOperation opB,
                         cusparseConstSpMatDescr matA,
                         cusparseConstSpMatDescr matB,
                         cusparseSpMatDescr matC,
                         cusparseSpGEMMAlg alg,
                         cusparseSpGEMMDescr spgemmDescr,
                         ref SizeT bufferSize5,
                         CUdeviceptr externalBuffer5);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMMreuse_compute(cusparseContext handle,
                            cusparseOperation opA,
                            cusparseOperation opB,
                            IntPtr alpha,
                            cusparseConstSpMatDescr matA,
                            cusparseConstSpMatDescr matB,
                            IntPtr beta,
                            cusparseSpMatDescr matC,
                            cudaDataType computeType,
                            cusparseSpGEMMAlg alg,
                            cusparseSpGEMMDescr spgemmDescr);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpGEMMreuse_compute(cusparseContext handle,
                            cusparseOperation opA,
                            cusparseOperation opB,
                            CUdeviceptr alpha,
                            cusparseConstSpMatDescr matA,
                            cusparseConstSpMatDescr matB,
                            CUdeviceptr beta,
                            cusparseSpMatDescr matC,
                            cudaDataType computeType,
                            cusparseSpGEMMAlg alg,
                            cusparseSpGEMMDescr spgemmDescr);

        #endregion

        #region GENERAL MATRIX-MATRIX PATTERN-CONSTRAINED MULTIPLICATION
        // #############################################################################
        // # GENERAL MATRIX-MATRIX PATTERN-CONSTRAINED MULTIPLICATION
        // #############################################################################

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSDDMM_bufferSize(cusparseContext handle,
                         cusparseOperation opA,
                         cusparseOperation opB,
                         IntPtr alpha,
                         cusparseConstDnMatDescr matA,
                         cusparseConstDnMatDescr matB,
                         IntPtr beta,
                         cusparseSpMatDescr matC,
                         cudaDataType computeType,
                         cusparseSDDMMAlg alg,
                         ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSDDMM_preprocess(cusparseContext handle,
                         cusparseOperation opA,
                         cusparseOperation opB,
                         IntPtr alpha,
                         cusparseConstDnMatDescr matA,
                         cusparseConstDnMatDescr matB,
                         IntPtr beta,
                         cusparseSpMatDescr matC,
                         cudaDataType computeType,
                         cusparseSDDMMAlg alg,
                         CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSDDMM(cusparseContext handle,
              cusparseOperation opA,
              cusparseOperation opB,
              IntPtr alpha,
              cusparseConstDnMatDescr matA,
              cusparseConstDnMatDescr matB,
              IntPtr beta,
              cusparseSpMatDescr matC,
              cudaDataType computeType,
              cusparseSDDMMAlg alg,
              CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSDDMM_bufferSize(cusparseContext handle,
                         cusparseOperation opA,
                         cusparseOperation opB,
                         CUdeviceptr alpha,
                         cusparseConstDnMatDescr matA,
                         cusparseConstDnMatDescr matB,
                         CUdeviceptr beta,
                         cusparseSpMatDescr matC,
                         cudaDataType computeType,
                         cusparseSDDMMAlg alg,
                         ref SizeT bufferSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSDDMM_preprocess(cusparseContext handle,
                         cusparseOperation opA,
                         cusparseOperation opB,
                         CUdeviceptr alpha,
                         cusparseConstDnMatDescr matA,
                         cusparseConstDnMatDescr matB,
                         CUdeviceptr beta,
                         cusparseSpMatDescr matC,
                         cudaDataType computeType,
                         cusparseSDDMMAlg alg,
                         CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus
        cusparseSDDMM(cusparseContext handle,
              cusparseOperation opA,
              cusparseOperation opB,
              CUdeviceptr alpha,
              cusparseConstDnMatDescr matA,
              cusparseConstDnMatDescr matB,
              CUdeviceptr beta,
              cusparseSpMatDescr matC,
              cudaDataType computeType,
              cusparseSDDMMAlg alg,
              CUdeviceptr externalBuffer);
        #endregion

        #region GENERIC APIs WITH CUSTOM OPERATORS (PREVIEW)



        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMMOp_createPlan(cusparseContext handle,
                          ref cusparseSpMMOpPlan plan,
                          cusparseOperation opA,
                          cusparseOperation opB,
                          cusparseConstSpMatDescr matA,
                          cusparseConstDnMatDescr matB,
                          cusparseDnMatDescr matC,
                          cudaDataType computeType,
                          cusparseSpMMOpAlg alg,
                          byte[] addOperationNvvmBuffer,
                          SizeT addOperationBufferSize,
                          byte[] mulOperationNvvmBuffer,
                          SizeT mulOperationBufferSize,
                          byte[] epilogueNvvmBuffer,
                          SizeT epilogueBufferSize,
                          ref SizeT SpMMWorkspaceSize);
        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMMOp_createPlan(cusparseContext handle,
                          ref cusparseSpMMOpPlan plan,
                          cusparseOperation opA,
                          cusparseOperation opB,
                          cusparseConstSpMatDescr matA,
                          cusparseConstDnMatDescr matB,
                          cusparseDnMatDescr matC,
                          cudaDataType computeType,
                          cusparseSpMMOpAlg alg,
                          IntPtr addOperationNvvmBuffer,
                          SizeT addOperationBufferSize,
                          IntPtr mulOperationNvvmBuffer,
                          SizeT mulOperationBufferSize,
                          IntPtr epilogueNvvmBuffer,
                          SizeT epilogueBufferSize,
                          ref SizeT SpMMWorkspaceSize);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMMOp(cusparseSpMMOpPlan plan,
               CUdeviceptr externalBuffer);

        [DllImport(CUSPARSE_API_DLL_NAME)]
        public static extern cusparseStatus cusparseSpMMOp_destroyPlan(cusparseSpMMOpPlan plan);

        #endregion
        #endregion
    }
}
