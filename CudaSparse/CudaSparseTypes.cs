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
using System.Runtime.InteropServices;

namespace ManagedCuda.CudaSparse
{
    #region Enums
    /// <summary>
    /// This is a status type returned by the library functions and it can have the following values.
    /// </summary>
    public enum cusparseStatus
    {
        /// <summary>
        /// The operation completed successfully.
        /// </summary>
        Success = 0,
        /// <summary>
        /// "The CUSPARSE library was not initialized. This is usually caused by the lack of a prior 
        /// cusparseCreate() call, an error in the CUDA Runtime API called by the CUSPARSE routine, or an 
        /// error in the hardware setup. To correct: call cusparseCreate() prior to the function call; and
        ///  check that the hardware, an appropriate version of the driver, and the CUSPARSE library are 
        /// correctly installed.
        /// </summary>
        NotInitialized = 1,
        /// <summary>
        ///  "Resource allocation failed inside the CUSPARSE library. This is usually caused by a 
        /// cudaMalloc() failure. To correct: prior to the function call, deallocate previously allocated
        /// memory as much as possible.
        /// </summary>
        AllocFailed = 2,
        /// <summary>
        /// "An unsupported value or parameter was passed to the function (a negative vector size, 
        /// for example). To correct: ensure that all the parameters being passed have valid values.
        /// </summary>
        InvalidValue = 3,
        /// <summary>
        /// "The function requires a feature absent from the device architecture; usually caused by 
        /// the lack of support for atomic operations or double precision. To correct: compile and run the
        ///  application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic 
        /// operations and 1.3 for double precision.
        /// </summary>
        ArchMismatch = 4,
        /// <summary>
        /// "An access to GPU memory space failed, which is usually caused by a failure to bind a texture. 
        /// To correct: prior to the function call, unbind any previously bound textures.
        /// </summary>
        MappingError = 5,
        /// <summary>
        /// "The GPU program failed to execute. This is often caused by a launch failure of the kernel on 
        /// the GPU, which can be caused by multiple reasons. To correct: check that the hardware, an appropriate
        ///  version of the driver, and the CUSPARSE library are correctly installed.
        /// </summary>
        ExecutionFailed = 6,
        /// <summary>
        /// "An internal CUSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() 
        /// failure. To correct: check that the hardware, an appropriate version of the driver, and the CUSPARSE
        ///  library are correctly installed. Also, check that the memory passed as a parameter to the routine 
        /// is not being deallocated prior to the routine’s completion.
        /// </summary>
        InternalError = 7,
        /// <summary>
        /// "The matrix type is not supported by this function. This is usually caused by passing an invalid 
        /// matrix descriptor to the function. To correct: check that the fields in cusparseMatDescr_t descrA were 
        /// set correctly.
        /// </summary>
        MatrixTypeNotSupported = 8,
        /// <summary>
        ///
        /// </summary>
        ZeroPivot = 9,
        /// <summary>
        /// 
        /// </summary>
        NotSupported = 10,
        /// <summary>
        /// 
        /// </summary>
        InsufficientResources = 11
    }

    /// <summary>
    /// This type indicates whether the scalar values are passed by reference on the host or device.
    /// It is important to point out that if several scalar values are passed by reference in the
    /// function call, all of them will conform to the same single pointer mode. The pointer mode
    /// can be set and retrieved using <see cref="CudaSparseContext.SetPointerMode"/> and
    /// <see cref="CudaSparseContext.GetPointerMode()"/> routines, respectively.
    /// </summary>
    public enum cusparsePointerMode
    {
        /// <summary>
        /// Use host pointers.
        /// </summary>
        Host = 0,
        /// <summary>
        /// Use device pointers.
        /// </summary>
        Device = 1
    }

    /// <summary>
    /// This type indicates whether the operation is performed only on indices or on data and indices.
    /// </summary>
    public enum cusparseAction
    {
        /// <summary>
        /// the operation is performed only on indices.
        /// </summary>
        Symbolic = 0,
        /// <summary>
        /// the operation is performed on data and indices.
        /// </summary>
        Numeric = 1
    }

    /// <summary>
    /// This type indicates the type of matrix stored in sparse storage. Notice that for symmetric,
    /// Hermitian and triangular matrices only their lower or upper part is assumed to be stored.
    /// </summary>
    public enum cusparseMatrixType
    {
        /// <summary>
        /// the matrix is general.
        /// </summary>
        General = 0,
        /// <summary>
        /// the matrix is symmetric.
        /// </summary>
        Symmetric = 1,
        /// <summary>
        /// the matrix is Hermitian.
        /// </summary>
        Hermitian = 2,
        /// <summary>
        /// the matrix is triangular.
        /// </summary>
        Triangular = 3
    }

    /// <summary>
    /// This type indicates if the lower or upper part of a matrix is stored in sparse storage.
    /// </summary>
    public enum cusparseFillMode
    {
        /// <summary>
        /// the lower triangular part is stored.
        /// </summary>
        Lower = 0,
        /// <summary>
        /// the upper triangular part is stored.
        /// </summary>
        Upper = 1
    }

    /// <summary>
    /// This type indicates if the matrix diagonal entries are unity. The diagonal elements are
    /// always assumed to be present, but if CUSPARSE_DIAG_TYPE_UNIT is passed to an API
    /// routine, then the routine will assume that all diagonal entries are unity and will not read
    /// or modify those entries. Note that in this case the routine assumes the diagonal entries are
    /// equal to one, regardless of what those entries are actuall set to in memory.
    /// </summary>
    public enum cusparseDiagType
    {
        /// <summary>
        /// the matrix diagonal has non-unit elements.
        /// </summary>
        NonUnit = 0,
        /// <summary>
        /// the matrix diagonal has unit elements.
        /// </summary>
        Unit = 1
    }

    /// <summary>
    /// This type indicates if the base of the matrix indices is zero or one.
    /// </summary>
    public enum IndexBase
    {
        /// <summary>
        /// the base index is zero.
        /// </summary>
        Zero = 0,
        /// <summary>
        /// the base index is one.
        /// </summary>
        One = 1
    }

    /// <summary>
    /// This type indicates which operations need to be performed with the sparse matrix.
    /// </summary>
    public enum cusparseOperation
    {
        /// <summary>
        /// the non-transpose operation is selected.
        /// </summary>
        NonTranspose = 0,
        /// <summary>
        /// the transpose operation is selected.
        /// </summary>
        Transpose = 1,
        /// <summary>
        /// the conjugate transpose operation is selected.
        /// </summary>
        ConjugateTranspose = 2
    }

    /// <summary>
    /// This type indicates whether the elements of a dense matrix should be parsed by rows or by
    /// columns (assuming column-major storage in memory of the dense matrix).
    /// </summary>
    public enum cusparseDirection
    {
        /// <summary>
        /// the matrix should be parsed by rows.
        /// </summary>
        Row = 0,
        /// <summary>
        /// the matrix should be parsed by columns.
        /// </summary>
        Column = 1
    }


    /// <summary>
    /// used in csrsv2, csric02, and csrilu02
    /// </summary>
    [Obsolete("Deprecated in Cuda 12.3")]
    public enum cusparseSolvePolicy
    {
        /// <summary>
        /// no level information is generated, only reports structural zero.
        /// </summary>
        NoLevel = 0,
        /// <summary>
        /// 
        /// </summary>
        UseLevel = 1
    }

    /// <summary>
    /// 
    /// </summary>
    [Obsolete("Deprecated in Cuda 12.3")]
    public enum cusparseColorAlg
    {
        /// <summary>
        /// default
        /// </summary>
        ALG0 = 0,
        /// <summary>
        /// 
        /// </summary>
        ALG1 = 1
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cusparseCsr2CscAlg
    {
        /// <summary>
        /// </summary>
        CSR2CSC_ALG_DEFAULT = 1,
        /// <summary>
        /// </summary>
        CSR2CSC_ALG1 = 1
    }

    /// <summary>
    /// Index type
    /// </summary>
    public enum IndexType
    {
        /// <summary>
        /// 16-bit unsigned integer for matrix/vector indices
        /// </summary>
        Index16U = 1,
        /// <summary>
        /// 32-bit signed integer for matrix/vector indices
        /// </summary>
        Index32I = 2,
        /// <summary>
        /// 64-bit signed integer for matrix/vector indices
        /// </summary>
        Index64I = 3
    }

    /// <summary>
    /// 
    /// </summary>
    public enum Format
    {
        /// <summary>
        /// Compressed Sparse Row (CSR)
        /// </summary>
        CSR = 1,
        /// <summary>
        /// Compressed Sparse Column (CSC)
        /// </summary>
        CSC = 2,
        /// <summary>
        /// Coordinate (COO) - Structure of Arrays
        /// </summary>
        COO = 3,
        /// <summary>
        /// Blocked ELL
        /// </summary>
        BLOCKED_ELL = 5,
        /// <summary>
        /// Blocked Compressed Sparse Row (BSR)
        /// </summary>
        BSR = 6,
        /// <summary>
        /// Sliced ELL
        /// </summary>
        SLICED_ELLPACK = 7
    }

    /// <summary>
    /// 
    /// </summary>
    public enum Order
    {
        /// <summary>
        /// Column-Major Order - Matrix memory layout
        /// </summary>
        Col = 1,
        /// <summary>
        /// Row-Major Order - Matrix memory layout
        /// </summary>
        Row = 2
    }

    /// <summary>
    /// 
    /// </summary>
    public enum SpMVAlg
    {
        CUSPARSE_SPMV_ALG_DEFAULT = 0,
        CUSPARSE_SPMV_CSR_ALG1 = 2,
        CUSPARSE_SPMV_CSR_ALG2 = 3,
        CUSPARSE_SPMV_COO_ALG1 = 1,
        CUSPARSE_SPMV_COO_ALG2 = 4,
        CUSPARSE_SPMV_SELL_ALG1 = 5
    }

    /// <summary>
    /// 
    /// </summary>
    public enum SpMMAlg
    {
        /// <summary/>
        CUSPARSE_SPMM_ALG_DEFAULT = 0,
        /// <summary/>
        CUSPARSE_SPMM_COO_ALG1 = 1,
        /// <summary/>
        CUSPARSE_SPMM_COO_ALG2 = 2,
        /// <summary/>
        CUSPARSE_SPMM_COO_ALG3 = 3,
        /// <summary/>
        CUSPARSE_SPMM_COO_ALG4 = 5,
        /// <summary/>
        CUSPARSE_SPMM_CSR_ALG1 = 4,
        /// <summary/>
        CUSPARSE_SPMM_CSR_ALG2 = 6,
        /// <summary/>
        CUSPARSE_SPMM_CSR_ALG3 = 12,
        /// <summary/>
        CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cusparseSpGEMMAlg
    {
        /// <summary>
        /// 
        /// </summary>
        Default = 0,
        CsrAlgDeterministic = 1,
        CsrAlgNonDeterministic = 2,
        Alg1 = 3,
        Alg2 = 4,
        Alg3 = 5
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cusparseSparseToDenseAlg
    {
        /// <summary>
        /// 
        /// </summary>
        Default = 0
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cusparseDenseToSparseAlg
    {
        /// <summary>
        /// 
        /// </summary>
        Default = 0
    }
    /// <summary>
    /// 
    /// </summary>
    public enum cusparseSpMatAttribute
    {
        /// <summary>
        /// 
        /// </summary>
        FillMode,
        /// <summary>
        /// 
        /// </summary>
        DiagType
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cusparseSpSVAlg
    {
        /// <summary>
        /// 
        /// </summary>
        Default = 0,
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cusparseSpSVUpdate
    {
        General = 0,
        Diagonal = 1
    }

    public enum cusparseSDDMMAlg
    {
        /// <summary>
        /// 
        /// </summary>
        Default = 0,
    }

    public enum cusparseSpSMAlg
    {
        Default = 0,
    }

    public enum cusparseSpMMOpAlg
    {
        Default = 0,
    }
    #endregion

    #region structs (opaque handles)
    /// <summary>
    /// Opaque structure holding CUSPARSE library context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseContext
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the matrix descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseMatDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the sparse triangular solve information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct bsrsv2Info
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the sparse triangular solve information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct csric02Info
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the sparse triangular solve information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct bsric02Info
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the sparse triangular solve information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct bsrsm2Info
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the sparse triangular solve information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct csrilu02Info
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the sparse triangular solve information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct bsrilu02Info
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the sorting information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct csru2csrInfo
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the coloring information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct cusparseColorInfo
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// Opaque structure holding the prune information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Obsolete("Deprecated in Cuda 12.3")]
    public struct pruneInfo
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSpVecDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
        public static implicit operator cusparseConstSpVecDescr(cusparseSpVecDescr descr)
        {
            cusparseConstSpVecDescr ret = new cusparseConstSpVecDescr();
            ret.Handle = descr.Handle;
            return ret;
        }
    }
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseDnVecDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
        public static implicit operator cusparseConstDnVecDescr(cusparseDnVecDescr descr)
        {
            cusparseConstDnVecDescr ret = new cusparseConstDnVecDescr();
            ret.Handle = descr.Handle;
            return ret;
        }
    }
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSpMatDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
        public static implicit operator cusparseConstSpMatDescr(cusparseSpMatDescr descr)
        {
            cusparseConstSpMatDescr ret = new cusparseConstSpMatDescr();
            ret.Handle = descr.Handle;
            return ret;
        }
    }
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseDnMatDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
        public static implicit operator cusparseConstDnMatDescr(cusparseDnMatDescr descr)
        {
            cusparseConstDnMatDescr ret = new cusparseConstDnMatDescr();
            ret.Handle = descr.Handle;
            return ret;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseConstSpVecDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseConstDnVecDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseConstSpMatDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseConstDnMatDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSpGEMMDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }


    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSpSVDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }


    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSpSMDescr
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusparseSpMMOpPlan
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Handle;
    }
    #endregion
}
