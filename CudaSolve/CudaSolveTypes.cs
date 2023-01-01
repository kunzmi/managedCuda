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


namespace ManagedCuda.CudaSolve
{
    #region common
    /// <summary>
    /// This is a status type returned by the library functions and it can have the following values.
    /// </summary>
    public enum cusolverStatus
    {
        /// <summary>
        /// The operation completed successfully
        /// </summary>
        Success = 0,
        /// <summary>
        /// The cuSolver library was not initialized. This is usually caused by the
        /// lack of a prior call, an error in the CUDA Runtime API called by the
        /// cuSolver routine, or an error in the hardware setup.<para/>
        /// To correct: call cusolverCreate() prior to the function call; and
        /// check that the hardware, an appropriate version of the driver, and the
        /// cuSolver library are correctly installed.
        /// </summary>
        NotInititialized = 1,
        /// <summary>
        /// Resource allocation failed inside the cuSolver library. This is usually
        /// caused by a cudaMalloc() failure.<para/>
        /// To correct: prior to the function call, deallocate previously allocated
        /// memory as much as possible.
        /// </summary>
        AllocFailed = 2,
        /// <summary>
        /// An unsupported value or parameter was passed to the function (a
        /// negative vector size, for example).<para/>
        /// To correct: ensure that all the parameters being passed have valid
        /// values.
        /// </summary>
        InvalidValue = 3,
        /// <summary>
        /// The function requires a feature absent from the device architecture;
        /// usually caused by the lack of support for atomic operations or double
        /// precision.<para/>
        /// To correct: compile and run the application on a device with compute
        /// capability 2.0 or above.
        /// </summary>
        ArchMismatch = 4,
        /// <summary>
        /// 
        /// </summary>
        MappingError = 5,
        /// <summary>
        /// The GPU program failed to execute. This is often caused by a launch
        /// failure of the kernel on the GPU, which can be caused by multiple
        /// reasons.<para/>
        /// To correct: check that the hardware, an appropriate version of the
        /// driver, and the cuSolver library are correctly installed.
        /// </summary>
        ExecutionFailed = 6,
        /// <summary>
        /// An internal cuSolver operation failed. This error is usually caused by a
        /// cudaMemcpyAsync() failure.<para/>
        /// To correct: check that the hardware, an appropriate version of the
        /// driver, and the cuSolver library are correctly installed. Also, check
        /// that the memory passed as a parameter to the routine is not being
        /// deallocated prior to the routine’s completion.
        /// </summary>
        InternalError = 7,
        /// <summary>
        /// The matrix type is not supported by this function. This is usually caused
        /// by passing an invalid matrix descriptor to the function.<para/>
        /// To correct: check that the fields in descrA were set correctly.
        /// </summary>
        MatrixTypeNotSupported = 8,
        /// <summary>
        /// 
        /// </summary>
        NotSupported = 9,
        /// <summary>
        /// 
        /// </summary>
        ZeroPivot = 10,
        /// <summary>
        /// 
        /// </summary>
        InvalidLicense = 11,
        CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID = 13,

        CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16,
        CUSOLVER_STATUS_IRS_INTERNAL_ERROR = 20,
        CUSOLVER_STATUS_IRS_NOT_SUPPORTED = 21,
        CUSOLVER_STATUS_IRS_OUT_OF_RANGE = 22,
        CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23,
        CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25,
        CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26,
        CUSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30,
        CUSOLVER_STATUS_INVALID_WORKSPACE = 31
    }


    /// <summary>
    /// </summary>
    public enum cusolverEigType
    {
        /// <summary/>
        Type1 = 1,
        /// <summary/>
        Type2 = 2,
        /// <summary/>
        Type3 = 3
    }

    /// <summary>
    /// </summary>
    public enum cusolverEigMode
    {
        /// <summary/>
        NoVector = 0,
        /// <summary/>
        Vector = 1
    }


    /// <summary>
    /// </summary>
    public enum cusolverEigRange
    {
        All = 1001,
        I = 1002,
        V = 1003,
    }



    /// <summary>
    /// </summary>
    public enum cusolverNorm
    {
        InfNorm = 104,
        MaxNorm = 105,
        OneNorm = 106,
        FroNorm = 107,
    }


    /// <summary>
    /// </summary>
    public enum cusolverIRSRefinement
    {
        NotSet = 1100,
        None = 1101,
        Classical = 1102,
        Classical_GMRES = 1103,
        GMRES = 1104,
        GMRES_GMRES = 1105,
        GMRES_NOPCOND = 1106,

        PrecDD = 1150,
        PrecSS = 1151,
        PrecSHT = 1152,
    }

    public enum cusolverPrecType
    {
        CUSOLVER_R_8I = 1201,
        CUSOLVER_R_8U = 1202,
        CUSOLVER_R_64F = 1203,
        CUSOLVER_R_32F = 1204,
        CUSOLVER_R_16F = 1205,
        CUSOLVER_R_16BF = 1206,
        CUSOLVER_R_TF32 = 1207,
        CUSOLVER_R_AP = 1208,
        CUSOLVER_C_8I = 1211,
        CUSOLVER_C_8U = 1212,
        CUSOLVER_C_64F = 1213,
        CUSOLVER_C_32F = 1214,
        CUSOLVER_C_16F = 1215,
        CUSOLVER_C_16BF = 1216,
        CUSOLVER_C_TF32 = 1217,
        CUSOLVER_C_AP = 1218,
    }

    public enum cusolverAlgMode
    {
        CUSOLVER_ALG_0 = 0,  /* default algorithm */
        CUSOLVER_ALG_1 = 1,
        CUSOLVER_ALG_2 = 2
    }


    public enum cusolverStorevMode
    {
        CUBLAS_STOREV_COLUMNWISE = 0,
        CUBLAS_STOREV_ROWWISE = 1
    }

    public enum cusolverDirectMode
    {
        CUBLAS_DIRECT_FORWARD = 0,
        CUBLAS_DIRECT_BACKWARD = 1
    }

    #endregion

    #region Dense
    /// <summary>
    /// This is a pointer type to an opaque cuSolverDN context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusolverDnHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct syevjInfo
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }


    /// <summary>
    ///
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct gesvdjInfo
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }


    /// <summary>
    /// opaque cusolverDnIRS structure for IRS solver
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusolverDnIRSParams
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }


    /// <summary>
    /// opaque cusolverDnIRS structure for IRS solver
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusolverDnIRSInfos
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusolverDnParams
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    public enum cusolverDnFunction
    {
        CUSOLVERDN_GETRF = 0,
        CUSOLVERDN_POTRF = 1
    }
    #endregion

    #region Sparse
    /// <summary>
    /// This is a pointer type to an opaque cuSolverSP context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusolverSpHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// This is a pointer type to an opaque csrqrInfo
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct csrqrInfo
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }
    #endregion

    #region Refactorization
    /// <summary>
    /// This is a pointer type to an opaque cuSolverRF context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cusolverRfHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }


    /// <summary>
    /// The ResetValuesFastMode is an enum that indicates the mode used for
    /// the cusolverRfResetValues() routine. The fast mode requires extra memory and is
    /// recommended only if very fast calls to cusolverRfResetValues() are needed.
    /// </summary>
    public enum ResetValuesFastMode
    {
        /// <summary>
        /// default
        /// </summary>
        Off = 0,
        /// <summary/>  
        On = 1
    }

    /// <summary>
    /// The MatrixFormat is an enum that indicates the input/output
    /// matrix format assumed by the cusolverRfSetup(), cusolverRfSetupHost(),
    /// cusolverRfResetValues(), cusolveRfExtractBundledFactorsHost() and
    /// cusolverRfExtractSplitFactorsHost() routines.
    /// </summary>
    public enum MatrixFormat
    {
        /// <summary>
        /// default
        /// </summary>
        Csr = 0,
        /// <summary/>
        Csc = 1
    }

    /// <summary>
    /// The UnitDiagonal is an enum that indicates whether
    /// and where the unit diagonal is stored in the input/output triangular
    /// factors in the cusolverRfSetup(), cusolverRfSetupHost() and
    /// cusolverRfExtractSplitFactorsHost() routines.
    /// </summary>
    public enum UnitDiagonal
    {
        /// <summary>
        /// unit diagonal is stored in lower triangular factor. (default)
        /// </summary>
        StoredL = 0,
        /// <summary>
        /// unit diagonal is stored in upper triangular factor.
        /// </summary>
        StoredU = 1,
        /// <summary>
        /// unit diagonal is assumed in lower triangular factor.
        /// </summary>
        AssumedL = 2,
        /// <summary>
        /// unit diagonal is assumed in upper triangular factor.
        /// </summary>
        AssumedU = 3
    }

    /// <summary>
    /// The Factorization is an enum that indicates which (internal)
    /// algorithm is used for refactorization in the cusolverRfRefactor() routine.
    /// </summary>
    public enum Factorization
    {
        /// <summary>
        /// algorithm 0. (default)
        /// </summary>
        Alg0 = 0,
        /// <summary>
        /// algorithm 1.
        /// </summary>
        Alg1 = 1,
        /// <summary>
        /// algorithm 2. Domino-based scheme.
        /// </summary>
        Alg2 = 2,
    }

    /// <summary>
    /// The TriangularSolve is an enum that indicates which (internal)
    /// algorithm is used for triangular solve in the cusolverRfSolve() routine.
    /// </summary>
    public enum TriangularSolve
    {
        ///// <summary>
        ///// algorithm 0.
        ///// </summary>
        //Alg0 = 0,
        /// <summary>
        /// algorithm 1. (default)
        /// </summary>
        Alg1 = 1,
        /// <summary>
        /// algorithm 2. Domino-based scheme.
        /// </summary>
        Alg2 = 2,
        /// <summary>
        /// algorithm 3. Domino-based scheme.
        /// </summary>
        Alg3 = 3
    }

    /// <summary>
    /// The cusolverRfNumericBoostReport_t is an enum that indicates whether
    /// numeric boosting (of the pivot) was used during the cusolverRfRefactor() and
    /// cusolverRfSolve() routines. The numeric boosting is disabled by default.
    /// </summary>
    public enum NumericBoostReport
    {
        /// <summary>
        /// default
        /// </summary>
        NotUsed = 0,
        /// <summary/>
        Used = 1
    }
    #endregion
}
