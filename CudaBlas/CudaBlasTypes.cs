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

namespace ManagedCuda.CudaBlas
{
    /// <summary>
    /// CUBLAS status type returns
    /// </summary>
    public enum CublasStatus
    {
        /// <summary>
        /// 
        /// </summary>
        Success = 0,
        /// <summary>
        /// 
        /// </summary>
        NotInitialized = 1,
        /// <summary>
        /// 
        /// </summary>
        AllocFailed = 3,
        /// <summary>
        /// 
        /// </summary>
        InvalidValue = 7,
        /// <summary>
        /// 
        /// </summary>
        ArchMismatch = 8,
        /// <summary>
        /// 
        /// </summary>
        MappingError = 11,
        /// <summary>
        /// 
        /// </summary>
        ExecutionFailed = 13,
        /// <summary>
        /// 
        /// </summary>
        InternalError = 14,
        /// <summary>
        /// 
        /// </summary>
        NotSupported = 15,
        /// <summary>
        /// 
        /// </summary>
        LicenseError = 16
    }

    /// <summary>
    /// The FillMode type indicates which part (lower or upper) of the dense matrix was
    /// filled and consequently should be used by the function. Its values correspond to Fortran
    /// characters ‘L’ or ‘l’ (lower) and ‘U’ or ‘u’ (upper) that are often used as parameters to
    /// legacy BLAS implementations.
    /// </summary>
    public enum FillMode
    {
        /// <summary>
        /// the lower part of the matrix is filled
        /// </summary>
        Lower = 0,
        /// <summary>
        /// the upper part of the matrix is filled
        /// </summary>
        Upper = 1,
        /// <summary>
        /// Full
        /// </summary>
        Full = 2
    }

    /// <summary>
    /// The DiagType type indicates whether the main diagonal of the dense matrix is
    /// unity and consequently should not be touched or modified by the function. Its values
    /// correspond to Fortran characters ‘N’ or ‘n’ (non-unit) and ‘U’ or ‘u’ (unit) that are
    /// often used as parameters to legacy BLAS implementations.
    /// </summary>
    public enum DiagType
    {
        /// <summary>
        /// the matrix diagonal has non-unit elements
        /// </summary>
        NonUnit = 0,
        /// <summary>
        /// the matrix diagonal has unit elements
        /// </summary>
        Unit = 1
    }

    /// <summary>
    /// The SideMode type indicates whether the dense matrix is on the left or right side
    /// in the matrix equation solved by a particular function. Its values correspond to Fortran
    /// characters ‘L’ or ‘l’ (left) and ‘R’ or ‘r’ (right) that are often used as parameters to
    /// legacy BLAS implementations.
    /// </summary>
    public enum SideMode
    {
        /// <summary>
        /// the matrix is on the left side in the equation
        /// </summary>
        Left = 0,
        /// <summary>
        /// the matrix is on the right side in the equation
        /// </summary>
        Right = 1
    }

    /// <summary>
    /// The Operation type indicates which operation needs to be performed with the
    /// dense matrix. Its values correspond to Fortran characters ‘N’ or ‘n’ (non-transpose), ‘T’
    /// or ‘t’ (transpose) and ‘C’ or ‘c’ (conjugate transpose) that are often used as parameters
    /// to legacy BLAS implementations
    /// </summary>
    public enum Operation
    {
        /// <summary>
        /// the non-transpose operation is selected
        /// </summary>
        NonTranspose = 0,
        /// <summary>
        /// the transpose operation is selected
        /// </summary>
        Transpose = 1,
        /// <summary>
        /// the conjugate transpose operation is selected
        /// </summary>
        ConjugateTranspose = 2,
        /// <summary>
        /// synonym of ConjugateTranspose
        /// </summary>
        Hermitan = 2,
        /// <summary>
        /// the conjugate operation is selected
        /// </summary>
        Conjugate = 3

    }

    /// <summary>
    /// The PointerMode type indicates whether the scalar values are passed by
    /// reference on the host or device. It is important to point out that if several scalar values are
    /// present in the function call, all of them must conform to the same single pointer mode.
    /// The pointer mode can be set and retrieved using cublasSetPointerMode() and
    /// cublasGetPointerMode() routines, respectively.
    /// </summary>
    public enum PointerMode
    {
        /// <summary>
        /// the scalars are passed by reference on the host
        /// </summary>
        Host = 0,
        /// <summary>
        /// the scalars are passed by reference on the device
        /// </summary>
        Device = 1
    }

    /// <summary>
    /// The type indicates whether cuBLAS routines which has an alternate implementation
    /// using atomics can be used. The atomics mode can be set and queried using and routines,
    /// respectively.
    /// </summary>
    public enum AtomicsMode
    {
        /// <summary>
        /// the usage of atomics is not allowed
        /// </summary>
        NotAllowed = 0,
        /// <summary>
        /// the usage of atomics is allowed
        /// </summary>
        Allowed = 1
    }

    /// <summary>
    /// For different GEMM algorithm
    /// </summary>
    public enum GemmAlgo
    {
        /// <summary>
        /// </summary>
        Default = -1,
        /// <summary>
        /// </summary>
        Algo0 = 0,
        /// <summary>
        /// </summary>
        Algo1 = 1,
        /// <summary>
        /// </summary>
        Algo2 = 2,
        /// <summary>
        /// </summary>
        Algo3 = 3,
        /// <summary>
        /// </summary>
        Algo4 = 4,
        /// <summary>
        /// </summary>
        Algo5 = 5,
        /// <summary>
        /// </summary>
        Algo6 = 6,
        /// <summary>
        /// </summary>
        Algo7 = 7,
        /// <summary>
        /// </summary>
        Algo8 = 8,
        /// <summary>
        /// </summary>
        Algo9 = 9,
        /// <summary>
        /// </summary>
        Algo10 = 10,
        /// <summary>
        /// </summary>
        Algo11 = 11,
        /// <summary>
        /// </summary>
        Algo12 = 12,
        /// <summary>
        /// </summary>
        Algo13 = 13,
        /// <summary>
        /// </summary>
        Algo14 = 14,
        /// <summary>
        /// </summary>
        Algo15 = 15,
        /// <summary>
        /// </summary>
        Algo16 = 16,
        /// <summary>
        /// </summary>
        Algo17 = 17,
        /// <summary>
        /// sliced 32x32  
        /// </summary>
        Algo18 = 18,
        /// <summary>
        /// sliced 64x32
        /// </summary>  
        Algo19 = 19,
        /// <summary>
        /// sliced 128x32
        /// </summary>  
        Algo20 = 20,
        /// <summary>
        /// sliced 32x32  -splitK
        /// </summary>   
        Algo21 = 21,
        /// <summary>
        /// sliced 64x32  -splitK
        /// </summary>      
        Algo22 = 22,
        /// <summary>
        /// sliced 128x32 -splitK 
        /// </summary>    
        Algo23 = 23, //     
        /// <summary>
        /// </summary>
        DefaultTensorOp = 99,
        /// <summary>
        /// </summary>
        Algo0TensorOp = 100,
        /// <summary>
        /// </summary>
        Algo1TensorOp = 101,
        /// <summary>
        /// </summary>
        Algo2TensorOp = 102,
        /// <summary>
        /// </summary>
        Algo3TensorOp = 103,
        /// <summary>
        /// </summary>
        Algo4TensorOp = 104,
        /// <summary>
        /// </summary>
        Algo5TensorOp = 105,
        /// <summary>
        /// </summary>
        Algo6TensorOp = 106,
        /// <summary>
        /// </summary>
        Algo7TensorOp = 107,
        /// <summary>
        /// </summary>
        Algo8TensorOp = 108,
        /// <summary>
        /// </summary>
        Algo9TensorOp = 109,
        /// <summary>
        /// </summary>
        Algo10TensorOp = 110,
        /// <summary>
        /// </summary>
        Algo11TensorOp = 111,
        /// <summary>
        /// </summary>
        Algo12TensorOp = 112,
        /// <summary>
        /// </summary>
        Algo13TensorOp = 113,
        /// <summary>
        /// </summary>
        Algo14TensorOp = 114,
        /// <summary>
        /// </summary>
        Algo15TensorOp = 115
    }

    /// <summary>
    /// Enum for default math mode/tensor operation
    /// </summary>
    public enum Math
    {
        /// <summary>
        /// </summary>
        DefaultMath = 0,
        /// <summary>
        /// </summary>
		[Obsolete("deprecated, same effect as using CUBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release")]
        TensorOpMath = 1,
        /// <summary>
        /// same as using matching _PEDANTIC compute type when using cublas routine calls or cublasEx() calls with cudaDataType as compute type
        /// </summary>
        PedanticMath = 2,
        /// <summary>
        /// allow accelerating single precision routines using TF32 tensor cores
        /// </summary>
        TF32TensorOpMath = 3,
        /// <summary>
        /// flag to force any reductons to use the accumulator type and not output type in case of mixed precision routines with lower size output type
        /// </summary>
        DisallowReducedPrecisionReduction = 16
    }

    /// <summary>
    /// Enum for compute type<para/>
    /// - default types provide best available performance using all available hardware features
    ///   and guarantee internal storage precision with at least the same precision and range;<para/>
    /// - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;<para/>
    /// - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
    /// </summary>
    public enum ComputeType
    {
        /// <summary>
        /// half - default
        /// </summary>
        Compute16F = 64,
        /// <summary>
        /// half - pedantic
        /// </summary>
        Compute16FPedantic = 65,
        /// <summary>
        /// float - default
        /// </summary>
        Compute32F = 68,
        /// <summary>
        /// float - pedantic
        /// </summary>
        Compute32FPedantic = 69,
        /// <summary>
        /// float - fast, allows down-converting inputs to half or TF32
        /// </summary>
        Compute32FFast16F = 74,
        /// <summary>
        /// float - fast, allows down-converting inputs to bfloat16 or TF32
        /// </summary>
        Compute32FFast16BF = 75,
        /// <summary>
        /// float - fast, allows down-converting inputs to TF32
        /// </summary>
        Compute32FFastTF32 = 77,
        /// <summary>
        /// double - default
        /// </summary>
        Compute64F = 70,
        /// <summary>
        /// double - pedantic
        /// </summary>
        Compute64FPedantic = 71,
        /// <summary>
        /// signed 32-bit int - default
        /// </summary>
        Compute32I = 72,
        /// <summary>
        /// signed 32-bit int - pedantic
        /// </summary>
        Compute32IPedantic = 73,
    }

    /// <summary>
    /// The cublasDataType_t type is an enumerant to specify the data precision. It is used
    /// when the data reference does not carry the type itself (e.g void *).
    /// To mimic the typedef in cublas_api.h, we redefine the enum identically to cudaDataType
    /// </summary>
    public enum DataType
    {
        ///// <summary>
        ///// the data type is 32-bit floating-point
        ///// </summary>
        //Float = 0,
        ///// <summary>
        ///// the data type is 64-bit floating-point
        ///// </summary>
        //Double = 1,
        ///// <summary>
        ///// the data type is 16-bit floating-point
        ///// </summary>
        //Half = 2,
        ///// <summary>
        ///// the data type is 8-bit signed integer
        ///// </summary>
        //Int8 = 3

        /// <summary>
        /// 16 bit real 
        /// </summary>
        CUDA_R_16F = 2,

        /// <summary>
        /// 16 bit complex
        /// </summary>
        CUDA_C_16F = 6,

        /// <summary>
        /// 32 bit real
        /// </summary>
        CUDA_R_32F = 0,

        /// <summary>
        /// 32 bit complex
        /// </summary>
        CUDA_C_32F = 4,

        /// <summary>
        /// 64 bit real
        /// </summary>
        CUDA_R_64F = 1,

        /// <summary>
        /// 64 bit complex
        /// </summary>
        CUDA_C_64F = 5,

        /// <summary>
        /// 8 bit real as a signed integer 
        /// </summary>
        CUDA_R_8I = 3,

        /// <summary>
        /// 8 bit complex as a pair of signed integers
        /// </summary>
        CUDA_C_8I = 7,

        /// <summary>
        /// 8 bit real as a signed integer 
        /// </summary>
        CUDA_R_8U = 8,

        /// <summary>
        /// 8 bit complex as a pair of signed integers
        /// </summary>
        CUDA_C_8U = 9
    }

    /// <summary>
    /// Opaque structure holding CUBLAS library context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaBlasHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// Cublas logging
    /// </summary>
    public delegate void cublasLogCallback([MarshalAs(UnmanagedType.LPStr)] string msg);
}
