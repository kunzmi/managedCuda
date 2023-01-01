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
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.CudaBlas
{
    /// <summary>
    /// C# wrapper for cublas_v2.h and cublas_api.h
    /// </summary>
    public static class CudaBlasNativeMethods
    {
        //32bit is no more supported, only 64 bit...
        internal const string CUBLAS_API_DLL_NAME = "cublas64_12";


#if (NETCOREAPP)
        internal const string CUBLAS_API_DLL_NAME_LINUX = "cublas";

        static CudaBlasNativeMethods()
        {
            NativeLibrary.SetDllImportResolver(typeof(CudaBlasNativeMethods).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            IntPtr libHandle = IntPtr.Zero;

            if (libraryName == CUBLAS_API_DLL_NAME)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    bool res = NativeLibrary.TryLoad(CUBLAS_API_DLL_NAME_LINUX, assembly, DllImportSearchPath.SafeDirectories, out libHandle);
                    if (!res)
                    {
                        Debug.WriteLine("Failed to load '" + CUBLAS_API_DLL_NAME_LINUX + "' shared library. Falling back to (Windows-) default library name '"
                            + CUBLAS_API_DLL_NAME + "'. Check LD_LIBRARY_PATH environment variable for correct paths.");
                    }
                }
            }
            //On Windows, use the default library name
            return libHandle;
        }
#endif

        #region Generic-API
        #region Basics
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCreate_v2(ref CudaBlasHandle handle);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDestroy_v2(CudaBlasHandle handle);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetVersion_v2(CudaBlasHandle handle, ref int version);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetWorkspace_v2(CudaBlasHandle handle, CUdeviceptr workspace, SizeT workspaceSizeInBytes);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetStream_v2(CudaBlasHandle handle, CUstream streamId);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetStream_v2(CudaBlasHandle handle, ref CUstream streamId);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetPointerMode_v2(CudaBlasHandle handle, ref PointerMode mode);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetPointerMode_v2(CudaBlasHandle handle, PointerMode mode);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetAtomicsMode(CudaBlasHandle handle, ref AtomicsMode mode);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetAtomicsMode(CudaBlasHandle handle, AtomicsMode mode);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetMathMode(CudaBlasHandle handle, ref Math mode);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetMathMode(CudaBlasHandle handle, Math mode);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetSmCountTarget(CudaBlasHandle handle, ref int smCountTarget);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetSmCountTarget(CudaBlasHandle handle, int smCountTarget);

        #endregion

        //New in Cuda 9.2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, [MarshalAs(UnmanagedType.LPStr)] string logFileName);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetLoggerCallback(cublasLogCallback userCallback);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetLoggerCallback(ref cublasLogCallback userCallback);
        #endregion

        #region 32Bit-API

        #region Set and Get

        /// <summary>
        /// copies n elements from a vector x in CPU memory space to a vector y 
        /// in GPU memory space. Elements in both vectors are assumed to have a 
        /// size of elemSize bytes. Storage spacing between consecutive elements
        /// is incx for the source vector x and incy for the destination vector
        /// y. In general, y points to an object, or part of an object, allocated
        /// via cublasAlloc(). Column major format for two-dimensional matrices
        /// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
        /// is equal to 1, this access a column vector while using an increment 
        /// equal to the leading dimension of the respective matrix accesses a 
        /// row vector.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetVector(int n, int elemSize, [In] IntPtr x, int incx, CUdeviceptr devicePtr, int incy);

        /// <summary>
        /// copies n elements from a vector x in GPU memory space to a vector y 
        /// in CPU memory space. Elements in both vectors are assumed to have a 
        /// size of elemSize bytes. Storage spacing between consecutive elements
        /// is incx for the source vector x and incy for the destination vector
        /// y. In general, x points to an object, or part of an object, allocated
        /// via cublasAlloc(). Column major format for two-dimensional matrices
        /// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
        /// is equal to 1, this access a column vector while using an increment 
        /// equal to the leading dimension of the respective matrix accesses a 
        /// row vector.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetVector(int n, int elemSize, [In] CUdeviceptr x, int incx, IntPtr y, int incy);

        /// <summary>
        /// copies a tile of rows x cols elements from a matrix A in CPU memory
        /// space to a matrix B in GPU memory space. Each element requires storage
        /// of elemSize bytes. Both matrices are assumed to be stored in column 
        /// major format, with the leading dimension (i.e. number of rows) of 
        /// source matrix A provided in lda, and the leading dimension of matrix B
        /// provided in ldb. In general, B points to an object, or part of an 
        /// object, that was allocated via cublasAlloc().
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetMatrix(int rows, int cols, int elemSize, [In] IntPtr A, int lda, CUdeviceptr B, int ldb);

        /// <summary>
        /// copies a tile of rows x cols elements from a matrix A in GPU memory
        /// space to a matrix B in CPU memory space. Each element requires storage
        /// of elemSize bytes. Both matrices are assumed to be stored in column 
        /// major format, with the leading dimension (i.e. number of rows) of 
        /// source matrix A provided in lda, and the leading dimension of matrix B
        /// provided in ldb. In general, A points to an object, or part of an 
        /// object, that was allocated via cublasAlloc().
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetMatrix(int rows, int cols, int elemSize, [In] CUdeviceptr A, int lda, IntPtr B, int ldb);

        /// <summary>
        /// cublasSetVectorAsync has the same functionnality as cublasSetVector
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetVectorAsync(int n, int elemSize, [In] IntPtr hostPtr, int incx, CUdeviceptr devicePtr, int incy, CUstream stream);
        /// <summary>
        /// cublasGetVectorAsync has the same functionnality as cublasGetVector
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetVectorAsync(int n, int elemSize, [In] CUdeviceptr devicePtr, int incx, IntPtr hostPtr, int incy, CUstream stream);

        /// <summary>
        /// cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetMatrixAsync(int rows, int cols, int elemSize, [In] IntPtr A, int lda, CUdeviceptr B, int ldb, CUstream stream);

        /// <summary>
        /// cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetMatrixAsync(int rows, int cols, int elemSize, [In] CUdeviceptr A, int lda, IntPtr B, int ldb, CUstream stream);

        #endregion

        #region BLAS1
        #region host/device independent
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCopyEx(CudaBlasHandle handle,
                                                      int n,
                                                      [In] CUdeviceptr x,
                                                      cudaDataType xType,
                                                      int incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      int incy);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScopy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDcopy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCcopy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZcopy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSswap_v2(CudaBlasHandle handle,
                                         int n,
                                         CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDswap_v2(CudaBlasHandle handle,
                                         int n,
                                         CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCswap_v2(CudaBlasHandle handle,
                                         int n,
                                         CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZswap_v2(CudaBlasHandle handle,
                                         int n,
                                         CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSwapEx(CudaBlasHandle handle,
                                                      int n,
                                                      CUdeviceptr x,
                                                      cudaDataType xType,
                                                      int incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      int incy);

        #endregion

        #region Host pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasNrm2Ex(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     IntPtr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType); /* host or device pointer */

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     IntPtr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotcEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     IntPtr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSnrm2_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        ref float result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDnrm2_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        ref double result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScnrm2_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref float result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDznrm2_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref double result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSdot_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        ref float result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDdot_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        ref double result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotu_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         ref cuFloatComplex result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotc_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         ref cuFloatComplex result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotu_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         ref cuDoubleComplex result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotc_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         ref cuDoubleComplex result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScalEx(CudaBlasHandle handle,
                                                     int n,
                                                     IntPtr alpha,  /* host or device pointer */
                                                     cudaDataType alphaType,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     cudaDataType executionType);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] ref float alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] ref double alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] ref cuFloatComplex alpha, // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsscal_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] ref float alpha, // host or device pointer
                                         CUdeviceptr x,
                                         int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] ref cuDoubleComplex alpha, // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdscal_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] ref double alpha, // host or device pointer
                                         CUdeviceptr x,
                                         int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAxpyEx(CudaBlasHandle handle,
                                                      int n,
                                                      IntPtr alpha, /* host or device pointer */
                                                      cudaDataType alphaType,
                                                      CUdeviceptr x,
                                                      cudaDataType xType,
                                                      int incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      int incy,
                                                      cudaDataType executiontype);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIamaxEx(CudaBlasHandle handle,
                                                      int n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      int incx,
                                                      ref int result  /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref int result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIaminEx(CudaBlasHandle handle,
                                                      int n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      int incx,
                                                      ref int result /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAsumEx(CudaBlasHandle handle,
                                                     int n,

                                                     [In] CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     IntPtr result,
                                                     cudaDataType resultType, /* host or device pointer */
                                                     cudaDataType executiontype
                                                  );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSasum_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        ref float result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDasum_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        ref double result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScasum_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref float result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDzasum_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         ref double result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] ref float c,  // host or device pointer
                                        [In] ref float s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] ref double c,  // host or device pointer
                                        [In] ref double s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] ref float c,      // host or device pointer
                                        [In] ref cuFloatComplex s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] ref float c,  // host or device pointer
                                        [In] ref float s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] ref double c,            // host or device pointer
                                        [In] ref cuDoubleComplex s);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] ref double c,  // host or device pointer
                                        [In] ref double s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     IntPtr c,  /* host or device pointer */
                                                     IntPtr s,
                                                     cudaDataType csType,
                                                     cudaDataType executiontype);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotg_v2(CudaBlasHandle handle,
                                        ref float a,   // host or device pointer
                                        ref float b,   // host or device pointer
                                        ref float c,   // host or device pointer
                                        ref float s);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotg_v2(CudaBlasHandle handle,
                                        ref double a,  // host or device pointer
                                        ref double b,  // host or device pointer
                                        ref double c,  // host or device pointer
                                        ref double s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCrotg_v2(CudaBlasHandle handle,
                                        ref cuFloatComplex a,  // host or device pointer
                                        ref cuFloatComplex b,  // host or device pointer
                                        ref float c,      // host or device pointer
                                        ref cuFloatComplex s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZrotg_v2(CudaBlasHandle handle,
                                        ref cuDoubleComplex a,  // host or device pointer
                                        ref cuDoubleComplex b,  // host or device pointer
                                        ref double c,           // host or device pointer
                                        ref cuDoubleComplex s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotgEx(CudaBlasHandle handle,
                                                     IntPtr a,   /* host or device pointer */
                                                     IntPtr b,   /* host or device pointer */
                                                     cudaDataType abType,
                                                     IntPtr c,   /* host or device pointer */
                                                     IntPtr s,   /* host or device pointer */
                                                     cudaDataType csType,
                                                     cudaDataType executiontype);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotm_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In, MarshalAs(UnmanagedType.LPArray, SizeConst = 5)] float[] param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotm_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In, MarshalAs(UnmanagedType.LPArray, SizeConst = 5)] double[] param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotmEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     IntPtr param, /* host or device pointer */
                                                     cudaDataType paramType,
                                                     cudaDataType executiontype);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotmg_v2(CudaBlasHandle handle,
                                         ref float d1,        // host or device pointer
                                         ref float d2,        // host or device pointer
                                         ref float x1,        // host or device pointer
                                         [In] ref float y1,  // host or device pointer
                                         [MarshalAs(UnmanagedType.LPArray, SizeConst = 5)] float[] param);    // host or device pointer

        /// <summary>
        /// </summary> 
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotmg_v2(CudaBlasHandle handle,
                                         ref double d1,        // host or device pointer  
                                         ref double d2,        // host or device pointer  
                                         ref double x1,        // host or device pointer  
                                         [In] ref double y1,  // host or device pointer  
                                         [MarshalAs(UnmanagedType.LPArray, SizeConst = 5)] double[] param);    // host or device pointer  

        /// <summary>
        /// </summary> 
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotmgEx(CudaBlasHandle handle,
                                                      IntPtr d1,        /* host or device pointer */
                                                      cudaDataType d1Type,
                                                      IntPtr d2,        /* host or device pointer */
                                                      cudaDataType d2Type,
                                                      IntPtr x1,        /* host or device pointer */
                                                      cudaDataType x1Type,
                                                      IntPtr y1,  /* host or device pointer */
                                                      cudaDataType y1Type,
                                                      IntPtr param,     /* host or device pointer */
                                                      cudaDataType paramType,
                                                      cudaDataType executiontype
                                                      );
        #endregion

        #region Device pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasNrm2Ex(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType); /* host or device pointer */

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotcEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSnrm2_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDnrm2_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScnrm2_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDznrm2_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSdot_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDdot_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotu_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotc_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotu_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotc_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScalEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr alpha,  /* host or device pointer */
                                                     cudaDataType alphaType,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     cudaDataType executionType);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsscal_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         CUdeviceptr x,
                                         int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZscal_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        CUdeviceptr x,
                                        int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdscal_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         CUdeviceptr x,
                                         int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAxpyEx(CudaBlasHandle handle,
                                                      int n,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      cudaDataType alphaType,
                                                      CUdeviceptr x,
                                                      cudaDataType xType,
                                                      int incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      int incy,
                                                      cudaDataType executiontype);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZaxpy_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamax_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIamaxEx(CudaBlasHandle handle,
                                                      int n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      int incx,
                                                      CUdeviceptr result  /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamin_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIaminEx(CudaBlasHandle handle,
                                                      int n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      int incx,
                                                      CUdeviceptr result /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAsumEx(CudaBlasHandle handle,
                                                     int n,

                                                     [In] CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType, /* host or device pointer */
                                                     cudaDataType executiontype
                                                  );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSasum_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDasum_v2(CudaBlasHandle handle,
                                        int n,
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScasum_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDzasum_v2(CudaBlasHandle handle,
                                         int n,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr c,      // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr c,            // host or device pointer
                                        [In] CUdeviceptr s);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdrot_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     CUdeviceptr c,  /* host or device pointer */
                                                     CUdeviceptr s,
                                                     cudaDataType csType,
                                                     cudaDataType executiontype);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotg_v2(CudaBlasHandle handle,
                                        CUdeviceptr a,   // host or device pointer
                                        CUdeviceptr b,   // host or device pointer
                                        CUdeviceptr c,   // host or device pointer
                                        CUdeviceptr s);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotg_v2(CudaBlasHandle handle,
                                        CUdeviceptr a,  // host or device pointer
                                        CUdeviceptr b,  // host or device pointer
                                        CUdeviceptr c,  // host or device pointer
                                        CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCrotg_v2(CudaBlasHandle handle,
                                        CUdeviceptr a,  // host or device pointer
                                        CUdeviceptr b,  // host or device pointer
                                        CUdeviceptr c,      // host or device pointer
                                        CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZrotg_v2(CudaBlasHandle handle,
                                        CUdeviceptr a,  // host or device pointer
                                        CUdeviceptr b,  // host or device pointer
                                        CUdeviceptr c,           // host or device pointer
                                        CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotgEx(CudaBlasHandle handle,
                                                     CUdeviceptr a,   /* host or device pointer */
                                                     CUdeviceptr b,   /* host or device pointer */
                                                     cudaDataType abType,
                                                     CUdeviceptr c,   /* host or device pointer */
                                                     CUdeviceptr s,   /* host or device pointer */
                                                     cudaDataType csType,
                                                     cudaDataType executiontype);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotm_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotm_v2(CudaBlasHandle handle,
                                        int n,
                                        CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr y,
                                        int incy,
                                        [In] CUdeviceptr param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotmEx(CudaBlasHandle handle,
                                                     int n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     int incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     int incy,
                                                     CUdeviceptr param, /* host or device pointer */
                                                     cudaDataType paramType,
                                                     cudaDataType executiontype);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotmg_v2(CudaBlasHandle handle,
                                         CUdeviceptr d1,        // host or device pointer
                                         CUdeviceptr d2,        // host or device pointer
                                         CUdeviceptr x1,        // host or device pointer
                                         [In] CUdeviceptr y1,  // host or device pointer
                                         CUdeviceptr param);    // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotmg_v2(CudaBlasHandle handle,
                                         CUdeviceptr d1,        // host or device pointer
                                         CUdeviceptr d2,        // host or device pointer
                                         CUdeviceptr x1,        // host or device pointer
                                         [In] CUdeviceptr y1,  // host or device pointer
                                         CUdeviceptr param);    // host or device pointer  

        /// <summary>
        /// </summary> 
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotmgEx(CudaBlasHandle handle,
                                                      CUdeviceptr d1,        /* host or device pointer */
                                                      cudaDataType d1Type,
                                                      CUdeviceptr d2,        /* host or device pointer */
                                                      cudaDataType d2Type,
                                                      CUdeviceptr x1,        /* host or device pointer */
                                                      cudaDataType x1Type,
                                                      [In] CUdeviceptr y1,  /* host or device pointer */
                                                      cudaDataType y1Type,
                                                      CUdeviceptr param,     /* host or device pointer */
                                                      cudaDataType paramType,
                                                      cudaDataType executiontype
                                                      );
        #endregion
        #endregion

        #region BLAS2
        #region host/device independent
        #region TRMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);
        #endregion
        #region TBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStbmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtbmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtbmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtbmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A, int lda,
                                         CUdeviceptr x, int incx);
        #endregion
        #region TPMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStpmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtpmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtpmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtpmv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP,
                                         CUdeviceptr x, int incx);
        #endregion
        #region TRSV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr A, int lda,
                                         CUdeviceptr x, int incx);
        #endregion
        #region TPSV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStpsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP,
                                         CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtpsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtpsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtpsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, [In] CUdeviceptr AP,
                                         CUdeviceptr x, int incx);
        #endregion
        #region TBSV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStbsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A,
                                         int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtbsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A,
                                         int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtbsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A,
                                         int lda, CUdeviceptr x, int incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtbsv_v2(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, int n, int k, [In] CUdeviceptr A,
                                         int lda, CUdeviceptr x, int incx);
        #endregion
        #endregion

        #region host pointer
        #region GEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref float beta,  // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref double beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region GBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref float beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref double beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region SYMV/HEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref float beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref double beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuDoubleComplex alpha,  // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuDoubleComplex beta,   // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region SBMV/HBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] ref float alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref float beta,  // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] ref double alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref double beta,   // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region SPMV/HPMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref float alpha,  // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref float beta,   // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref double beta,  // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region GER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSger_v2(CudaBlasHandle handle,
                                        int m,
                                        int n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDger_v2(CudaBlasHandle handle,
                                        int m,
                                        int n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeru_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgerc_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeru_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgerc_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);
        #endregion
        #region SYR/HER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref cuFloatComplex alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref cuDoubleComplex alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);
        #endregion
        #region SPR/HPR
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);
        #endregion
        #region SYR2/HER2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] ref cuFloatComplex alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2_v2(CudaBlasHandle handle,
                                         FillMode uplo, int n,
                                         [In] ref cuFloatComplex alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuDoubleComplex alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        #endregion
        #region SPR2/HPR2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref float alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref double alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);
        #endregion
        #endregion

        #region device pointer
        #region GEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta,  // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region GBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgbmv_v2(CudaBlasHandle handle,
                                         Operation trans,
                                         int m,
                                         int n,
                                         int kl,
                                         int ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region SYMV/HEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta,   // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region SBMV/HBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta,  // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta,   // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhbmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region SPMV/HPMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta,   // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta,  // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpmv_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         int incy);
        #endregion
        #region GER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSger_v2(CudaBlasHandle handle,
                                        int m,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDger_v2(CudaBlasHandle handle,
                                        int m,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeru_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgerc_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeru_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgerc_v2(CudaBlasHandle handle,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);
        #endregion
        #region SYR/HER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr A,
                                        int lda);
        #endregion
        #region SPR/HPR
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        CUdeviceptr AP);
        #endregion
        #region SYR2/HER2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        int n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        int incx,
                                        [In] CUdeviceptr y,
                                        int incy,
                                        CUdeviceptr A,
                                        int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2_v2(CudaBlasHandle handle,
                                         FillMode uplo, int n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr A,
                                         int lda);

        #endregion
        #region SPR2/HPR2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr2_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         int n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         int incx,
                                         [In] CUdeviceptr y,
                                         int incy,
                                         CUdeviceptr AP);
        #endregion
        #endregion
        #endregion

        #region BLAS3
        #region host pointer
        #region GEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] ref float alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] ref float beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] ref double alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] ref double beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] ref cuFloatComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] ref cuFloatComplex beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3m(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      ref cuFloatComplex alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      ref cuFloatComplex beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3m(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm3m(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      ref cuDoubleComplex alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      ref cuDoubleComplex beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm3m(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mEx(CudaBlasHandle handle,
                                                     Operation transa, Operation transb,
                                                     int m, int n, int k,
                                                     ref cuFloatComplex alpha,
                                                     CUdeviceptr A,
                                                     cudaDataType Atype,
                                                     int lda,
                                                     CUdeviceptr B,
                                                     cudaDataType Btype,
                                                     int ldb,
                                                     ref cuFloatComplex beta,
                                                     CUdeviceptr C,
                                                     cudaDataType Ctype,
                                                     int ldc);




        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] ref cuDoubleComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] ref cuDoubleComplex beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemm(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      ref half alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      ref half beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemm(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmBatched(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      ref half alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      ref half beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc,
                                                      int batchCount);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmBatched(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      int ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      int ldc,
                                                      int batchCount);

        /* IO in FP16/FP32, computation in float */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmEx(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      ref float alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      DataType Atype,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      DataType Btype,
                                                      int ldb,
                                                      ref float beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      DataType Ctype,
                                                      int ldc);



        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmEx(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      IntPtr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      IntPtr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      ComputeType computeType,
                                                      GemmAlgo algo);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmEx(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      ComputeType computeType,
                                                      GemmAlgo algo);

        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmEx(CudaBlasHandle handle,
                                                     Operation transa, Operation transb,
                                                     int m, int n, int k,
                                                     ref cuFloatComplex alpha,
                                                     CUdeviceptr A,
                                                     cudaDataType Atype,
                                                     int lda,
                                                     CUdeviceptr B,
                                                     cudaDataType Btype,
                                                     int ldb,
                                                     ref cuFloatComplex beta,
                                                     CUdeviceptr C,
                                                     cudaDataType Ctype,
                                                     int ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmEx(CudaBlasHandle handle,
                                                     Operation transa, Operation transb,
                                                     int m, int n, int k,
                                                     CUdeviceptr alpha,
                                                     CUdeviceptr A,
                                                     cudaDataType Atype,
                                                     int lda,
                                                     CUdeviceptr B,
                                                     cudaDataType Btype,
                                                     int ldb,
                                                     CUdeviceptr beta,
                                                     CUdeviceptr C,
                                                     cudaDataType Ctype,
                                                     int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        [Obsolete]
        public static extern CublasStatus cublasUint8gemmBias(CudaBlasHandle handle,
                                                           Operation transa, Operation transb, Operation transc,
                                                           int m, int n, int k,
                                                           CUdeviceptr A, int A_bias, int lda,
                                                           CUdeviceptr B, int B_bias, int ldb,
                                                           CUdeviceptr C, int C_bias, int ldc,
                                                           int C_mult, int C_shift);

        /* SYRK */


        /* IO in FP16/FP32, computation in float */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmEx(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      DataType Atype,
                                                      int lda,
                                                      CUdeviceptr B,
                                                      DataType Btype,
                                                      int ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      DataType Ctype,
                                                      int ldc);



        #endregion
        #region SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrk_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        int n,
                                        int k,
                                        [In] ref float alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] ref float beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] ref double alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] ref double beta,  //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] ref cuFloatComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] ref cuFloatComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] ref cuDoubleComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] ref cuDoubleComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkEx(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              int n,
                                                              int k,
                                                              ref cuFloatComplex alpha, /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              int lda,
                                                              ref cuFloatComplex beta, /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              int ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkEx(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              int n,
                                                              int k,
                                                              CUdeviceptr alpha, /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              int lda,
                                                              CUdeviceptr beta, /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              int ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk3mEx(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              int n,
                                                              int k,
                                                              ref cuFloatComplex alpha,
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              int lda,
                                                              ref cuFloatComplex beta,
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              int ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk3mEx(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              int n,
                                                              int k,
                                                              CUdeviceptr lpha,
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              int lda,
                                                              CUdeviceptr beta,
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              int ldc);

        #endregion
        #region HERK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] ref float alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] ref float beta,   //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherk_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        int n,
                                        int k,
                                        [In] ref double alpha,  //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] ref double beta,  //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        #endregion
        #region SYR2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] ref float alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] ref float beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] ref double alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] ref double beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] ref cuFloatComplex alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] ref cuFloatComplex beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] ref cuDoubleComplex alpha,  //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] ref cuDoubleComplex beta,  //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);
        #endregion
        #region HER2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] ref cuFloatComplex alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] ref float beta,   //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] ref cuDoubleComplex alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] ref double beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkEx(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              int n,
                                                              int k,
                                                              ref float alpha,  /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              int lda,
                                                              ref float beta,   /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              int ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk3mEx(CudaBlasHandle handle,
                                                               FillMode uplo,
                                                               Operation trans,
                                                               int n,
                                                               int k,
                                                               ref float alpha,
                                                               CUdeviceptr A, cudaDataType Atype,
                                                               int lda,
                                                               ref float beta,
                                                               CUdeviceptr C,
                                                               cudaDataType Ctype,
                                                               int ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkEx(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              int n,
                                                              int k,
                                                              CUdeviceptr alpha,  /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              int lda,
                                                              CUdeviceptr beta,   /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              int ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk3mEx(CudaBlasHandle handle,
                                                               FillMode uplo,
                                                               Operation trans,
                                                               int n,
                                                               int k,
                                                               CUdeviceptr alpha,
                                                               CUdeviceptr A, cudaDataType Atype,
                                                               int lda,
                                                               CUdeviceptr beta,
                                                               CUdeviceptr C,
                                                               cudaDataType Ctype,
                                                               int ldc);

        /* SYR2K */

        #endregion
        #region SYRKX : eXtended SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] ref float alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] ref float beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] ref double alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] ref double beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] ref cuFloatComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] ref cuFloatComplex beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] ref cuDoubleComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] ref cuDoubleComplex beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);
        #endregion

        #region HERKX : eXtended HERK         
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] ref cuFloatComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] ref float beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] ref cuDoubleComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] ref double beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);
        #endregion

        #region SYMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] ref float alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] ref float beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] ref double alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] ref double beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] ref cuFloatComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] ref cuFloatComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] ref cuDoubleComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        #endregion
        #region HEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] ref cuFloatComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] ref cuFloatComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] ref cuDoubleComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] ref cuDoubleComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        #endregion
        #region TRSM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] ref float alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         CUdeviceptr B,
                                         int ldb);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] ref double alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         CUdeviceptr B,
                                         int ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsm_v2(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] ref cuFloatComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        CUdeviceptr B,
                                        int ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsm_v2(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] ref cuDoubleComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        CUdeviceptr B,
                                        int ldb);

        #endregion
        #region TRMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrmm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] ref float alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrmm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] ref double alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrmm_v2(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] ref cuFloatComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrmm_v2(CudaBlasHandle handle, SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] ref cuDoubleComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        CUdeviceptr C,
                                        int ldc);
        #endregion
        #endregion

        #region device pointer
        #region GEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm_v2(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        int m,
                                        int n,
                                        int k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);
        #endregion
        #region SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrk_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        int n,
                                        int k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr beta,  //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);
        #endregion
        #region HERK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk_v2(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         int n,
                                         int k,
                                         [In] CUdeviceptr alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr beta,   //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherk_v2(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        int n,
                                        int k,
                                        [In] CUdeviceptr alpha,  //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr beta,  //host or device pointer  
                                        CUdeviceptr C,
                                        int ldc);

        #endregion
        #region SYR2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] CUdeviceptr alpha,  //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] CUdeviceptr beta,  //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);
        #endregion
        #region HER2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] CUdeviceptr beta,   //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2k_v2(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          int n,
                                          int k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          int lda,
                                          [In] CUdeviceptr B,
                                          int ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          int ldc);

        #endregion
        #region SYRKX : eXtended SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);
        #endregion

        #region HERKX : eXtended HERK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherkx(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    int n,
                                                    int k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    int lda,
                                                    [In] CUdeviceptr B,
                                                    int ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    int ldc);
        #endregion
        #region SYMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        #endregion
        #region HEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         int ldc);

        #endregion
        #region TRSM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         CUdeviceptr B,
                                         int ldb);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         CUdeviceptr B,
                                         int ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsm_v2(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        CUdeviceptr B,
                                        int ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsm_v2(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        CUdeviceptr B,
                                        int ldb);

        #endregion
        #region TRMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrmm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrmm_v2(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         int m,
                                         int n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         int lda,
                                         [In] CUdeviceptr B,
                                         int ldb,
                                         CUdeviceptr C,
                                         int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrmm_v2(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        CUdeviceptr C,
                                        int ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrmm_v2(CudaBlasHandle handle, SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        int m,
                                        int n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        int lda,
                                        [In] CUdeviceptr B,
                                        int ldb,
                                        CUdeviceptr C,
                                        int ldc);
        #endregion
        #endregion
        #endregion

        #region CUBLAS BLAS-like extension
        #region GEAM
        #region device ptr
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);
        #endregion
        #region host ptr
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  ref float alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  ref float beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  ref double alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  ref double beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  ref cuFloatComplex alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  ref cuFloatComplex beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeam(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  int m,
                                                  int n,
                                                  ref cuDoubleComplex alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  int lda,
                                                  ref cuDoubleComplex beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  int ldb,
                                                  CUdeviceptr C,
                                                  int ldc);
        #endregion
        #endregion

        #region Batched - MATINV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSmatinvBatched(CudaBlasHandle handle,
                                                          int n,
                                                          CUdeviceptr A,                  /*Device pointer*/
                                                          int lda,
                                                          CUdeviceptr Ainv,               /*Device pointer*/
                                                          int lda_inv,
                                                          CUdeviceptr INFO,                   /*Device Pointer*/
                                                          int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDmatinvBatched(CudaBlasHandle handle,
                                                          int n,
                                                          CUdeviceptr A,                 /*Device pointer*/
                                                          int lda,
                                                          CUdeviceptr Ainv,              /*Device pointer*/
                                                          int lda_inv,
                                                          CUdeviceptr INFO,                   /*Device Pointer*/
                                                          int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCmatinvBatched(CudaBlasHandle handle,
                                                          int n,
                                                          CUdeviceptr A,              /*Device pointer*/
                                                          int lda,
                                                          CUdeviceptr Ainv,           /*Device pointer*/
                                                          int lda_inv,
                                                          CUdeviceptr INFO,                   /*Device Pointer*/
                                                          int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZmatinvBatched(CudaBlasHandle handle,
                                                          int n,
                                                          CUdeviceptr A,        /*Device pointer*/
                                                          int lda,
                                                          CUdeviceptr Ainv,     /*Device pointer*/
                                                          int lda_inv,
                                                          CUdeviceptr INFO,                   /*Device Pointer*/
                                                          int batchSize);

        #endregion

        #region DGMM

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSdgmm(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr x,
                                                  int incx,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDdgmm(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr x,
                                                  int incx,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdgmm(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr x,
                                                  int incx,
                                                  CUdeviceptr C,
                                                  int ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdgmm(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  int m,
                                                  int n,
                                                  CUdeviceptr A,
                                                  int lda,
                                                  CUdeviceptr x,
                                                  int incx,
                                                  CUdeviceptr C,
                                                  int ldc);
        #endregion
        #endregion
        //Ab hier NEU

        #region BATCH GEMM
        #region device pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   CUdeviceptr alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   CUdeviceptr beta,   /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   CUdeviceptr alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   CUdeviceptr beta,  /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   CUdeviceptr alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   CUdeviceptr beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   CUdeviceptr alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   CUdeviceptr beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);


        //Missing before:
        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mBatched(CudaBlasHandle handle,
                                                          Operation transa,
                                                          Operation transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          ref cuFloatComplex alpha, /* host or device pointer */
                                                          CUdeviceptr Aarray,
                                                          int lda,
                                                          CUdeviceptr Barray,
                                                          int ldb,
                                                          ref cuFloatComplex beta, /* host or device pointer */
                                                          CUdeviceptr Carray,
                                                          int ldc,
                                                          int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mBatched(CudaBlasHandle handle,
                                                          Operation transa,
                                                          Operation transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          CUdeviceptr alpha, /* host or device pointer */
                                                          CUdeviceptr Aarray,
                                                          int lda,
                                                          CUdeviceptr Barray,
                                                          int ldb,
                                                          CUdeviceptr beta, /* host or device pointer */
                                                          CUdeviceptr Carray,
                                                          int ldc,
                                                          int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mStridedBatched(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         int m,
                                                                         int n,
                                                                         int k,
                                                                 ref cuFloatComplex alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 ref cuFloatComplex beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mStridedBatched(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         int m,
                                                                         int n,
                                                                         int k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);




        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmBatchedEx(CudaBlasHandle handle,
                                                              Operation transa,
                                                              Operation transb,
                                                              int m,
                                                              int n,
                                                              int k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr Aarray,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      CUdeviceptr Barray,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr Carray,
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      int batchCount,
                                                      ComputeType computeType,
                                                      GemmAlgo algo);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmStridedBatchedEx(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         int m,
                                                                         int n,
                                                                         int k,
                                                                 CUdeviceptr alpha,  /* host or device pointer */
                                                                 CUdeviceptr A,
                                                                 cudaDataType Atype,
                                                                 int lda,
                                                                 long strideA,   /* purposely signed */
                                                                 CUdeviceptr B,
                                                                 cudaDataType Btype,
                                                                 int ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   /* host or device pointer */
                                                                 CUdeviceptr C,
                                                                 cudaDataType Ctype,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount,
                                                                 ComputeType computeType,
                                                                 GemmAlgo algo);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer   
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 CUdeviceptr alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 CUdeviceptr alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);


        #endregion
        #region host pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   ref float alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   ref float beta,   /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   ref double alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   ref double beta,  /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   ref cuFloatComplex alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   ref cuFloatComplex beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmBatched(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   int m,
                                   int n,
                                   int k,
                                   ref cuDoubleComplex alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   int lda,
                                   CUdeviceptr Barray,
                                   int ldb,
                                   ref cuDoubleComplex beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   int ldc,
                                   int batchCount);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmBatchedEx(CudaBlasHandle handle,
                                                              Operation transa,
                                                              Operation transb,
                                                              int m,
                                                              int n,
                                                              int k,
                                                      IntPtr alpha, /* host or device pointer */
                                                      CUdeviceptr Aarray,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      CUdeviceptr Barray,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      IntPtr beta, /* host or device pointer */
                                                      CUdeviceptr Carray,
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      int batchCount,
                                                      cudaDataType computeType,
                                                      GemmAlgo algo);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmStridedBatchedEx(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         int m,
                                                                         int n,
                                                                         int k,
                                                                 IntPtr alpha,  /* host or device pointer */
                                                                 CUdeviceptr A,
                                                                 cudaDataType Atype,
                                                                 int lda,
                                                                 long strideA,   /* purposely signed */
                                                                 CUdeviceptr B,
                                                                 cudaDataType Btype,
                                                                 int ldb,
                                                                 long strideB,
                                                                 IntPtr beta,   /* host or device pointer */
                                                                 CUdeviceptr C,
                                                                 cudaDataType Ctype,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount,
                                                                 cudaDataType computeType,
                                                                 GemmAlgo algo);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 ref float alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 ref float beta,   // host or device pointer   
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 ref double alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 ref double beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 ref cuFloatComplex alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 ref cuFloatComplex beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 ref cuDoubleComplex alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 ref cuDoubleComplex beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmStridedBatched(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 ref half alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 int lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 int ldb,
                                                                 long strideB,
                                                                 ref half beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 int ldc,
                                                                 long strideC,
                                                                 int batchCount);

        #endregion
        #endregion

        #region Batched LU - GETRF
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgetrfBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,                      /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                          /*Device Pointer*/
                                                  CUdeviceptr INFO,                       /*Device Pointer*/
                                                  int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgetrfBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,                     /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                          /*Device Pointer*/
                                                  CUdeviceptr INFO,                       /*Device Pointer*/
                                                  int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgetrfBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,                 /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                         /*Device Pointer*/
                                                  CUdeviceptr INFO,                      /*Device Pointer*/
                                                  int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgetrfBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,           /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                         /*Device Pointer*/
                                                  CUdeviceptr INFO,                      /*Device Pointer*/
                                                  int batchSize);

        #endregion

        #region Batched inversion based on LU factorization from getrf
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgetriBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,                     /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                         /*Device pointer*/
                                                  CUdeviceptr C,                     /*Device pointer*/
                                                  int ldc,
                                                  CUdeviceptr INFO,
                                                  int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgetriBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,                    /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                         /*Device pointer*/
                                                  CUdeviceptr C,                    /*Device pointer*/
                                                  int ldc,
                                                  CUdeviceptr INFO,
                                                  int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgetriBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,                 /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                         /*Device pointer*/
                                                  CUdeviceptr C,                 /*Device pointer*/
                                                  int ldc,
                                                  CUdeviceptr INFO,
                                                  int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgetriBatched(CudaBlasHandle handle,
                                                  int n,
                                                  CUdeviceptr A,           /*Device pointer*/
                                                  int lda,
                                                  CUdeviceptr P,                         /*Device pointer*/
                                                  CUdeviceptr C,           /*Device pointer*/
                                                  int ldc,
                                                  CUdeviceptr INFO,
                                                  int batchSize);

        #endregion

        #region TRSM - Batched Triangular Solver
        #region device pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          CUdeviceptr alpha,           /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          CUdeviceptr alpha,          /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          CUdeviceptr alpha,       /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          CUdeviceptr salpha, /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);


        #endregion
        #region host pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          ref float alpha,           /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          ref double alpha,          /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          ref cuFloatComplex alpha,       /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsmBatched(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          int m,
                                                          int n,
                                                          ref cuDoubleComplex alpha, /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          int lda,
                                                          CUdeviceptr B,
                                                          int ldb,
                                                          int batchCount);


        #endregion
        #endregion


        #region TPTTR : Triangular Pack format to Triangular format
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStpttr(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr AP,
                                                     CUdeviceptr A,
                                                     int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtpttr(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr AP,
                                                     CUdeviceptr A,
                                                     int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtpttr(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr AP,
                                                     CUdeviceptr A,
                                                     int lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtpttr(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr AP,
                                                     CUdeviceptr A,
                                                     int lda);
        #endregion

        #region TRTTP : Triangular format to Triangular Pack format 
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrttp(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr A,
                                                     int lda,
                                                     CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrttp(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr A,
                                                     int lda,
                                                     CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrttp(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr A,
                                                     int lda,
                                                     CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrttp(CudaBlasHandle handle,
                                                     FillMode uplo,
                                                     int n,
                                                     CUdeviceptr A,
                                                     int lda,
                                                     CUdeviceptr AP);
        #endregion




        #region Batch QR Factorization

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgeqrfBatched(CudaBlasHandle handle,
                                                           int m,
                                                           int n,
                                                           CUdeviceptr Aarray,           /*Device pointer*/
                                                           int lda,
                                                           CUdeviceptr TauArray,        /* Device pointer*/
                                                           ref int info,
                                                           int batchSize);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgeqrfBatched(CudaBlasHandle handle,
                                                            int m,
                                                            int n,
                                                            CUdeviceptr Aarray,           /*Device pointer*/
                                                            int lda,
                                                            CUdeviceptr TauArray,        /* Device pointer*/
                                                            ref int info,
                                                            int batchSize);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeqrfBatched(CudaBlasHandle handle,
                                                            int m,
                                                            int n,
                                                            CUdeviceptr Aarray,           /*Device pointer*/
                                                            int lda,
                                                            CUdeviceptr TauArray,        /* Device pointer*/
                                                            ref int info,
                                                            int batchSize);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeqrfBatched(CudaBlasHandle handle,
                                                            int m,
                                                            int n,
                                                            CUdeviceptr Aarray,           /*Device pointer*/
                                                            int lda,
                                                            CUdeviceptr TauArray,        /* Device pointer*/
                                                            ref int info,
                                                            int batchSize);
        #endregion

        #region Least Square Min only m >= n and Non-transpose supported
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgelsBatched(CudaBlasHandle handle,
                                                           Operation trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           CUdeviceptr Aarray, /*Device pointer*/
                                                           int lda,
                                                           CUdeviceptr Carray, /* Device pointer*/
                                                           int ldc,
                                                           ref int info,
                                                           CUdeviceptr devInfoArray, /* Device pointer*/
                                                           int batchSize);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgelsBatched(CudaBlasHandle handle,
                                                           Operation trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           CUdeviceptr Aarray, /*Device pointer*/
                                                           int lda,
                                                           CUdeviceptr Carray, /* Device pointer*/
                                                           int ldc,
                                                           ref int info,
                                                           CUdeviceptr devInfoArray, /* Device pointer*/
                                                           int batchSize);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgelsBatched(CudaBlasHandle handle,
                                                           Operation trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           CUdeviceptr Aarray, /*Device pointer*/
                                                           int lda,
                                                           CUdeviceptr Carray, /* Device pointer*/
                                                           int ldc,
                                                           ref int info,
                                                           CUdeviceptr devInfoArray,
                                                           int batchSize);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgelsBatched(CudaBlasHandle handle,
                                                           Operation trans,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           CUdeviceptr Aarray, /*Device pointer*/
                                                           int lda,
                                                           CUdeviceptr Carray, /* Device pointer*/
                                                           int ldc,
                                                           ref int info,
                                                           CUdeviceptr devInfoArray,
                                                           int batchSize);
        #endregion

        //New in Cuda 7.0

        #region Batched solver based on LU factorization from getrf

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgetrsBatched(CudaBlasHandle handle,
                                                            Operation trans,
                                                            int n,
                                                            int nrhs,
                                                            CUdeviceptr Aarray,
                                                            int lda,
                                                            CUdeviceptr devIpiv,
                                                            CUdeviceptr Barray,
                                                            int ldb,
                                                            ref int info,
                                                            int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgetrsBatched(CudaBlasHandle handle,
                                                           Operation trans,
                                                           int n,
                                                           int nrhs,
                                                           CUdeviceptr Aarray,
                                                           int lda,
                                                           CUdeviceptr devIpiv,
                                                           CUdeviceptr Barray,
                                                           int ldb,
                                                           ref int info,
                                                           int batchSize);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgetrsBatched(CudaBlasHandle handle,
                                                            Operation trans,
                                                            int n,
                                                            int nrhs,
                                                            CUdeviceptr Aarray,
                                                            int lda,
                                                            CUdeviceptr devIpiv,
                                                            CUdeviceptr Barray,
                                                            int ldb,
                                                            ref int info,
                                                            int batchSize);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgetrsBatched(CudaBlasHandle handle,
                                                            Operation trans,
                                                            int n,
                                                            int nrhs,
                                                            CUdeviceptr Aarray,
                                                            int lda,
                                                            CUdeviceptr devIpiv,
                                                            CUdeviceptr Barray,
                                                            int ldb,
                                                            ref int info,
                                                            int batchSize);
        #endregion



        #endregion

        #region 64Bit-API


        #region Set and Get

        /// <summary>
        /// copies n elements from a vector x in CPU memory space to a vector y 
        /// in GPU memory space. Elements in both vectors are assumed to have a 
        /// size of elemSize bytes. Storage spacing between consecutive elements
        /// is incx for the source vector x and incy for the destination vector
        /// y. In general, y points to an object, or part of an object, allocated
        /// via cublasAlloc(). Column major format for two-dimensional matrices
        /// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
        /// is equal to 1, this access a column vector while using an increment 
        /// equal to the leading dimension of the respective matrix accesses a 
        /// row vector.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetVector_64(long n, long elemSize, [In] IntPtr x, long incx, CUdeviceptr devicePtr, long incy);

        /// <summary>
        /// copies n elements from a vector x in GPU memory space to a vector y 
        /// in CPU memory space. Elements in both vectors are assumed to have a 
        /// size of elemSize bytes. Storage spacing between consecutive elements
        /// is incx for the source vector x and incy for the destination vector
        /// y. In general, x points to an object, or part of an object, allocated
        /// via cublasAlloc(). Column major format for two-dimensional matrices
        /// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
        /// is equal to 1, this access a column vector while using an increment 
        /// equal to the leading dimension of the respective matrix accesses a 
        /// row vector.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetVector_64(long n, long elemSize, [In] CUdeviceptr x, long incx, IntPtr y, long incy);

        /// <summary>
        /// copies a tile of rows x cols elements from a matrix A in CPU memory
        /// space to a matrix B in GPU memory space. Each element requires storage
        /// of elemSize bytes. Both matrices are assumed to be stored in column 
        /// major format, with the leading dimension (i.e. number of rows) of 
        /// source matrix A provided in lda, and the leading dimension of matrix B
        /// provided in ldb. In general, B points to an object, or part of an 
        /// object, that was allocated via cublasAlloc().
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetMatrix_64(long rows, long cols, long elemSize, [In] IntPtr A, long lda, CUdeviceptr B, long ldb);

        /// <summary>
        /// copies a tile of rows x cols elements from a matrix A in GPU memory
        /// space to a matrix B in CPU memory space. Each element requires storage
        /// of elemSize bytes. Both matrices are assumed to be stored in column 
        /// major format, with the leading dimension (i.e. number of rows) of 
        /// source matrix A provided in lda, and the leading dimension of matrix B
        /// provided in ldb. In general, A points to an object, or part of an 
        /// object, that was allocated via cublasAlloc().
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetMatrix_64(long rows, long cols, long elemSize, [In] CUdeviceptr A, long lda, IntPtr B, long ldb);

        /// <summary>
        /// cublasSetVectorAsync has the same functionnality as cublasSetVector
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetVectorAsync_64(long n, long elemSize, [In] IntPtr hostPtr, long incx, CUdeviceptr devicePtr, long incy, CUstream stream);
        /// <summary>
        /// cublasGetVectorAsync has the same functionnality as cublasGetVector
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetVectorAsync_64(long n, long elemSize, [In] CUdeviceptr devicePtr, long incx, IntPtr hostPtr, long incy, CUstream stream);

        /// <summary>
        /// cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSetMatrixAsync_64(long rows, long cols, long elemSize, [In] IntPtr A, long lda, CUdeviceptr B, long ldb, CUstream stream);

        /// <summary>
        /// cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
        /// but the transfer is done asynchronously within the CUDA stream passed
        /// in parameter.
        /// </summary>
        /// <returns>
        /// CudaBlas Error Codes: <see cref="CublasStatus.Success"/>, <see cref="CublasStatus.InvalidValue"/>,
        /// <see cref="CublasStatus.MappingError"/>, <see cref="CublasStatus.NotInitialized"/>.
        /// </returns>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGetMatrixAsync_64(long rows, long cols, long elemSize, [In] CUdeviceptr A, long lda, IntPtr B, long ldb, CUstream stream);

        #endregion

        #region BLAS1
        #region host/device independent
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCopyEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      [In] CUdeviceptr x,
                                                      cudaDataType xType,
                                                      long incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      long incy);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScopy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDcopy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCcopy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZcopy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSswap_v2_64(CudaBlasHandle handle,
                                         long n,
                                         CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDswap_v2_64(CudaBlasHandle handle,
                                         long n,
                                         CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCswap_v2_64(CudaBlasHandle handle,
                                         long n,
                                         CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZswap_v2_64(CudaBlasHandle handle,
                                         long n,
                                         CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSwapEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      CUdeviceptr x,
                                                      cudaDataType xType,
                                                      long incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      long incy);

        #endregion

        #region Host pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasNrm2Ex_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     IntPtr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType); /* host or device pointer */

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     IntPtr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotcEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     IntPtr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSnrm2_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        ref float result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDnrm2_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        ref double result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScnrm2_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref float result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDznrm2_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref double result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSdot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        ref float result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDdot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        ref double result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotu_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         ref cuFloatComplex result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotc_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         ref cuFloatComplex result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotu_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         ref cuDoubleComplex result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotc_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         ref cuDoubleComplex result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScalEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     IntPtr alpha,  /* host or device pointer */
                                                     cudaDataType alphaType,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     cudaDataType executionType);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] ref float alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] ref double alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] ref cuFloatComplex alpha, // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsscal_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] ref float alpha, // host or device pointer
                                         CUdeviceptr x,
                                         long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] ref cuDoubleComplex alpha, // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdscal_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] ref double alpha, // host or device pointer
                                         CUdeviceptr x,
                                         long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAxpyEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      IntPtr alpha, /* host or device pointer */
                                                      cudaDataType alphaType,
                                                      CUdeviceptr x,
                                                      cudaDataType xType,
                                                      long incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      long incy,
                                                      cudaDataType executiontype);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIamaxEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      long incx,
                                                      ref long result  /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref long result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIaminEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      long incx,
                                                      ref long result /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAsumEx_64(CudaBlasHandle handle,
                                                     long n,

                                                     [In] CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     IntPtr result,
                                                     cudaDataType resultType, /* host or device pointer */
                                                     cudaDataType executiontype
                                                  );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSasum_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        ref float result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDasum_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        ref double result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScasum_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref float result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDzasum_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         ref double result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] ref float c,  // host or device pointer
                                        [In] ref float s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] ref double c,  // host or device pointer
                                        [In] ref double s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] ref float c,      // host or device pointer
                                        [In] ref cuFloatComplex s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] ref float c,  // host or device pointer
                                        [In] ref float s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] ref double c,            // host or device pointer
                                        [In] ref cuDoubleComplex s);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] ref double c,  // host or device pointer
                                        [In] ref double s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     IntPtr c,  /* host or device pointer */
                                                     IntPtr s,
                                                     cudaDataType csType,
                                                     cudaDataType executiontype);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotm_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In, MarshalAs(UnmanagedType.LPArray, SizeConst = 5)] float[] param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotm_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In, MarshalAs(UnmanagedType.LPArray, SizeConst = 5)] double[] param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotmEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     IntPtr param, /* host or device pointer */
                                                     cudaDataType paramType,
                                                     cudaDataType executiontype);

        #endregion

        #region Device pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasNrm2Ex_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType); /* host or device pointer */

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDotcEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSnrm2_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDnrm2_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScnrm2_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDznrm2_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSdot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDdot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotu_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr result);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdotc_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotu_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdotc_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScalEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr alpha,  /* host or device pointer */
                                                     cudaDataType alphaType,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     cudaDataType executionType);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr alpha,  // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsscal_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         CUdeviceptr x,
                                         long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZscal_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        CUdeviceptr x,
                                        long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdscal_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         CUdeviceptr x,
                                         long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAxpyEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      cudaDataType alphaType,
                                                      CUdeviceptr x,
                                                      cudaDataType xType,
                                                      long incx,
                                                      CUdeviceptr y,
                                                      cudaDataType yType,
                                                      long incy,
                                                      cudaDataType executiontype);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZaxpy_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamax_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIamaxEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      long incx,
                                                      CUdeviceptr result  /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIsamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIdamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIcamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIzamin_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasIaminEx_64(CudaBlasHandle handle,
                                                      long n,
                                                      [In] CUdeviceptr x, cudaDataType xType,
                                                      long incx,
                                                      CUdeviceptr result /* host or device pointer */
                                                    );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasAsumEx_64(CudaBlasHandle handle,
                                                     long n,

                                                     [In] CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr result,
                                                     cudaDataType resultType, /* host or device pointer */
                                                     cudaDataType executiontype
                                                  );

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSasum_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDasum_v2_64(CudaBlasHandle handle,
                                        long n,
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasScasum_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDzasum_v2_64(CudaBlasHandle handle,
                                         long n,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         CUdeviceptr result); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr c,      // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr c,            // host or device pointer
                                        [In] CUdeviceptr s);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdrot_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr c,  // host or device pointer
                                        [In] CUdeviceptr s); // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     CUdeviceptr c,  /* host or device pointer */
                                                     CUdeviceptr s,
                                                     cudaDataType csType,
                                                     cudaDataType executiontype);



        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSrotm_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDrotm_v2_64(CudaBlasHandle handle,
                                        long n,
                                        CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr y,
                                        long incy,
                                        [In] CUdeviceptr param);  // host or device pointer

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasRotmEx_64(CudaBlasHandle handle,
                                                     long n,
                                                     CUdeviceptr x,
                                                     cudaDataType xType,
                                                     long incx,
                                                     CUdeviceptr y,
                                                     cudaDataType yType,
                                                     long incy,
                                                     CUdeviceptr param, /* host or device pointer */
                                                     cudaDataType paramType,
                                                     cudaDataType executiontype);


        #endregion
        #endregion

        #region BLAS2
        #region host/device independent
        #region TRMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);
        #endregion
        #region TBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStbmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtbmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtbmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtbmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A, long lda,
                                         CUdeviceptr x, long incx);
        #endregion
        #region TPMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStpmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtpmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtpmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtpmv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP,
                                         CUdeviceptr x, long incx);
        #endregion
        #region TRSV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr A, long lda,
                                         CUdeviceptr x, long incx);
        #endregion
        #region TPSV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStpsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP,
                                         CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtpsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtpsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtpsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, [In] CUdeviceptr AP,
                                         CUdeviceptr x, long incx);
        #endregion
        #region TBSV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStbsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A,
                                         long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtbsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A,
                                         long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtbsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A,
                                         long lda, CUdeviceptr x, long incx);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtbsv_v2_64(CudaBlasHandle handle, FillMode uplo, Operation trans,
                                         DiagType diag, long n, long k, [In] CUdeviceptr A,
                                         long lda, CUdeviceptr x, long incx);
        #endregion
        #endregion

        #region host pointer
        #region GEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref float beta,  // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref double beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region GBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref float beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref double beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region SYMV/HEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref float alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref float beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref double beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuDoubleComplex alpha,  // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuDoubleComplex beta,   // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region SBMV/HBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] ref float alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref float beta,  // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] ref double alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref double beta,   // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region SPMV/HPMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref float alpha,  // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref float beta,   // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref double beta,  // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuFloatComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] ref cuDoubleComplex beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region GER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSger_v2_64(CudaBlasHandle handle,
                                        long m,
                                        long n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDger_v2_64(CudaBlasHandle handle,
                                        long m,
                                        long n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeru_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgerc_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeru_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgerc_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);
        #endregion
        #region SYR/HER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref cuFloatComplex alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref cuDoubleComplex alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);
        #endregion
        #region SPR/HPR
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref double alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);
        #endregion
        #region SYR2/HER2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref float alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref double alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] ref cuFloatComplex alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo, long n,
                                         [In] ref cuFloatComplex alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuDoubleComplex alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        #endregion
        #region SPR2/HPR2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref float alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref double alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuFloatComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);
        #endregion
        #endregion

        #region device pointer
        #region GEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta,  // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region GBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgbmv_v2_64(CudaBlasHandle handle,
                                         Operation trans,
                                         long m,
                                         long n,
                                         long kl,
                                         long ku,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region SYMV/HEMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta,   // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region SBMV/HBMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta,  // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha,   // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta,   // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhbmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region SPMV/HPMV
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta,   // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta,  // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpmv_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr AP,
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr beta, // host or device pointer
                                         CUdeviceptr y,
                                         long incy);
        #endregion
        #region GER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSger_v2_64(CudaBlasHandle handle,
                                        long m,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDger_v2_64(CudaBlasHandle handle,
                                        long m,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeru_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgerc_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeru_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgerc_v2_64(CudaBlasHandle handle,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);
        #endregion
        #region SYR/HER
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr A,
                                        long lda);
        #endregion
        #region SPR/HPR
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        CUdeviceptr AP);
        #endregion
        #region SYR2/HER2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        long n,
                                        [In] CUdeviceptr alpha, // host or device pointer
                                        [In] CUdeviceptr x,
                                        long incx,
                                        [In] CUdeviceptr y,
                                        long incy,
                                        CUdeviceptr A,
                                        long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo, long n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr A,
                                         long lda);

        #endregion
        #region SPR2/HPR2
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSspr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDspr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha,  // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChpr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhpr2_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         long n,
                                         [In] CUdeviceptr alpha, // host or device pointer
                                         [In] CUdeviceptr x,
                                         long incx,
                                         [In] CUdeviceptr y,
                                         long incy,
                                         CUdeviceptr AP);
        #endregion
        #endregion
        #endregion

        #region BLAS3
        #region host pointer
        #region GEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] ref float alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] ref float beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] ref double alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] ref double beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] ref cuFloatComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] ref cuFloatComplex beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3m_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      ref cuFloatComplex alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      ref cuFloatComplex beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3m_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm3m_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      ref cuDoubleComplex alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      ref cuDoubleComplex beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm3m_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mEx_64(CudaBlasHandle handle,
                                                     Operation transa, Operation transb,
                                                     long m, long n, long k,
                                                     ref cuFloatComplex alpha,
                                                     CUdeviceptr A,
                                                     cudaDataType Atype,
                                                     long lda,
                                                     CUdeviceptr B,
                                                     cudaDataType Btype,
                                                     long ldb,
                                                     ref cuFloatComplex beta,
                                                     CUdeviceptr C,
                                                     cudaDataType Ctype,
                                                     long ldc);




        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] ref cuDoubleComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] ref cuDoubleComplex beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemm_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      ref half alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      ref half beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemm_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmBatched_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      ref half alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      ref half beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc,
                                                      long batchCount);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmBatched_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      long ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      long ldc,
                                                      long batchCount);

        /* IO in FP16/FP32, computation in float */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmEx_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      ref float alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      DataType Atype,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      DataType Btype,
                                                      long ldb,
                                                      ref float beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      DataType Ctype,
                                                      long ldc);



        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmEx_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      IntPtr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      cudaDataType Atype,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      cudaDataType Btype,
                                                      long ldb,
                                                      IntPtr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      cudaDataType Ctype,
                                                      long ldc,
                                                      ComputeType computeType,
                                                      GemmAlgo algo);
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmEx_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      cudaDataType Atype,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      cudaDataType Btype,
                                                      long ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      cudaDataType Ctype,
                                                      long ldc,
                                                      ComputeType computeType,
                                                      GemmAlgo algo);

        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmEx_64(CudaBlasHandle handle,
                                                     Operation transa, Operation transb,
                                                     long m, long n, long k,
                                                     ref cuFloatComplex alpha,
                                                     CUdeviceptr A,
                                                     cudaDataType Atype,
                                                     long lda,
                                                     CUdeviceptr B,
                                                     cudaDataType Btype,
                                                     long ldb,
                                                     ref cuFloatComplex beta,
                                                     CUdeviceptr C,
                                                     cudaDataType Ctype,
                                                     long ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmEx_64(CudaBlasHandle handle,
                                                     Operation transa, Operation transb,
                                                     long m, long n, long k,
                                                     CUdeviceptr alpha,
                                                     CUdeviceptr A,
                                                     cudaDataType Atype,
                                                     long lda,
                                                     CUdeviceptr B,
                                                     cudaDataType Btype,
                                                     long ldb,
                                                     CUdeviceptr beta,
                                                     CUdeviceptr C,
                                                     cudaDataType Ctype,
                                                     long ldc);


        /* SYRK */


        /* IO in FP16/FP32, computation in float */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmEx_64(CudaBlasHandle handle,
                                                      Operation transa,
                                                      Operation transb,
                                                      long m,
                                                      long n,
                                                      long k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr A,
                                                      DataType Atype,
                                                      long lda,
                                                      CUdeviceptr B,
                                                      DataType Btype,
                                                      long ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr C,
                                                      DataType Ctype,
                                                      long ldc);



        #endregion
        #region SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrk_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        long n,
                                        long k,
                                        [In] ref float alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] ref float beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] ref double alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] ref double beta,  //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] ref cuFloatComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] ref cuFloatComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] ref cuDoubleComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] ref cuDoubleComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkEx_64(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              long n,
                                                              long k,
                                                              ref cuFloatComplex alpha, /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              long lda,
                                                              ref cuFloatComplex beta, /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              long ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkEx_64(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              long n,
                                                              long k,
                                                              CUdeviceptr alpha, /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              long lda,
                                                              CUdeviceptr beta, /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              long ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk3mEx_64(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              long n,
                                                              long k,
                                                              ref cuFloatComplex alpha,
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              long lda,
                                                              ref cuFloatComplex beta,
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              long ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk3mEx_64(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              long n,
                                                              long k,
                                                              CUdeviceptr lpha,
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              long lda,
                                                              CUdeviceptr beta,
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              long ldc);

        #endregion
        #region HERK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] ref float alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] ref float beta,   //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherk_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        long n,
                                        long k,
                                        [In] ref double alpha,  //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] ref double beta,  //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        #endregion
        #region SYR2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] ref float alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] ref float beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] ref double alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] ref double beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] ref cuFloatComplex alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] ref cuFloatComplex beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] ref cuDoubleComplex alpha,  //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] ref cuDoubleComplex beta,  //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);
        #endregion
        #region HER2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] ref cuFloatComplex alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] ref float beta,   //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] ref cuDoubleComplex alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] ref double beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkEx_64(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              long n,
                                                              long k,
                                                              ref float alpha,  /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              long lda,
                                                              ref float beta,   /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              long ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk3mEx_64(CudaBlasHandle handle,
                                                               FillMode uplo,
                                                               Operation trans,
                                                               long n,
                                                               long k,
                                                               ref float alpha,
                                                               CUdeviceptr A, cudaDataType Atype,
                                                               long lda,
                                                               ref float beta,
                                                               CUdeviceptr C,
                                                               cudaDataType Ctype,
                                                               long ldc);


        /* IO in Int8 complex/cuComplex, computation in cuComplex */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkEx_64(CudaBlasHandle handle,
                                                              FillMode uplo,
                                                              Operation trans,
                                                              long n,
                                                              long k,
                                                              CUdeviceptr alpha,  /* host or device pointer */
                                                              CUdeviceptr A,
                                                              cudaDataType Atype,
                                                              long lda,
                                                              CUdeviceptr beta,   /* host or device pointer */
                                                              CUdeviceptr C,
                                                              cudaDataType Ctype,
                                                              long ldc);

        /* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk3mEx_64(CudaBlasHandle handle,
                                                               FillMode uplo,
                                                               Operation trans,
                                                               long n,
                                                               long k,
                                                               CUdeviceptr alpha,
                                                               CUdeviceptr A, cudaDataType Atype,
                                                               long lda,
                                                               CUdeviceptr beta,
                                                               CUdeviceptr C,
                                                               cudaDataType Ctype,
                                                               long ldc);

        /* SYR2K */

        #endregion
        #region SYRKX : eXtended SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] ref float alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] ref float beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] ref double alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] ref double beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] ref cuFloatComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] ref cuFloatComplex beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] ref cuDoubleComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] ref cuDoubleComplex beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);
        #endregion

        #region HERKX : eXtended HERK         
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] ref cuFloatComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] ref float beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] ref cuDoubleComplex alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] ref double beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);
        #endregion

        #region SYMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] ref float alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] ref float beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] ref double alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] ref double beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] ref cuFloatComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] ref cuFloatComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] ref cuDoubleComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        #endregion
        #region HEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] ref cuFloatComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] ref cuFloatComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] ref cuDoubleComplex alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] ref cuDoubleComplex beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        #endregion
        #region TRSM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] ref float alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         CUdeviceptr B,
                                         long ldb);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] ref double alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         CUdeviceptr B,
                                         long ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsm_v2_64(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] ref cuFloatComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        CUdeviceptr B,
                                        long ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsm_v2_64(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] ref cuDoubleComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        CUdeviceptr B,
                                        long ldb);

        #endregion
        #region TRMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrmm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] ref float alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrmm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] ref double alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrmm_v2_64(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] ref cuFloatComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrmm_v2_64(CudaBlasHandle handle, SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] ref cuDoubleComplex alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        CUdeviceptr C,
                                        long ldc);
        #endregion
        #endregion

        #region device pointer
        #region GEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemm_v2_64(CudaBlasHandle handle,
                                        Operation transa,
                                        Operation transb,
                                        long m,
                                        long n,
                                        long k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);
        #endregion
        #region SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrk_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        long n,
                                        long k,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr beta, //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr beta,  //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);
        #endregion
        #region HERK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherk_v2_64(CudaBlasHandle handle,
                                         FillMode uplo,
                                         Operation trans,
                                         long n,
                                         long k,
                                         [In] CUdeviceptr alpha,  //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr beta,   //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherk_v2_64(CudaBlasHandle handle,
                                        FillMode uplo,
                                        Operation trans,
                                        long n,
                                        long k,
                                        [In] CUdeviceptr alpha,  //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr beta,  //host or device pointer  
                                        CUdeviceptr C,
                                        long ldc);

        #endregion
        #region SYR2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyr2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] CUdeviceptr alpha,  //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] CUdeviceptr beta,  //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);
        #endregion
        #region HER2K
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCher2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] CUdeviceptr beta,   //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZher2k_v2_64(CudaBlasHandle handle,
                                          FillMode uplo,
                                          Operation trans,
                                          long n,
                                          long k,
                                          [In] CUdeviceptr alpha, //host or device pointer  
                                          [In] CUdeviceptr A,
                                          long lda,
                                          [In] CUdeviceptr B,
                                          long ldb,
                                          [In] CUdeviceptr beta, //host or device pointer  
                                          CUdeviceptr C,
                                          long ldc);

        #endregion
        #region SYRKX : eXtended SYRK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsyrkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);
        #endregion

        #region HERKX : eXtended HERK
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCherkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZherkx_64(CudaBlasHandle handle,
                                                    FillMode uplo,
                                                    Operation trans,
                                                    long n,
                                                    long k,
                                                    [In] CUdeviceptr alpha, /* host or device pointer */
                                                    [In] CUdeviceptr A,
                                                    long lda,
                                                    [In] CUdeviceptr B,
                                                    long ldb,
                                                    [In] CUdeviceptr beta, /* host or device pointer */
                                                    [In] CUdeviceptr C,
                                                    long ldc);
        #endregion
        #region SYMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZsymm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        #endregion
        #region HEMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasChemm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZhemm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         [In] CUdeviceptr beta, //host or device pointer  
                                         CUdeviceptr C,
                                         long ldc);

        #endregion
        #region TRSM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         CUdeviceptr B,
                                         long ldb);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         CUdeviceptr B,
                                         long ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsm_v2_64(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        CUdeviceptr B,
                                        long ldb);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsm_v2_64(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        CUdeviceptr B,
                                        long ldb);

        #endregion
        #region TRMM
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrmm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrmm_v2_64(CudaBlasHandle handle,
                                         SideMode side,
                                         FillMode uplo,
                                         Operation trans,
                                         DiagType diag,
                                         long m,
                                         long n,
                                         [In] CUdeviceptr alpha, //host or device pointer  
                                         [In] CUdeviceptr A,
                                         long lda,
                                         [In] CUdeviceptr B,
                                         long ldb,
                                         CUdeviceptr C,
                                         long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrmm_v2_64(CudaBlasHandle handle,
                                        SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        CUdeviceptr C,
                                        long ldc);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrmm_v2_64(CudaBlasHandle handle, SideMode side,
                                        FillMode uplo,
                                        Operation trans,
                                        DiagType diag,
                                        long m,
                                        long n,
                                        [In] CUdeviceptr alpha, //host or device pointer  
                                        [In] CUdeviceptr A,
                                        long lda,
                                        [In] CUdeviceptr B,
                                        long ldb,
                                        CUdeviceptr C,
                                        long ldc);
        #endregion
        #endregion
        #endregion

        #region CUBLAS BLAS-like extension
        #region GEAM
        #region device ptr
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);
        #endregion
        #region host ptr
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  ref float alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  ref float beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  ref double alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  ref double beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  ref cuFloatComplex alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  ref cuFloatComplex beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgeam_64(CudaBlasHandle handle,
                                                  Operation transa,
                                                  Operation transb,
                                                  long m,
                                                  long n,
                                                  ref cuDoubleComplex alpha, /* host or device pointer */
                                                  CUdeviceptr A,
                                                  long lda,
                                                  ref cuDoubleComplex beta, /* host or device pointer */
                                                  CUdeviceptr B,
                                                  long ldb,
                                                  CUdeviceptr C,
                                                  long ldc);
        #endregion
        #endregion


        #region DGMM

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSdgmm_64(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr x,
                                                  long incx,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDdgmm_64(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr x,
                                                  long incx,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCdgmm_64(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr x,
                                                  long incx,
                                                  CUdeviceptr C,
                                                  long ldc);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZdgmm_64(CudaBlasHandle handle,
                                                  SideMode mode,
                                                  long m,
                                                  long n,
                                                  CUdeviceptr A,
                                                  long lda,
                                                  CUdeviceptr x,
                                                  long incx,
                                                  CUdeviceptr C,
                                                  long ldc);
        #endregion
        #endregion
        //Ab hier NEU

        #region BATCH GEMM
        #region device pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   CUdeviceptr alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   CUdeviceptr beta,   /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   CUdeviceptr alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   CUdeviceptr beta,  /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   CUdeviceptr alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   CUdeviceptr beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   CUdeviceptr alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   CUdeviceptr beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);


        //Missing before:
        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mBatched_64(CudaBlasHandle handle,
                                                          Operation transa,
                                                          Operation transb,
                                                          long m,
                                                          long n,
                                                          long k,
                                                          ref cuFloatComplex alpha, /* host or device pointer */
                                                          CUdeviceptr Aarray,
                                                          long lda,
                                                          CUdeviceptr Barray,
                                                          long ldb,
                                                          ref cuFloatComplex beta, /* host or device pointer */
                                                          CUdeviceptr Carray,
                                                          long ldc,
                                                          long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mBatched_64(CudaBlasHandle handle,
                                                          Operation transa,
                                                          Operation transb,
                                                          long m,
                                                          long n,
                                                          long k,
                                                          CUdeviceptr alpha, /* host or device pointer */
                                                          CUdeviceptr Aarray,
                                                          long lda,
                                                          CUdeviceptr Barray,
                                                          long ldb,
                                                          CUdeviceptr beta, /* host or device pointer */
                                                          CUdeviceptr Carray,
                                                          long ldc,
                                                          long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mStridedBatched_64(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         long m,
                                                                         long n,
                                                                         long k,
                                                                 ref cuFloatComplex alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 ref cuFloatComplex beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemm3mStridedBatched_64(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         long m,
                                                                         long n,
                                                                         long k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);




        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmBatchedEx_64(CudaBlasHandle handle,
                                                              Operation transa,
                                                              Operation transb,
                                                              long m,
                                                              long n,
                                                              long k,
                                                      CUdeviceptr alpha, /* host or device pointer */
                                                      CUdeviceptr Aarray,
                                                      cudaDataType Atype,
                                                      long lda,
                                                      CUdeviceptr Barray,
                                                      cudaDataType Btype,
                                                      long ldb,
                                                      CUdeviceptr beta, /* host or device pointer */
                                                      CUdeviceptr Carray,
                                                      cudaDataType Ctype,
                                                      long ldc,
                                                      long batchCount,
                                                      ComputeType computeType,
                                                      GemmAlgo algo);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmStridedBatchedEx_64(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         long m,
                                                                         long n,
                                                                         long k,
                                                                 CUdeviceptr alpha,  /* host or device pointer */
                                                                 CUdeviceptr A,
                                                                 cudaDataType Atype,
                                                                 long lda,
                                                                 long strideA,   /* purposely signed */
                                                                 CUdeviceptr B,
                                                                 cudaDataType Btype,
                                                                 long ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   /* host or device pointer */
                                                                 CUdeviceptr C,
                                                                 cudaDataType Ctype,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount,
                                                                 ComputeType computeType,
                                                                 GemmAlgo algo);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer   
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 CUdeviceptr alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 CUdeviceptr alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 CUdeviceptr alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 CUdeviceptr beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);


        #endregion
        #region host pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   ref float alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   ref float beta,   /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   ref double alpha,  /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   ref double beta,  /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   ref cuFloatComplex alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   ref cuFloatComplex beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);


        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmBatched_64(CudaBlasHandle handle,
                                   Operation transa,
                                   Operation transb,
                                   long m,
                                   long n,
                                   long k,
                                   ref cuDoubleComplex alpha, /* host or device pointer */
                                   CUdeviceptr Aarray,
                                   long lda,
                                   CUdeviceptr Barray,
                                   long ldb,
                                   ref cuDoubleComplex beta, /* host or device pointer */
                                   CUdeviceptr Carray,
                                   long ldc,
                                   long batchCount);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmBatchedEx_64(CudaBlasHandle handle,
                                                              Operation transa,
                                                              Operation transb,
                                                              long m,
                                                              long n,
                                                              long k,
                                                      IntPtr alpha, /* host or device pointer */
                                                      CUdeviceptr Aarray,
                                                      cudaDataType Atype,
                                                      long lda,
                                                      CUdeviceptr Barray,
                                                      cudaDataType Btype,
                                                      long ldb,
                                                      IntPtr beta, /* host or device pointer */
                                                      CUdeviceptr Carray,
                                                      cudaDataType Ctype,
                                                      long ldc,
                                                      long batchCount,
                                                      cudaDataType computeType,
                                                      GemmAlgo algo);

        /// <summary>
		/// </summary>
		[DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasGemmStridedBatchedEx_64(CudaBlasHandle handle,
                                                                         Operation transa,
                                                                         Operation transb,
                                                                         long m,
                                                                         long n,
                                                                         long k,
                                                                 IntPtr alpha,  /* host or device pointer */
                                                                 CUdeviceptr A,
                                                                 cudaDataType Atype,
                                                                 long lda,
                                                                 long strideA,   /* purposely signed */
                                                                 CUdeviceptr B,
                                                                 cudaDataType Btype,
                                                                 long ldb,
                                                                 long strideB,
                                                                 IntPtr beta,   /* host or device pointer */
                                                                 CUdeviceptr C,
                                                                 cudaDataType Ctype,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount,
                                                                 cudaDataType computeType,
                                                                 GemmAlgo algo);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasSgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 ref float alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 ref float beta,   // host or device pointer   
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 ref double alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 ref double beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 ref cuFloatComplex alpha,  // host or device pointer
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 ref cuFloatComplex beta,   // host or device pointer$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 ref cuDoubleComplex alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 ref cuDoubleComplex beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasHgemmStridedBatched_64(CudaBlasHandle handle,
                                                                 Operation transa,
                                                                 Operation transb,
                                                                 long m,
                                                                 long n,
                                                                 long k,
                                                                 ref half alpha,  // host or device poi$
                                                                 CUdeviceptr A,
                                                                 long lda,
                                                                 long strideA,   // purposely signed 
                                                                 CUdeviceptr B,
                                                                 long ldb,
                                                                 long strideB,
                                                                 ref half beta,   // host or device poi$
                                                                 CUdeviceptr C,
                                                                 long ldc,
                                                                 long strideC,
                                                                 long batchCount);

        #endregion
        #endregion





        #region TRSM - Batched Triangular Solver
        #region device pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          CUdeviceptr alpha,           /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          CUdeviceptr alpha,          /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          CUdeviceptr alpha,       /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          CUdeviceptr salpha, /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);


        #endregion
        #region host pointer
        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasStrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          ref float alpha,           /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasDtrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          ref double alpha,          /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasCtrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          ref cuFloatComplex alpha,       /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);

        /// <summary>
        /// </summary>
        [DllImport(CUBLAS_API_DLL_NAME)]
        public static extern CublasStatus cublasZtrsmBatched_64(CudaBlasHandle handle,
                                                          SideMode side,
                                                          FillMode uplo,
                                                          Operation trans,
                                                          DiagType diag,
                                                          long m,
                                                          long n,
                                                          ref cuDoubleComplex alpha, /*Host or Device Pointer*/
                                                          CUdeviceptr A,
                                                          long lda,
                                                          CUdeviceptr B,
                                                          long ldb,
                                                          long batchCount);


        #endregion
        #endregion




        #endregion
    }
}
