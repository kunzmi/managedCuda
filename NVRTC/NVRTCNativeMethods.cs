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
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ManagedCuda.NVRTC
{
    /// <summary/>
    public static class NVRTCNativeMethods
    {
        internal const string NVRTC_API_DLL_NAME = "nvrtc64_120_0";

#if (NETCOREAPP)
        internal const string NVRTC_API_DLL_NAME_LINUX = "nvrtc";

        static NVRTCNativeMethods()
        {
            NativeLibrary.SetDllImportResolver(typeof(NVRTCNativeMethods).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            IntPtr libHandle = IntPtr.Zero;

            if (libraryName == NVRTC_API_DLL_NAME)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    bool res = NativeLibrary.TryLoad(NVRTC_API_DLL_NAME_LINUX, assembly, DllImportSearchPath.SafeDirectories, out libHandle);
                    if (!res)
                    {
                        Debug.WriteLine("Failed to load '" + NVRTC_API_DLL_NAME_LINUX + "' shared library. Falling back to (Windows-) default library name '"
                            + NVRTC_API_DLL_NAME + "'. Check LD_LIBRARY_PATH environment variable for correct paths.");
                    }
                }
            }
            //On Windows, use the default library name
            return libHandle;
        }
#endif

        [DllImport(NVRTC_API_DLL_NAME, EntryPoint = "nvrtcGetErrorString")]
        internal static extern IntPtr nvrtcGetErrorStringInternal(nvrtcResult result);

        /// <summary>
        /// helper function that stringifies the given #nvrtcResult code, e.g., NVRTC_SUCCESS to
        /// "NVRTC_SUCCESS". For unrecognized enumeration values, it returns "NVRTC_ERROR unknown"
        /// </summary>
        /// <param name="result">CUDA Runtime Compiler API result code.</param>
        /// <returns>Message string for the given nvrtcResult code.</returns>
        public static string nvrtcGetErrorString(nvrtcResult result)
        {
            IntPtr ptr = nvrtcGetErrorStringInternal(result);
            return Marshal.PtrToStringAnsi(ptr);
        }


        /// <summary>
        /// sets the output parameters \p major and \p minor
        /// with the CUDA Runtime Compiler version number.
        /// </summary>
        /// <param name="major">CUDA Runtime Compiler major version number.</param>
        /// <param name="minor">CUDA Runtime Compiler minor version number.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcVersion(ref int major, ref int minor);


        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetNumSupportedArchs(ref int numArchs);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetSupportedArchs(int[] supportedArchs);

        /// <summary>
        /// creates an instance of ::nvrtcProgram with the
        /// given input parameters, and sets the output parameter \p prog with it.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compiler program.</param>
        /// <param name="src">CUDA program source.</param>
        /// <param name="name">CUDA program name.<para/>
        /// name can be NULL; "default_program" is used when name is NULL.</param>
        /// <param name="numHeaders">Number of headers used.<para/>
        /// numHeaders must be greater than or equal to 0.</param>
        /// <param name="headers">Sources of the headers.<para/>
        /// headers can be NULL when numHeaders is 0.</param>
        /// <param name="includeNames">Name of each header by which they can be included in the CUDA program source.<para/>
        /// includeNames can be NULL when numHeaders is 0.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(ref nvrtcProgram prog,
                               [MarshalAs(UnmanagedType.LPStr)] string src,
                               [MarshalAs(UnmanagedType.LPStr)] string name,
                               int numHeaders,
                               IntPtr[] headers,
                               IntPtr[] includeNames);




        /// <summary>
        /// destroys the given program.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compiler program.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref nvrtcProgram prog);

        /// <summary>
        /// compiles the given program.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compiler program.</param>
        /// <param name="numOptions">Number of compiler options passed.</param>
        /// <param name="options">Compiler options in the form of C string array.<para/>
        /// options can be NULL when numOptions is 0.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, IntPtr[] options);

        /// <summary>
        /// sets \p ptxSizeRet with the size of the PTX generated by the previous compilation of prog (including the trailing NULL).
        /// </summary>
        /// <param name="prog">CUDA Runtime Compiler program.</param>
        /// <param name="ptxSizeRet">Size of the generated PTX (including the trailing NULL).</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, ref SizeT ptxSizeRet);

        /// <summary>
        /// stores the PTX generated by the previous compilation
        /// of prog in the memory pointed by ptx.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compiler program.</param>
        /// <param name="ptx">Compiled result.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(nvrtcProgram prog, byte[] ptx);

        /// <summary>
        /// nvrtcGetCUBINSize sets \p cubinSizeRet with the size of the cubin
        /// generated by the previous compilation of \p prog.The value of
        /// cubinSizeRet is set to 0 if the value specified to \c -arch is a
        /// virtual architecture instead of an actual architecture.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compilation program.</param>
        /// <param name="cubinSizeRet">Size of the generated cubin.</param>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, ref SizeT cubinSizeRet);

        /// <summary>
        /// nvrtcGetCUBIN stores the cubin generated by the previous compilation
        /// of \p prog in the memory pointed by \p cubin.No cubin is available
        /// if the value specified to \c -arch is a virtual architecture instead
        /// of an actual architecture.
        /// </summary>
        /// <param name="prog">prog CUDA Runtime Compilation program.</param>
        /// <param name="cubin">cubin  Compiled and assembled result.</param>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, byte[] cubin);

        /// <summary>
        /// nvrtcGetNVVMSize sets \p nvvmSizeRet with the size of the NVVM
        /// generated by the previous compilation of \p prog.The value of
        /// nvvmSizeRet is set to 0 if the program was not compiled with 
        /// -dlto.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compilation program.</param>
        /// <param name="nvvmSizeRet">Size of the generated NVVM.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        [Obsolete("This function will be removed in a future release. Please use nvrtcGetLTOIR (and nvrtcGetLTOIRSize) instead.")]
        public static extern nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, ref SizeT nvvmSizeRet);

        /// <summary>
        /// nvrtcGetNVVM stores the NVVM generated by the previous compilation
        /// of \p prog in the memory pointed by \p nvvm.
        /// The program must have been compiled with -dlto,
        /// otherwise will return an error.
        /// </summary>
        /// <param name="prog">prog CUDA Runtime Compilation program.</param>
        /// <param name="nvvm">nvvm Compiled result.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        [Obsolete("This function will be removed in a future release. Please use nvrtcGetLTOIR (and nvrtcGetLTOIRSize) instead.")]
        public static extern nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, byte[] nvvm);

        /// <summary>
        /// nvrtcGetLTOIRSize sets \p LTOIRSizeRet with the size of the LTO IR
        /// generated by the previous compilation of \p prog.The value of
        /// LTOIRSizeRet is set to 0 if the program was not compiled with 
        /// -dlto.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compilation program.</param>
        /// <param name="LTOIRSizeRet">Size of the generated LTO IR.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, ref SizeT LTOIRSizeRet);

        /// <summary>
        /// nvrtcGetLTOIR stores the LTO IR generated by the previous compilation
        /// of \p prog in the memory pointed by \p LTOIR. No LTO IR is available
        /// if the program was compiled without \c -dlto.
        /// </summary>
        /// <param name="prog">prog CUDA Runtime Compilation program.</param>
        /// <param name="LTOIR">LTOIR Compiled result.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, byte[] LTOIR);

        /// <summary>
        /// nvrtcGetOptiXIRSize sets the value of \p optixirSizeRet with the size of the OptiX IR
        /// generated by the previous compilation of \p prog. The value of
        /// nvrtcGetOptiXIRSize is set to 0 if the program was compiled with
        /// options incompatible with OptiX IR generation.
        /// </summary>
        /// <param name="prog">prog CUDA Runtime Compilation program.</param>
        /// <param name="optixirSizeRet">Size of the generated LTO IR.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, ref SizeT optixirSizeRet);

        /// <summary>
        /// nvrtcGetOptiXIR stores the OptiX IR generated by the previous compilation
        /// of \p prog in the memory pointed by \p optixir. No OptiX IR is available
        /// if the program was compiled with options incompatible with OptiX IR generation.
        /// </summary>
        /// <param name="prog">prog CUDA Runtime Compilation program.</param>
        /// <param name="optixir">Optix IR Compiled result.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, byte[] optixir);

        /// <summary>
        /// sets logSizeRet with the size of the log generated by the previous compilation of prog (including the trailing NULL).
        /// </summary>
        /// <param name="prog">CUDA Runtime Compiler program.</param>
        /// <param name="logSizeRet">Size of the compilation log (including the trailing NULL).</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, ref SizeT logSizeRet);

        /// <summary>
        /// stores the log generated by the previous compilation of prog in the memory pointed by log.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compiler program.</param>
        /// <param name="log">Compilation log.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, byte[] log);


        /// <summary>
        /// nvrtcAddNameExpression notes the given name expression
        /// denoting a __global__ function or function template
        /// instantiation.<para/>
        /// The identical name expression string must be provided on a subsequent
        /// call to nvrtcGetLoweredName to extract the lowered name.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compilation program.</param>
        /// <param name="name_expression">constant expression denoting a __global__ function or function template instantiation.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, [MarshalAs(UnmanagedType.LPStr)] string name_expression);


        /// <summary>
        /// nvrtcGetLoweredName extracts the lowered (mangled) name
        /// for a __global__ function or function template instantiation,
        /// and updates *lowered_name to point to it. The memory containing
        /// the name is released when the NVRTC program is destroyed by 
        /// nvrtcDestroyProgram.<para/>
        /// The identical name expression must have been previously
        /// provided to nvrtcAddNameExpression.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compilation program.</param>
        /// <param name="name_expression">constant expression denoting a __global__ function or function template instantiation.</param>
        /// <param name="lowered_name">initialized by the function to point to a C string containing the lowered (mangled) name corresponding to the provided name expression.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                        [MarshalAs(UnmanagedType.LPStr)] string name_expression, ref IntPtr lowered_name);



        /// <summary>
        /// retrieve the current size of the PCH Heap.
        /// </summary>
        /// <param name="ret">pointer to location where the size of the PCH Heap will be stored</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPCHHeapSize(ref SizeT ret);

        /// <summary>
        /// Set the size of the PCH Heap. The requested size may be rounded up to a platform dependent
        /// alignment (e.g.page size). If the PCH Heap has already been allocated, the heap memory will 
        /// be freed and a new PCH Heap will be allocated.
        /// </summary>
        /// <param name="size">requested size of the PCH Heap, in bytes</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcSetPCHHeapSize(SizeT size);

        /// <summary>
        /// Returns the PCH creation status. <para/>
        /// NVRTC_SUCCESS indicates that the PCH was successfully created. <para/>
        /// NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED indicates that no PCH creation 
        /// was attempted, either because PCH functionality was not requested during
        /// the preceding nvrtcCompileProgram call, or automatic PCH processing was
        /// requested, and compiler chose not to create a PCH file. <para/>
        /// NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED indicates that a PCH file could
        /// potentially have been created, but the compiler ran out space in the PCH
        /// heap. In this scenario, the nvrtcGetPCHHeapSizeRequired() can be used to
        /// query the required heap size, the heap can be reallocated for this size with
        /// nvrtcSetPCHHeapSize() and PCH creation may be reattempted again invoking
        /// nvrtcCompileProgram() with a new NVRTC program instance. <para/>
        /// NVRTC_ERROR_PCH_CREATE indicates that an error condition prevented the
        /// PCH file from being created.
        /// </summary>
        /// <param name="prog">CUDA Runtime Compilation program.</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog);

        /// <summary>
        /// retrieve the required size of the PCH heap required to compile the given program. The size retrieved using this function is only valid if nvrtcGetPCHCreateStatus() returned NVRTC_SUCCESS or NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED
        /// </summary>
        /// <param name="prog">CUDA Runtime Compilation program.</param>
        /// <param name="size">pointer to location where the required size of the PCH Heap will be stored</param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, ref SizeT size);

        /// <summary>
        /// nvrtcSetFlowCallback registers a callback function that the compiler 
        ///          will invoke at different points during a call to nvrtcCompileProgram,
        ///          and the callback function can decide whether to cancel compilation by
        ///          returning specific values.
        /// <para/>
        /// The callback function must satisfy the following constraints:<para/>
        /// (1) Its signature should be:<para/>
        /// int callback(void* param1, void* param2);<para/>
        /// When invoking the callback, the compiler will always pass \p payload to
        /// param1 so that the callback may make decisions based on \p payload.It'll
        ///     always pass NULL to param2 for now which is reserved for future extensions.<para/>
        /// (2) It must return 1 to cancel compilation or 0 to continue. <para/>
        ///     Other return values are reserved for future use.<para/>
        /// (3) It must return consistent values.Once it returns 1 at one point, it must
        ///     return 1 in all following invocations during the current nvrtcCompileProgram
        /// call in progress.<para/>    
        /// (4) It must be thread-safe.<para/>
        /// (5) It must not invoke any nvrtc/libnvvm/ptx APIs.
        /// </summary>
        /// <param name="prog"></param>
        /// <param name="callback"></param>
        /// <param name="payload"></param>
        /// <returns></returns>
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog, setFlowCallback callback, IntPtr payload);
    }
}
