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

namespace ManagedCuda.NvJitLink
{
    /// <summary/>
    public static class NvJitLinkNativeMethods
    {
        internal const string NVJITLINK_API_DLL_NAME = "nvJitLink_120_0";

#if (NETCOREAPP)
        internal const string NVJITLINK_API_DLL_NAME_LINUX = "nvJitLink";

        static NvJitLinkNativeMethods()
        {
            NativeLibrary.SetDllImportResolver(typeof(NvJitLinkNativeMethods).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            IntPtr libHandle = IntPtr.Zero;

            if (libraryName == NVJITLINK_API_DLL_NAME)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    bool res = NativeLibrary.TryLoad(NVJITLINK_API_DLL_NAME_LINUX, assembly, DllImportSearchPath.SafeDirectories, out libHandle);
                    if (!res)
                    {
                        Debug.WriteLine("Failed to load '" + NVJITLINK_API_DLL_NAME_LINUX + "' shared library. Falling back to (Windows-) default library name '"
                            + NVJITLINK_API_DLL_NAME + "'. Check LD_LIBRARY_PATH environment variable for correct paths.");
                    }
                }
            }
            //On Windows, use the default library name
            return libHandle;
        }
#endif

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkCreate_12_6(
              ref nvJitLinkHandle handle, uint numOptions, IntPtr[] options);

        /// <summary>
        /// nvJitLinkCreate creates an instance of nvJitLinkHandle with the given input options, and sets the output parameter \p handle.<para/>
        /// nvJitLink supports the link options below.<para/>
        /// Option names are prefixed with a single dash(-).<para/>
        /// Options that take a value have an assignment operator (=)
        /// followed by the option value, with no spaces, e.g. "-arch=sm_90".<para/>
        ///
        /// The supported options are:<para/>
        /// - -arch=sm_&lt;N&gt;<para/>
        /// Pass SM architecture value. See nvcc for valid values of &lt;N&gt;. Can use compute_N value instead if only generating PTX. This is a required option.<para/>
        /// - -maxrregcount=&lt;N&gt;<para/>
        /// Maximum register count.<para/>
        /// - -time <para/>
        /// Print timing information to InfoLog.<para/>
        /// - -verbose <para/>
        /// Print verbose messages to InfoLog.<para/>
        /// - -lto <para/>
        /// Do link time optimization.<para/>
        /// - -ptx <para/>
        /// Emit ptx after linking instead of cubin; only supported with -lto<para/>
        /// - -O&lt;n&gt;<para/>
        /// Optimization level.<para/>
        /// - -g <para/>
        /// Generate debug information.<para/>
        /// - -lineinfo <para/>
        /// Generate line information.<para/>
        /// - -ftz=&lt; N &gt;<para/>
        /// Flush to zero.<para/>
        /// - -prec-div=&lt; N &gt;<para/>
        /// Precise divide.<para/>
        /// - -prec-sqrt=&lt; N &gt;<para/>
        /// Precise square root.<para/>
        /// - -fma=&lt; N &gt;<para/>
        /// Fast multiply add.<para/>
        /// - -kernels-used=&lt; name &gt;<para/>
        /// Pass list of kernels that are used; any not in the list can be removed.
        /// This option can be specified multiple times.<para/>
        /// - -variables-used=&lt; name &gt;<para/>
        /// Pass list of variables that are used; any not in the list can be removed.
        /// This option can be specified multiple times.<para/>
        /// - -optimize-unused-variables <para/>
        /// Normally device code optimization is limited by not knowing what the
        ///   host code references. With this option it can assume that if a variable
        ///   is not referenced in device code then it can be removed.<para/>
        /// - -Xptxas=&lt; opt &gt;<para/>
        /// Pass &lt; opt &gt; to ptxas. This option can be called multiple times.<para/>
        /// - -Xnvvm=&lt; opt &gt;<para/>
        /// Pass &lt;opt&gt; to nvvm. This option can be called multiple times.<para/>
        /// </summary>
        /// <param name="handle">Address of nvJitLink handle.</param>
        /// <param name="options">Array of size \p numOptions of option strings.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkCreate(ref nvJitLinkHandle handle, string[] options)
        {
            IntPtr[] stringPtrs = null;
            uint numOptions = 0;
            nvJitLinkResult retVal = nvJitLinkResult.Success;

            try
            {
                int arraySize = 0;
                if (options != null)
                {
                    arraySize = options.Length;
                    numOptions = (uint)arraySize;
                    stringPtrs = new IntPtr[arraySize];
                }

                for (int i = 0; i < arraySize; i++)
                {
                    stringPtrs[i] = Marshal.StringToHGlobalAnsi(options[i]);
                }

                retVal = __nvJitLinkCreate_12_6(ref handle, numOptions, stringPtrs);
            }
            catch
            {
                retVal = nvJitLinkResult.ErrorInternal;
            }
            finally
            {
                for (uint i = 0; i < numOptions; i++)
                {
                    Marshal.FreeHGlobal(stringPtrs[i]);
                }
            }
            return retVal;

        }


        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkDestroy_12_6(ref nvJitLinkHandle handle);


        /// <summary>
        /// nvJitLinkDestroy frees the memory associated with the given handle and sets it to NULL.
        /// </summary>
        /// <param name="handle">Address of nvJitLink handle.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkDestroy(ref nvJitLinkHandle handle)
        {
            return __nvJitLinkDestroy_12_6(ref handle);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkAddData_12_6(
          nvJitLinkHandle handle, nvJitLinkInputType inputType, IntPtr data,
          SizeT size, [MarshalAs(UnmanagedType.LPStr)] string name); // name can be null

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkAddData_12_6(
          nvJitLinkHandle handle, nvJitLinkInputType inputType, byte[] data,
          SizeT size, [MarshalAs(UnmanagedType.LPStr)] string name); // name can be null

        /// <summary>
        /// nvJitLinkAddData adds data image to the link. 
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="inputType">kind of input.</param>
        /// <param name="data">pointer to data image in memory.</param>
        /// <param name="size"></param>
        /// <param name="name">name of input object.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle,
            nvJitLinkInputType inputType, IntPtr data, SizeT size, string name) // name can be null
        {
            return __nvJitLinkAddData_12_6(handle, inputType, data, size, name);
        }
        /// <summary>
        /// nvJitLinkAddData adds data image to the link. 
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="inputType">kind of input.</param>
        /// <param name="data">pointer to data image in memory.</param>
        /// <param name="name">name of input object.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle,
            nvJitLinkInputType inputType, byte[] data, string name) // name can be null
        {
            return __nvJitLinkAddData_12_6(handle, inputType, data, data.Length, name);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkAddFile_12_6(nvJitLinkHandle handle,
            nvJitLinkInputType inputType, [MarshalAs(UnmanagedType.LPStr)] string fileName); // includes path to file

        /// <summary>
        /// nvJitLinkAddFile reads data from file and links it in. 
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="inputType">kind of input.</param>
        /// <param name="fileName">name of file.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkAddFile(
          nvJitLinkHandle handle,
          nvJitLinkInputType inputType,
          string fileName) // includes path to file
        {
            return __nvJitLinkAddFile_12_6(handle, inputType, fileName);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkComplete_12_6(nvJitLinkHandle handle);

        /// <summary>
        /// nvJitLinkComplete does the actual link.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle)
        {
            return __nvJitLinkComplete_12_6(handle);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetLinkedCubinSize_12_6(nvJitLinkHandle handle, ref SizeT size);

        /// <summary>
        /// nvJitLinkGetLinkedCubinSize gets the size of the linked cubin.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="size">Size of the linked cubin.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, ref SizeT size)
        {
            return __nvJitLinkGetLinkedCubinSize_12_6(handle, ref size);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetLinkedCubin_12_6(nvJitLinkHandle handle, byte[] cubin);

        /// <summary>
        /// nvJitLinkGetLinkedCubin gets the linked cubin.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="cubin">The linked cubin.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, byte[] cubin)
        {
            return __nvJitLinkGetLinkedCubin_12_6(handle, cubin);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetLinkedPtxSize_12_6(nvJitLinkHandle handle, ref SizeT size);

        /// <summary>
        /// nvJitLinkGetLinkedPtxSize gets the size of the linked ptx.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="size">Size of the linked PTX.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, ref SizeT size)
        {
            return __nvJitLinkGetLinkedPtxSize_12_6(handle, ref size);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetLinkedPtx_12_6(nvJitLinkHandle handle, byte[] ptx);

        /// <summary>
        /// nvJitLinkGetLinkedPtx gets the linked ptx.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="ptx">The linked PTX.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, byte[] ptx)
        {
            return __nvJitLinkGetLinkedPtx_12_6(handle, ptx);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetErrorLogSize_12_6(nvJitLinkHandle handle, ref SizeT size);

        /// <summary>
        /// nvJitLinkGetErrorLogSize gets the size of the error log.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="size">Size of the error log.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, ref SizeT size)
        {
            return __nvJitLinkGetErrorLogSize_12_6(handle, ref size);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetErrorLog_12_6(nvJitLinkHandle handle, byte[] log);

        /// <summary>
        /// nvJitLinkGetErrorLog puts any error messages in the log.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="log">The error log.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, byte[] log)
        {
            return __nvJitLinkGetErrorLog_12_6(handle, log);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetInfoLogSize_12_6(nvJitLinkHandle handle, ref SizeT size);

        /// <summary>
        /// nvJitLinkGetInfoLogSize gets the size of the info log.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="size">Size of the info log.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, ref SizeT size)
        {
            return __nvJitLinkGetInfoLogSize_12_6(handle, ref size);
        }

        [DllImport(NVJITLINK_API_DLL_NAME)]
        internal static extern nvJitLinkResult __nvJitLinkGetInfoLog_12_6(nvJitLinkHandle handle, byte[] log);

        /// <summary>
        /// nvJitLinkGetInfoLog puts any info messages in the log.
        /// </summary>
        /// <param name="handle">nvJitLink handle.</param>
        /// <param name="log">The info log.</param>
        /// <returns></returns>
        public static nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, byte[] log)
        {
            return __nvJitLinkGetInfoLog_12_6(handle, log);
        }



        /// <summary>
        /// nvJitLinkVersion returns the current version of nvJitLink.
        /// </summary>
        [DllImport(NVJITLINK_API_DLL_NAME)]
        public static extern nvJitLinkResult nvJitLinkVersion(ref uint major, ref uint minor);
    }
}
