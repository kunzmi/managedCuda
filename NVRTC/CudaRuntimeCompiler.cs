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
using System.Text;

namespace ManagedCuda.NVRTC
{
    /// <summary>
    /// Cuda runtime compiler
    /// </summary>
    public class CudaRuntimeCompiler : IDisposable
    {
        private nvrtcProgram _program;
        private bool disposed = false;
        private nvrtcResult res;

        #region Contructors
        /// <summary>
        /// Creates a runtime compiler instance.
        /// </summary>
		/// <param name="src">CUDA program source.</param>
		/// <param name="name">CUDA program name.<para/>
		/// name can be NULL; "default_program" is used when name is NULL.</param>
		/// <param name="includeNames">Sources of the headers.</param>
		/// <param name="headers">Name of each header by which they can be included in the CUDA program source.</param>
        public CudaRuntimeCompiler(string src, string name, string[] headers, string[] includeNames)
        {
            int headerCount = 0;
            IntPtr[] headersPtr = null;
            IntPtr[] includeNamesPtr = null;

            try
            {
                if (headers != null && includeNames != null)
                {
                    if (headers.Length != includeNames.Length)
                        throw new ArgumentException("headers and includeNames must have same length.");

                    if (headers == null)
                        throw new ArgumentNullException("headers can't be NULL if includeNames is not NULL");

                    if (includeNames == null)
                        throw new ArgumentNullException("includeNames can't be NULL if headers is not NULL");

                    headerCount = headers.Length;

                    headersPtr = new IntPtr[headerCount];
                    includeNamesPtr = new IntPtr[headerCount];

                    for (int i = 0; i < headerCount; i++)
                    {
                        headersPtr[i] = Marshal.StringToHGlobalAnsi(headers[i]);
                        includeNamesPtr[i] = Marshal.StringToHGlobalAnsi(includeNames[i]);
                    }
                }

                _program = new nvrtcProgram();
                res = NVRTCNativeMethods.nvrtcCreateProgram(ref _program, src, name, headerCount, headersPtr, includeNamesPtr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcCreateProgram", res));
            }
            finally
            {
                if (headersPtr != null)
                    for (int i = 0; i < headersPtr.Length; i++)
                    {
                        Marshal.FreeHGlobal(headersPtr[i]);
                    }

                if (includeNamesPtr != null)
                    for (int i = 0; i < includeNamesPtr.Length; i++)
                    {
                        Marshal.FreeHGlobal(includeNamesPtr[i]);
                    }
            }

            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);
        }
        /// <summary>
        /// Creates a runtime compiler instance.
        /// </summary>
		/// <param name="src">CUDA program source.</param>
		/// <param name="name">CUDA program name.<para/>
		/// name can be NULL; "default_program" is used when name is NULL.</param>
		public CudaRuntimeCompiler(string src, string name)
            : this(src, name, null, null)
        {

        }

        /// <summary>
        /// For dispose
        /// </summary>
		~CudaRuntimeCompiler()
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
                //Ignore if failing
                res = NVRTCNativeMethods.nvrtcDestroyProgram(ref _program);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcDestroyProgram", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Static Methods
        /// <summary/>
        public static Version GetVersion()
        {
            int major = 0;
            int minor = 0;
            nvrtcResult res = NVRTCNativeMethods.nvrtcVersion(ref major, ref minor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcVersion", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);
            return new Version(major, minor);
        }

        public static int[] GetSupportedArchs()
        {
            int num = 0;
            nvrtcResult res = NVRTCNativeMethods.nvrtcGetNumSupportedArchs(ref num);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetNumSupportedArchs", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            if (num <= 0)
            {
                return null;
            }

            int[] archs = new int[num];
            res = NVRTCNativeMethods.nvrtcGetSupportedArchs(archs);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetSupportedArchs", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return archs;
        }

        /// <summary>
        /// Retrieve the current size of the PCH Heap.
        /// </summary>
        public static SizeT GetPCHHeapSize()
        {
            SizeT size = 0;
            nvrtcResult res = NVRTCNativeMethods.nvrtcGetPCHHeapSize(ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetPCHHeapSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);
            return size;
        }

        /// <summary>
        /// Set the size of the PCH Heap.
        /// </summary>
        public static void SetPCHHeapSize(SizeT size)
        {
            nvrtcResult res = NVRTCNativeMethods.nvrtcSetPCHHeapSize(size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcSetPCHHeapSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);
        }
        #endregion

        #region Methods
        /// <summary/>
        public void Compile(string[] options)
        {
            int optionCount = 0;
            IntPtr[] optionsPtr = null;

            try
            {
                if (options != null)
                {
                    optionCount = options.Length;
                    optionsPtr = new IntPtr[optionCount];

                    for (int i = 0; i < optionCount; i++)
                    {
                        optionsPtr[i] = Marshal.StringToHGlobalAnsi(options[i]);
                    }
                }

                res = NVRTCNativeMethods.nvrtcCompileProgram(_program, optionCount, optionsPtr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcCompileProgram", res));
            }
            finally
            {
                if (optionsPtr != null)
                    for (int i = 0; i < optionsPtr.Length; i++)
                    {
                        Marshal.FreeHGlobal(optionsPtr[i]);
                    }
            }
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);
        }

        /// <summary/>
        public byte[] GetPTX()
        {
            SizeT ptxSize = new SizeT();

            res = NVRTCNativeMethods.nvrtcGetPTXSize(_program, ref ptxSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetPTXSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            byte[] ptxCode = new byte[ptxSize];

            res = NVRTCNativeMethods.nvrtcGetPTX(_program, ptxCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetPTX", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return ptxCode;
        }

        /// <summary/>
        public byte[] GetCubin()
        {
            SizeT cubinSize = new SizeT();

            res = NVRTCNativeMethods.nvrtcGetCUBINSize(_program, ref cubinSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetCUBINSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            byte[] cubinCode = new byte[cubinSize];

            res = NVRTCNativeMethods.nvrtcGetCUBIN(_program, cubinCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetCUBIN", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return cubinCode;
        }

        /// <summary/>
        [Obsolete("This function will be removed in a future release. Please use GetLTOIR instead.")]
        public byte[] GetNVVM()
        {
            SizeT nvvmSize = new SizeT();

            res = NVRTCNativeMethods.nvrtcGetNVVMSize(_program, ref nvvmSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetNVVMSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            byte[] nvvmCode = new byte[nvvmSize];

            res = NVRTCNativeMethods.nvrtcGetNVVM(_program, nvvmCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetNVVM", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return nvvmCode;
        }

        /// <summary/>
        public byte[] GetLTOIR()
        {
            SizeT ltoirSize = new SizeT();

            res = NVRTCNativeMethods.nvrtcGetLTOIRSize(_program, ref ltoirSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetLTOIRSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            byte[] ltoirCode = new byte[ltoirSize];

            res = NVRTCNativeMethods.nvrtcGetLTOIR(_program, ltoirCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetLTOIR", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return ltoirCode;
        }

        /// <summary/>
        public byte[] GetOptiXIR()
        {
            SizeT optiXIRSize = new SizeT();

            res = NVRTCNativeMethods.nvrtcGetOptiXIRSize(_program, ref optiXIRSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetOptiXIRSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            byte[] optiXIRCode = new byte[optiXIRSize];

            res = NVRTCNativeMethods.nvrtcGetOptiXIR(_program, optiXIRCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetOptiXIR", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return optiXIRCode;
        }

        /// <summary/>
        public string GetPTXAsString()
        {
            byte[] ptxCode = GetPTX();
            ASCIIEncoding enc = new ASCIIEncoding();

            string ptxString = enc.GetString(ptxCode);
            return ptxString.Replace("\0", "");
        }

        /// <summary/>
        public byte[] GetLog()
        {
            SizeT logSize = new SizeT();

            res = NVRTCNativeMethods.nvrtcGetProgramLogSize(_program, ref logSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetProgramLogSize", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            byte[] logCode = new byte[logSize];

            res = NVRTCNativeMethods.nvrtcGetProgramLog(_program, logCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetProgramLog", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return logCode;
        }

        /// <summary/>
        public string GetLogAsString()
        {
            byte[] logCode = GetLog();
            ASCIIEncoding enc = new ASCIIEncoding();

            string logString = enc.GetString(logCode);
            return logString.Replace("\0", "");
        }

        /// <summary/>
        public void AddNameExpression(string nameExpression)
        {

            res = NVRTCNativeMethods.nvrtcAddNameExpression(_program, nameExpression);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcAddNameExpression", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);
        }

        /// <summary/>
        public string GetLoweredName(string nameExpression)
        {
            IntPtr ret = new IntPtr();
            res = NVRTCNativeMethods.nvrtcGetLoweredName(_program, nameExpression, ref ret);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetLoweredName", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            //ret ptr is freed when _program is destroyed!
            return Marshal.PtrToStringAnsi(ret);
        }

        /// <summary/>
        public nvrtcResult GetPCHCreateStatus()
        {
            res = NVRTCNativeMethods.nvrtcGetPCHCreateStatus(_program);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetPCHCreateStatus", res));

            return res;
        }

        /// <summary/>
        public SizeT GetPCHHeapSizeRequired()
        {
            SizeT size = new SizeT();
            res = NVRTCNativeMethods.nvrtcGetPCHHeapSizeRequired(_program, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetPCHHeapSizeRequired", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);

            return size;
        }

        /// <summary/>
        public void SetFlowCallback(setFlowCallback callback, IntPtr payload)
        {
            res = NVRTCNativeMethods.nvrtcSetFlowCallback(_program, callback, payload);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcSetFlowCallback", res));
            if (res != nvrtcResult.Success)
                throw new NVRTCException(res);
        }
        #endregion
    }
}
