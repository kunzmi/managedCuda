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
using ManagedCuda.BasicTypes;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Represents an executable Cuda graph.
    /// </summary>
    public class CudaLibrary : IDisposable
    {
        private bool disposed;
        private CUResult res;
        private CUlibrary _library;

        #region Constructor

        /// <summary>
        /// Load library
        /// </summary>
        public CudaLibrary(string filename, CudaJitOptionCollection jitOptions, CudaLibraryOptionCollection libOptions)
        {
            _library = new CUlibrary();

            uint jitCount = 0;
            CUJITOption[] jitOpt = null;
            IntPtr[] jitOptVal = null;
            uint libCount = 0;
            CUlibraryOption[] libOpt = null;
            IntPtr[] libOptVal = null;

            if (jitOptions != null)
            {
                jitCount = (uint)jitOptions.Count;
                jitOpt = jitOptions.Options;
                jitOptVal = jitOptions.Values;
            }
            if (libOptions != null)
            {
                libCount = (uint)libOptions.Count;
                libOpt = libOptions.Options;
                libOptVal = libOptions.Values;
            }

            res = DriverAPINativeMethods.LibraryManagement.cuLibraryLoadFromFile(ref _library, filename, jitOpt, jitOptVal, jitCount, libOpt, libOptVal, libCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryLoadFromFile", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Load library
        /// </summary>
        public CudaLibrary(System.IO.Stream image, CudaJitOptionCollection jitOptions, CudaLibraryOptionCollection libOptions)
        {
            _library = new CUlibrary();

            uint jitCount = 0;
            CUJITOption[] jitOpt = null;
            IntPtr[] jitOptVal = null;
            uint libCount = 0;
            CUlibraryOption[] libOpt = null;
            IntPtr[] libOptVal = null;

            if (jitOptions != null)
            {
                jitCount = (uint)jitOptions.Count;
                jitOpt = jitOptions.Options;
                jitOptVal = jitOptions.Values;
            }
            if (libOptions != null)
            {
                libCount = (uint)libOptions.Count;
                libOpt = libOptions.Options;
                libOptVal = libOptions.Values;
            }

            if (image == null) throw new ArgumentNullException("image");
            byte[] libImg = new byte[image.Length + 1]; //append a zero byte to be on the safe side...
            libImg[libImg.Length - 1] = 0; //should already be set by .net, but now we are sure...

            int bytesToRead = (int)image.Length;
            image.Position = 0;
            while (bytesToRead > 0)
            {
                bytesToRead -= image.Read(libImg, (int)image.Position, bytesToRead);
            }
            image.Position = 0;

            res = DriverAPINativeMethods.LibraryManagement.cuLibraryLoadData(ref _library, libImg, jitOpt, jitOptVal, jitCount, libOpt, libOptVal, libCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryLoadData", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Load library
        /// </summary>
        public CudaLibrary(byte[] image, CudaJitOptionCollection jitOptions, CudaLibraryOptionCollection libOptions)
        {
            image = AddZeroToArray(image);

            _library = new CUlibrary();

            uint jitCount = 0;
            CUJITOption[] jitOpt = null;
            IntPtr[] jitOptVal = null;
            uint libCount = 0;
            CUlibraryOption[] libOpt = null;
            IntPtr[] libOptVal = null;

            if (jitOptions != null)
            {
                jitCount = (uint)jitOptions.Count;
                jitOpt = jitOptions.Options;
                jitOptVal = jitOptions.Values;
            }
            if (libOptions != null)
            {
                libCount = (uint)libOptions.Count;
                libOpt = libOptions.Options;
                libOptVal = libOptions.Values;
            }

            res = DriverAPINativeMethods.LibraryManagement.cuLibraryLoadData(ref _library, image, jitOpt, jitOptVal, jitCount, libOpt, libOptVal, libCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryLoadData", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Load library
        /// </summary>
        public CudaLibrary(IntPtr image, CudaJitOptionCollection jitOptions, CudaLibraryOptionCollection libOptions)
        {
            _library = new CUlibrary();

            uint jitCount = 0;
            CUJITOption[] jitOpt = null;
            IntPtr[] jitOptVal = null;
            uint libCount = 0;
            CUlibraryOption[] libOpt = null;
            IntPtr[] libOptVal = null;

            if (jitOptions != null)
            {
                jitCount = (uint)jitOptions.Count;
                jitOpt = jitOptions.Options;
                jitOptVal = jitOptions.Values;
            }
            if (libOptions != null)
            {
                libCount = (uint)libOptions.Count;
                libOpt = libOptions.Options;
                libOptVal = libOptions.Values;
            }

            res = DriverAPINativeMethods.LibraryManagement.cuLibraryLoadData(ref _library, image, jitOpt, jitOptVal, jitCount, libOpt, libOptVal, libCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryLoadData", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaLibrary()
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
                res = DriverAPINativeMethods.LibraryManagement.cuLibraryUnload(_library);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryUnload", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods
        /// <summary>
        /// Make sure the library image arrays are zero terminated by appending a zero
        /// </summary>
        protected byte[] AddZeroToArray(byte[] image)
        {
            byte[] retArr = new byte[image.LongLength + 1];
            Array.Copy(image, retArr, image.LongLength);
            retArr[image.LongLength] = 0;
            return retArr;
        }


        /// <summary>
        /// Returns a module handle<para/>
        /// Returns in \p pMod the module handle associated with the current context located in library \p library.<para/>
        /// If module handle is not found, the call returns::CUDA_ERROR_NOT_FOUND.
        /// </summary>
        public CUmodule GetModule()
        {
            CUmodule ret = new CUmodule();
            res = DriverAPINativeMethods.LibraryManagement.cuLibraryGetModule(ref ret, _library);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryGetModule", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return ret;
        }

        /// <summary>
        /// Returns a kernel handle<para/>
        /// Returns in \p pKernel the handle of the kernel with name \p name located in library \p library.<para/>
        /// If kernel handle is not found, the call returns::CUDA_ERROR_NOT_FOUND.
        /// </summary>
        public CUkernel GetCUKernel(string kernelName)
        {
            CUkernel ret = new CUkernel();
            res = DriverAPINativeMethods.LibraryManagement.cuLibraryGetKernel(ref ret, _library, kernelName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryGetKernel", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return ret;
        }

        /// <summary>
        /// Returns a global device pointer<para/>
        /// Returns in \p *dptr and \p *bytes the base pointer and size of the global with
        /// name \p name for the requested library \p library and the current context.
        /// If no global for the requested name \p name exists, the call returns::CUDA_ERROR_NOT_FOUND.
        /// One of the parameters \p dptr or \p bytes (not both) can be NULL in which case it is ignored.
        /// </summary>
        /// <param name="name">Name of global to retrieve</param>
        /// <returns>CudaDeviceVariable</returns>
        public CudaDeviceVariable<T> GetGlobal<T>(string name) where T : struct
        {
            CUdeviceptr dptr = new CUdeviceptr();
            SizeT size = new SizeT();

            res = DriverAPINativeMethods.LibraryManagement.cuLibraryGetGlobal(ref dptr, ref size, _library, name);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLibraryGetGlobal", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return new CudaDeviceVariable<T>(dptr, false, size);
        }

        /// <summary>
        /// Returns a CudaKernel (the managedCuda wrapper for CUfunction, not to be confused with CUkernel)
        /// </summary>
        public CudaKernel GetCudaKernel(string kernelName)
        {
            CUkernel kernel = GetCUKernel(kernelName);

            return new CudaKernel(kernel, kernelName);
        }
        #endregion

        #region Properties

        /// <summary>
        /// Returns the inner library handle
        /// </summary>
        public CUlibrary Library
        {
            get { return _library; }
        }
        #endregion
    }
}
