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
using System.Text;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NvJitLink
{
    /// <summary>
    /// Cuda Jit Link
    /// </summary>
    public class NvJitLink : IDisposable
    {
        nvJitLinkHandle _handle;
        bool disposed = false;
        nvJitLinkResult res;
        #region Contructors
        /// <summary>
        /// Creates a Cuda Jit Link instance.
        /// </summary>
        /// <param name="options">Array of size \p numOptions of option strings.</param>
        public NvJitLink(string[] options)
        {
            _handle = new nvJitLinkHandle();
            res = NvJitLinkNativeMethods.nvJitLinkCreate(ref _handle, options);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkCreate", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
		~NvJitLink()
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
                res = NvJitLinkNativeMethods.nvJitLinkDestroy(ref _handle);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods

        /// <summary>
        /// nvJitLinkAddData adds data image to the link. 
        /// </summary>
        /// <param name="inputType">kind of input.</param>
        /// <param name="data">pointer to data image in memory.</param>
        /// <param name="name">name of input object.</param>
        public void AddData(nvJitLinkInputType inputType, byte[] data, string name)
        {
            res = NvJitLinkNativeMethods.nvJitLinkAddData(_handle, inputType, data, name);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkAddData", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);
        }

        /// <summary>
        /// nvJitLinkAddData adds data image to the link. 
        /// </summary>
        /// <param name="inputType">kind of input.</param>
        /// <param name="data">pointer to data image in memory.</param>
        /// <param name="size"></param>
        /// <param name="name">name of input object.</param>
        public void AddData(nvJitLinkInputType inputType, IntPtr data, SizeT size, string name)
        {
            res = NvJitLinkNativeMethods.nvJitLinkAddData(_handle, inputType, data, size, name);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkAddData", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);
        }

        /// <summary>
        /// nvJitLinkAddFile reads data from file and links it in. 
        /// </summary>
        /// <param name="inputType">kind of input.</param>
        /// <param name="fileName">name of file.</param>
        public void AddFile(nvJitLinkInputType inputType, string fileName)
        {
            res = NvJitLinkNativeMethods.nvJitLinkAddFile(_handle, inputType, fileName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkAddFile", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);
        }

        /// <summary>
        /// nvJitLinkComplete does the actual link.
        /// </summary>
        public void Complete()
        {
            res = NvJitLinkNativeMethods.nvJitLinkComplete(_handle);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkComplete", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);
        }

        /// <summary>
        /// nvJitLinkGetLinkedCubin gets the linked cubin.
        /// </summary>
        public byte[] GetLinkedCubin()
        {
            SizeT cubinSize = new SizeT();

            res = NvJitLinkNativeMethods.nvJitLinkGetLinkedCubinSize(_handle, ref cubinSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetLinkedCubinSize", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            byte[] cubinCode = new byte[cubinSize];

            res = NvJitLinkNativeMethods.nvJitLinkGetLinkedCubin(_handle, cubinCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetLinkedCubin", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            return cubinCode;
        }

        /// <summary>
        /// nvJitLinkGetLinkedPtx gets the linked ptx.
        /// </summary>
        public byte[] GetLinkedPtx()
        {
            SizeT ptxSize = new SizeT();

            res = NvJitLinkNativeMethods.nvJitLinkGetLinkedPtxSize(_handle, ref ptxSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetLinkedPtxSize", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            byte[] ptxCode = new byte[ptxSize];

            res = NvJitLinkNativeMethods.nvJitLinkGetLinkedPtx(_handle, ptxCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetLinkedPtx", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            return ptxCode;
        }

        /// <summary>
        /// nvJitLinkGetLinkedPtx gets the linked ptx.
        /// </summary>
        public string GetLinkedPtxAsString()
        {
            byte[] ptxCode = GetLinkedPtx();

            ASCIIEncoding enc = new ASCIIEncoding();

            string ptxString = enc.GetString(ptxCode);
            return ptxString.Replace("\0", "");
        }

        /// <summary>
        /// nvJitLinkGetErrorLog puts any error messages in the log.
        /// </summary>
        public byte[] GetErrorLog()
        {
            SizeT logSize = new SizeT();

            res = NvJitLinkNativeMethods.nvJitLinkGetErrorLogSize(_handle, ref logSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetErrorLogSize", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            byte[] logCode = new byte[logSize];

            res = NvJitLinkNativeMethods.nvJitLinkGetErrorLog(_handle, logCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetErrorLog", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            return logCode;
        }

        /// <summary>
        /// nvJitLinkGetErrorLog puts any error messages in the log.
        /// </summary>
        public string GetErrorLogAsString()
        {
            byte[] logCode = GetErrorLog();
            ASCIIEncoding enc = new ASCIIEncoding();

            string logString = enc.GetString(logCode);
            return logString.Replace("\0", "");
        }

        /// <summary>
        /// nvJitLinkGetInfoLog puts any info messages in the log.
        /// </summary>
        public byte[] GetInfoLog()
        {
            SizeT logSize = new SizeT();

            res = NvJitLinkNativeMethods.nvJitLinkGetInfoLogSize(_handle, ref logSize);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetInfoLogSize", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            byte[] logCode = new byte[logSize];

            res = NvJitLinkNativeMethods.nvJitLinkGetInfoLog(_handle, logCode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvJitLinkGetInfoLog", res));
            if (res != nvJitLinkResult.Success)
                throw new NvJitLinkException(res);

            return logCode;
        }

        /// <summary>
        /// nvJitLinkGetInfoLog puts any info messages in the log.
        /// </summary>
        public string GetInfoLogAsString()
        {
            byte[] logCode = GetInfoLog();
            ASCIIEncoding enc = new ASCIIEncoding();

            string logString = enc.GetString(logCode);
            return logString.Replace("\0", "");
        }

        #endregion
    }
}
