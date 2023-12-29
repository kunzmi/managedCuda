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
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NvJpeg
{

    /// <summary>
    /// Wrapper class for nvjpegBufferPinned
    /// </summary>
    public class BufferPinned : IDisposable
    {
        private nvjpegBufferPinned _buffer;
        private NvJpeg _nvJpeg;
        private nvjpegStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        internal BufferPinned(NvJpeg nvJpeg, nvjpegPinnedAllocator pinnedAllocator)
        {
            _nvJpeg = nvJpeg;
            _buffer = new nvjpegBufferPinned();
            res = NvJpegNativeMethods.nvjpegBufferPinnedCreate(nvJpeg.Handle, ref pinnedAllocator, ref _buffer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferPinnedCreate", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }
        /// <summary>
        /// </summary>
        internal BufferPinned(NvJpeg nvJpeg, nvjpegPinnedAllocatorV2 pinnedAllocator)
        {
            _nvJpeg = nvJpeg;
            _buffer = new nvjpegBufferPinned();
            res = NvJpegNativeMethods.nvjpegBufferPinnedCreateV2(nvJpeg.Handle, ref pinnedAllocator, ref _buffer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferPinnedCreateV2", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~BufferPinned()
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
                res = NvJpegNativeMethods.nvjpegBufferPinnedDestroy(_buffer);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferPinnedDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        public void Resize(SizeT size, CUstream stream)
        {
            res = NvJpegNativeMethods.nvjpegBufferPinnedResize(_buffer, size, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferPinnedResize", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }

        public void Resize(SizeT size, CudaStream stream)
        {
            Resize(size, stream.Stream);
        }

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public nvjpegBufferPinned Buffer
        {
            get { return _buffer; }
        }

        public SizeT Size
        {
            get
            {
                SizeT value = 0;
                IntPtr dummy = new IntPtr();
                res = NvJpegNativeMethods.nvjpegBufferPinnedRetrieve(_buffer, ref value, ref dummy);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferPinnedRetrieve", res));
                if (res != nvjpegStatus.Success)
                    throw new NvJpegException(res);
                return value;
            }
        }

        public IntPtr Ptr
        {
            get
            {
                SizeT dummy = 0;
                IntPtr value = new IntPtr();
                res = NvJpegNativeMethods.nvjpegBufferPinnedRetrieve(_buffer, ref dummy, ref value);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegBufferPinnedRetrieve", res));
                if (res != nvjpegStatus.Success)
                    throw new NvJpegException(res);
                return value;
            }
        }
    }
}
