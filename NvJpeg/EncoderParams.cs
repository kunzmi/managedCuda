﻿// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
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
    /// Wrapper class for nvjpegEncoderParams
    /// </summary>
    public class EncoderParams : IDisposable
    {
        private nvjpegEncoderParams _params;
        private NvJpeg _nvJpeg;
        private nvjpegStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        internal EncoderParams(NvJpeg nvJpeg, CudaStream stream)
        {
            _nvJpeg = nvJpeg;
            _params = new nvjpegEncoderParams();
            res = NvJpegNativeMethods.nvjpegEncoderParamsCreate(nvJpeg.Handle, ref _params, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsCreate", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~EncoderParams()
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
                res = NvJpegNativeMethods.nvjpegEncoderParamsDestroy(_params);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public nvjpegEncoderParams Params
        {
            get { return _params; }
        }


        public void SetQuality(int quality, CudaStream stream)
        {
            res = NvJpegNativeMethods.nvjpegEncoderParamsSetQuality(_params, quality, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsSetQuality", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }

        public void SetEncoding(nvjpegJpegEncoding etype, CudaStream stream)
        {
            res = NvJpegNativeMethods.nvjpegEncoderParamsSetEncoding(_params, etype, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsSetEncoding", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }

        public void SetOptimizedHuffman(int optimized, CudaStream stream)
        {
            res = NvJpegNativeMethods.nvjpegEncoderParamsSetOptimizedHuffman(_params, optimized, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsSetOptimizedHuffman", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }

        public void SetSamplingFactors(nvjpegChromaSubsampling chroma_subsampling, CudaStream stream)
        {
            res = NvJpegNativeMethods.nvjpegEncoderParamsSetSamplingFactors(_params, chroma_subsampling, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsSetSamplingFactors", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }

        public SizeT GetBufferSize(int image_width, int image_height)
        {
            SizeT max_stream_length = new SizeT();
            res = NvJpegNativeMethods.nvjpegEncodeGetBufferSize(_nvJpeg.Handle, _params, image_width, image_height, ref max_stream_length);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncodeGetBufferSize", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
            return max_stream_length;
        }

        // copies quantization tables from parsed stream
        public void CopyQuantizationTables(JpegStream jpeg, CudaStream stream)
        {
            res = NvJpegNativeMethods.nvjpegEncoderParamsCopyQuantizationTables(_params, jpeg.Stream, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvjpegEncoderParamsCopyQuantizationTables", res));
            if (res != nvjpegStatus.Success)
                throw new NvJpegException(res);
        }


    }
}
