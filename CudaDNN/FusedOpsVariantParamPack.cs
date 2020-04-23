//	Copyright (c) 2015, Michael Kunz. All rights reserved.
//	http://kunzmi.github.io/managedCuda
//
//	This file is part of ManagedCuda.
//
//	ManagedCuda is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as 
//	published by the Free Software Foundation, either version 2.1 of the 
//	License, or (at your option) any later version.
//
//	ManagedCuda is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.


using System;
using System.Text;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaDNN
{
    /// <summary>
    /// </summary>
    public class FusedOpsVariantParamPack : IDisposable
    {
        private cudnnFusedOpsVariantParamPack _pack;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        public FusedOpsVariantParamPack(cudnnFusedOps ops)
        {
            _pack = new cudnnFusedOpsVariantParamPack();
            res = CudaDNNNativeMethods.cudnnCreateFusedOpsVariantParamPack(ref _pack, ops);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateFusedOpsVariantParamPack", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~FusedOpsVariantParamPack()
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
                res = CudaDNNNativeMethods.cudnnDestroyFusedOpsVariantParamPack(_pack);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyFusedOpsVariantParamPack", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnFusedOpsVariantParamPack Pack
        {
            get { return _pack; }
        }

        public void SetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamLabel paramLabel, IntPtr param)
        {
            res = CudaDNNNativeMethods.cudnnSetFusedOpsVariantParamPackAttribute(_pack, paramLabel, param);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetFusedOpsVariantParamPackAttribute", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        public void SetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamLabel paramLabel, CUdeviceptr param)
        {
            res = CudaDNNNativeMethods.cudnnSetFusedOpsVariantParamPackAttribute(_pack, paramLabel, param);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetFusedOpsVariantParamPackAttribute", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        public void GetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamLabel paramLabel, IntPtr param)
        {
            res = CudaDNNNativeMethods.cudnnGetFusedOpsVariantParamPackAttribute(_pack, paramLabel, param);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetFusedOpsVariantParamPackAttribute", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        public void GetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamLabel paramLabel, CUdeviceptr param)
        {
            res = CudaDNNNativeMethods.cudnnGetFusedOpsVariantParamPackAttribute(_pack, paramLabel, param);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetFusedOpsVariantParamPackAttribute", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
