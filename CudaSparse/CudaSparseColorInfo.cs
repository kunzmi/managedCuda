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

namespace ManagedCuda.CudaSparse
{
    /// <summary>
    /// Wrapper class for cusparseColorInfo
    /// </summary>
    public class CudaSparseColorInfo : IDisposable
    {
        private cusparseColorInfo _info;
        private cusparseStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        public CudaSparseColorInfo()
        {
            _info = new cusparseColorInfo();
            res = CudaSparseNativeMethods.cusparseCreateColorInfo(ref _info);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateColorInfo", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaSparseColorInfo()
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
                res = CudaSparseNativeMethods.cusparseDestroyColorInfo(_info);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroyColorInfo", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cusparseColorInfo ColorInfo
        {
            get { return _info; }
        }

        #region Methods
        ///// <summary>
        ///// SetColorAlgs
        ///// </summary>
        //public void SetColorAlgs(cusparseColorAlg alg)
        //{
        //    res = CudaSparseNativeMethods.cusparseSetColorAlgs(_info, alg);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSetColorAlgs", res));
        //    if (res != cusparseStatus.Success)
        //        throw new CudaSparseException(res);
        //}
        ///// <summary>
        ///// GetColorAlgs
        ///// </summary>
        //public cusparseColorAlg GetColorAlgs()
        //{
        //    cusparseColorAlg retVal = new cusparseColorAlg();
        //    res = CudaSparseNativeMethods.cusparseGetColorAlgs(_info, ref retVal);
        //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetColorAlgs", res));
        //    if (res != cusparseStatus.Success)
        //        throw new CudaSparseException(res);
        //    return retVal;
        //}
        #endregion
    }
}
