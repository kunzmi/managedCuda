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

namespace ManagedCuda.CudaSparse
{
    /// <summary>
    /// 
    /// </summary>
    public class ConstDenseVector<dataT> : IDisposable where dataT : struct
    {
        private cusparseStatus res;
        private bool disposed;
        private cusparseConstDnVecDescr descr;
        private cudaDataType typeData;
        private long size;
        private bool noCleanup = false;



        #region Contructors
        /// <summary>
        /// </summary>
        public ConstDenseVector(long aSize, CudaDeviceVariable<dataT> values)
        {
            size = aSize;
            descr = new cusparseConstDnVecDescr();
            typeData = CudaDataTypeTranslator.GetType(typeof(dataT));
            res = CudaSparseNativeMethods.cusparseCreateConstDnVec(ref descr, size, values.DevicePointer, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstDnVec", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// </summary>
        internal ConstDenseVector(DenseVector<dataT> denseVector)
        {
            size = denseVector.Size;
            descr = denseVector.Descr;
            typeData = CudaDataTypeTranslator.GetType(typeof(dataT));
            noCleanup = true;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~ConstDenseVector()
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
            if (!noCleanup)
            {
                if (fDisposing && !disposed)
                {
                    //Ignore if failing
                    res = CudaSparseNativeMethods.cusparseDestroyDnVec(descr);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroyDnVec", res));
                    disposed = true;
                }
                if (!fDisposing && !disposed)
                    Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
            }
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cusparseConstDnVecDescr Descr
        {
            get { return descr; }
        }
        /// <summary>
        /// 
        /// </summary>
        public cudaDataType TypeData
        {
            get { return typeData; }
        }
        /// <summary>
        /// 
        /// </summary>
        public long Size
        {
            get { return size; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CudaDeviceVariable<dataT> GetValues()
        {
            CUdeviceptr devPtr = new CUdeviceptr();
            res = CudaSparseNativeMethods.cusparseConstDnVecGetValues(descr, ref devPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstDnVecGetValues", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return new CudaDeviceVariable<dataT>(devPtr);
        }

        /// <summary>
        /// 
        /// </summary>
        public CudaDeviceVariable<dataT> Get()
        {
            CUdeviceptr ptrValues = new CUdeviceptr();

            res = CudaSparseNativeMethods.cusparseConstDnVecGet(descr, ref size, ref ptrValues, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstDnVecGet", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new CudaDeviceVariable<dataT>(ptrValues);
        }
    }
    /// <summary>
    /// 
    /// </summary>
    public class DenseVector<dataT> : IDisposable where dataT : struct
    {
        private cusparseStatus res;
        private bool disposed;
        private cusparseDnVecDescr descr;
        private cudaDataType typeData;
        private long size;



        #region Contructors
        /// <summary>
        /// </summary>
        public DenseVector(long aSize, CudaDeviceVariable<dataT> values)
        {
            size = aSize;
            descr = new cusparseDnVecDescr();
            typeData = CudaDataTypeTranslator.GetType(typeof(dataT));
            res = CudaSparseNativeMethods.cusparseCreateDnVec(ref descr, size, values.DevicePointer, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateDnVec", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DenseVector()
        {
            Dispose(false);
        }
        #endregion

        /// <summary>
        ///
        /// </summary>
        public static implicit operator ConstDenseVector<dataT>(DenseVector<dataT> denseVector)
        {
            return new ConstDenseVector<dataT>(denseVector);
        }

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
                res = CudaSparseNativeMethods.cusparseDestroyDnVec(descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroyDnVec", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cusparseDnVecDescr Descr
        {
            get { return descr; }
        }
        /// <summary>
        /// 
        /// </summary>
        public cudaDataType TypeData
        {
            get { return typeData; }
        }
        /// <summary>
        /// 
        /// </summary>
        public long Size
        {
            get { return size; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CudaDeviceVariable<dataT> GetValues()
        {
            CUdeviceptr devPtr = new CUdeviceptr();
            res = CudaSparseNativeMethods.cusparseDnVecGetValues(descr, ref devPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnVecGetValues", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return new CudaDeviceVariable<dataT>(devPtr);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data"></param>
        public void SetValues(CudaDeviceVariable<dataT> data)
        {
            res = CudaSparseNativeMethods.cusparseDnVecSetValues(descr, data.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnVecSetValues", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        public CudaDeviceVariable<dataT> Get()
        {
            CUdeviceptr ptrValues = new CUdeviceptr();

            res = CudaSparseNativeMethods.cusparseDnVecGet(descr, ref size, ref ptrValues, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnVecGet", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new CudaDeviceVariable<dataT>(ptrValues);
        }
    }
}
