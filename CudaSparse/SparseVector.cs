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
    public class ConstSparseVector<indexT, dataT> : IDisposable where indexT : struct where dataT : struct
    {
        private cusparseStatus res;
        private bool disposed;
        private cusparseConstSpVecDescr descr;
        private IndexType typeIndices;
        private cudaDataType typeData;
        private long size;
        private long nnz;
        private IndexBase idxBase;
        private bool noCleanup = false;



        #region Contructors
        /// <summary>
        /// </summary>
        public ConstSparseVector(long aSize, long aNnz, CudaDeviceVariable<indexT> indices, CudaDeviceVariable<dataT> values, IndexBase aIdxBase)
        {
            size = aSize;
            nnz = aNnz;
            idxBase = aIdxBase;
            descr = new cusparseConstSpVecDescr();
            typeIndices = IndexTypeTranslator.GetType(typeof(indexT));
            typeData = CudaDataTypeTranslator.GetType(typeof(dataT));
            res = CudaSparseNativeMethods.cusparseCreateConstSpVec(ref descr, size, nnz, indices.DevicePointer, values.DevicePointer, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstSpVec", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// </summary>
        internal ConstSparseVector(SparseVector<indexT, dataT> sparseVector)
        {
            size = sparseVector.Size;
            nnz = sparseVector.Nnz;
            idxBase = sparseVector.IdxBase;
            descr = sparseVector.Descr;
            typeIndices = IndexTypeTranslator.GetType(typeof(indexT));
            typeData = CudaDataTypeTranslator.GetType(typeof(dataT));
            noCleanup = true;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~ConstSparseVector()
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
                    res = CudaSparseNativeMethods.cusparseDestroySpVec(descr);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroySpVec", res));
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
        public cusparseConstSpVecDescr Descr
        {
            get { return descr; }
        }
        /// <summary>
        /// 
        /// </summary>
        public IndexType TypeIndices
        {
            get { return typeIndices; }
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
        public long Nnz
        {
            get { return nnz; }
        }
        /// <summary>
        /// 
        /// </summary>
        public IndexBase IdxBase
        {
            get
            {
                res = CudaSparseNativeMethods.cusparseSpVecGetIndexBase(descr, ref idxBase);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVecGetIndexBase", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
                return idxBase;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CudaDeviceVariable<dataT> GetValues()
        {
            CUdeviceptr devPtr = new CUdeviceptr();
            res = CudaSparseNativeMethods.cusparseConstSpVecGetValues(descr, ref devPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstSpVecGetValues", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return new CudaDeviceVariable<dataT>(devPtr);
        }

        /// <summary>
        /// 
        /// </summary>
        public void Get(out CudaDeviceVariable<indexT> indices, out CudaDeviceVariable<dataT> values)
        {
            CUdeviceptr ptrIndices = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();

            res = CudaSparseNativeMethods.cusparseConstSpVecGet(descr, ref size, ref nnz, ref ptrIndices, ref ptrValues, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstSpVecGet", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            indices = new CudaDeviceVariable<indexT>(ptrIndices);
            values = new CudaDeviceVariable<dataT>(ptrValues);
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class SparseVector<indexT, dataT> : IDisposable where indexT : struct where dataT : struct
    {
        private cusparseStatus res;
        private bool disposed;
        private cusparseSpVecDescr descr;
        private IndexType typeIndices;
        private cudaDataType typeData;
        private long size;
        private long nnz;
        private IndexBase idxBase;



        #region Contructors
        /// <summary>
        /// </summary>
        public SparseVector(long aSize, long aNnz, CudaDeviceVariable<indexT> indices, CudaDeviceVariable<dataT> values, IndexBase aIdxBase)
        {
            size = aSize;
            nnz = aNnz;
            idxBase = aIdxBase;
            descr = new cusparseSpVecDescr();
            typeIndices = IndexTypeTranslator.GetType(typeof(indexT));
            typeData = CudaDataTypeTranslator.GetType(typeof(dataT));
            res = CudaSparseNativeMethods.cusparseCreateSpVec(ref descr, size, nnz, indices.DevicePointer, values.DevicePointer, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateSpVec", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~SparseVector()
        {
            Dispose(false);
        }
        #endregion

        /// <summary>
        ///
        /// </summary>
        public static implicit operator ConstSparseVector<indexT, dataT>(SparseVector<indexT, dataT> sparseVector)
        {
            return new ConstSparseVector<indexT, dataT>(sparseVector);
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
                res = CudaSparseNativeMethods.cusparseDestroySpVec(descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroySpVec", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cusparseSpVecDescr Descr
        {
            get { return descr; }
        }
        /// <summary>
        /// 
        /// </summary>
        public IndexType TypeIndices
        {
            get { return typeIndices; }
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
        public long Nnz
        {
            get { return nnz; }
        }
        /// <summary>
        /// 
        /// </summary>
        public IndexBase IdxBase
        {
            get
            {
                res = CudaSparseNativeMethods.cusparseSpVecGetIndexBase(descr, ref idxBase);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVecGetIndexBase", res));
                if (res != cusparseStatus.Success)
                    throw new CudaSparseException(res);
                return idxBase;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CudaDeviceVariable<dataT> GetValues()
        {
            CUdeviceptr devPtr = new CUdeviceptr();
            res = CudaSparseNativeMethods.cusparseSpVecGetValues(descr, ref devPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVecGetValues", res));
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
            res = CudaSparseNativeMethods.cusparseSpVecSetValues(descr, data.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVecSetValues", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        public void Get(out CudaDeviceVariable<indexT> indices, out CudaDeviceVariable<dataT> values)
        {
            CUdeviceptr ptrIndices = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();

            res = CudaSparseNativeMethods.cusparseSpVecGet(descr, ref size, ref nnz, ref ptrIndices, ref ptrValues, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpVecGet", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            indices = new CudaDeviceVariable<indexT>(ptrIndices);
            values = new CudaDeviceVariable<dataT>(ptrValues);
        }
    }
}
