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

namespace ManagedCuda.CudaSparse
{
    /// <summary>
    ///
    /// </summary>
    public class ConstSparseMatrix<indexT, dataT> : IDisposable where indexT : struct where dataT : struct
    {
        private cusparseStatus res;
        private bool disposed;
        private cusparseConstSpMatDescr descr;
        private IndexType typeIndices;
        private cudaDataType typeData;
        private long rows;
        private long cols;
        private long nnz;
        private IndexBase idxBase;
        private Format format;
        private bool noCleanup = false;



        #region Contructors
        /// <summary>
        /// </summary>
        private ConstSparseMatrix(cusparseConstSpMatDescr aDescr, long aRows, long aCols, long aNnz, IndexBase aIdxBase, IndexType aTypeIndices, cudaDataType aTypeData)
        {
            rows = aRows;
            cols = aCols;
            nnz = aNnz;
            idxBase = aIdxBase;
            descr = aDescr;
            typeIndices = aTypeIndices;
            typeData = aTypeData;
            format = Format.COO;
            res = CudaSparseNativeMethods.cusparseSpMatGetFormat(descr, ref format);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetFormat", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// </summary>
        internal ConstSparseMatrix(SparseMatrix<indexT, dataT> sparseMatrix)
        {
            rows = sparseMatrix.Rows;
            cols = sparseMatrix.Cols;
            nnz = sparseMatrix.Nnz;
            idxBase = sparseMatrix.IdxBase;
            descr = sparseMatrix.Descr;
            typeIndices = sparseMatrix.TypeIndices;
            typeData = sparseMatrix.TypeData;
            format = sparseMatrix.Format;
            noCleanup = true;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~ConstSparseMatrix()
        {
            Dispose(false);
        }
        #endregion

        #region Create
        //CSR
        public static ConstSparseMatrix<indexT1, dataT1> CreateConstCSR<indexT1, dataT1>(
            long rows,
            long cols,
            long nnz,
            CudaDeviceVariable<indexT1> csrRowOffsets,
            CudaDeviceVariable<indexT1> csrColInd,
            CudaDeviceVariable<dataT1> csrValues,
            IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseConstSpMatDescr descr = new cusparseConstSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateConstCsr(ref descr, rows, cols, nnz, csrRowOffsets.DevicePointer,
                csrColInd.DevicePointer, csrValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstCsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new ConstSparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }
        //CSR
        public static ConstSparseMatrix<indexT1, dataT1> CreateConstCSC<indexT1, dataT1>(
            long rows,
            long cols,
            long nnz,
            CudaDeviceVariable<indexT1> cscColOffsets,
            CudaDeviceVariable<indexT1> cscRowInd,
            CudaDeviceVariable<dataT1> cscValues,
            IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseConstSpMatDescr descr = new cusparseConstSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateConstCsc(ref descr, rows, cols, nnz, cscColOffsets.DevicePointer,
                cscRowInd.DevicePointer, cscValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstCsc", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new ConstSparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }
        //CSR
        public static ConstSparseMatrix<indexT1, dataT1> CreateConstCOO<indexT1, dataT1>(
            long rows,
            long cols,
            long nnz,
            CudaDeviceVariable<indexT1> cooRowInd,
            CudaDeviceVariable<indexT1> cooColInd,
            CudaDeviceVariable<dataT1> cooValues,
            IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseConstSpMatDescr descr = new cusparseConstSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateConstCoo(ref descr, rows, cols, nnz, cooRowInd.DevicePointer,
                cooColInd.DevicePointer, cooValues.DevicePointer, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstCoo", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new ConstSparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }

        //BlockedEll
        public static ConstSparseMatrix<indexT1, dataT1> CreateConstBlockedEll<indexT1, dataT1>(
                         long rows,
                         long cols,
                         long ellBlockSize,
                         long ellCols,
                         CudaDeviceVariable<indexT1> ellColInd,
                         CudaDeviceVariable<dataT1> ellValue,
                         IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseConstSpMatDescr descr = new cusparseConstSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateConstBlockedEll(ref descr, rows, cols, ellBlockSize, ellCols, ellColInd.DevicePointer,
                ellValue.DevicePointer, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstBlockedEll", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new ConstSparseMatrix<indexT1, dataT1>(descr, rows, cols, 0, idxBase, typeIndices, typeData);
        }

        //BSR
        public static ConstSparseMatrix<indexT1, dataT1> CreateConstBSR<indexT1, dataT1>(
                        long brows,
                        long bcols,
                        long bnnz,
                        long rowBlockDim,
                        long colBlockDim,
                        CudaDeviceVariable<indexT1> bsrRowOffsets,
                        CudaDeviceVariable<indexT1> bsrColInd,
                        CudaDeviceVariable<dataT1> bsrValues,
                        IndexBase idxBase,
                        Order order) where indexT1 : struct where dataT1 : struct
        {
            cusparseConstSpMatDescr descr = new cusparseConstSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateConstBsr(ref descr, brows, bcols, bnnz, rowBlockDim, colBlockDim, bsrRowOffsets.DevicePointer,
                bsrColInd.DevicePointer, bsrValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData, order);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstBsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new ConstSparseMatrix<indexT1, dataT1>(descr, brows, bcols, 0, idxBase, typeIndices, typeData);
        }


        //SlicedEll
        public static ConstSparseMatrix<indexT1, dataT1> CreateConstSlicedEll<indexT1, dataT1>(
                         long rows,
                         long cols,
                         long nnz,
                         long sellValuesSize,
                         long sliceSize,
                         CudaDeviceVariable<indexT1> sellSliceOffsets,
                         CudaDeviceVariable<indexT1> sellColInd,
                         CudaDeviceVariable<dataT1> sellValues,
                         IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseConstSpMatDescr descr = new cusparseConstSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateConstSlicedEll(ref descr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets.DevicePointer,
                sellColInd.DevicePointer, sellValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateConstSlicedEll", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new ConstSparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }
        #endregion

        #region Get
        /// <summary>
        /// 
        /// </summary>
        public void CscGet(out CudaDeviceVariable<indexT> cscColOffsets,
            out CudaDeviceVariable<indexT> cscRowInd,
            out CudaDeviceVariable<dataT> cscValues)
        {
            CUdeviceptr ptrColOffsets = new CUdeviceptr();
            CUdeviceptr ptrRowIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();
            IndexType indexTypeOffset = IndexType.Index32I;
            res = CudaSparseNativeMethods.cusparseConstCscGet(descr, ref rows, ref cols, ref nnz, ref ptrColOffsets,
                    ref ptrRowIdx, ref ptrValues, ref indexTypeOffset, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstCscGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            cscColOffsets = new CudaDeviceVariable<indexT>(ptrColOffsets);
            cscRowInd = new CudaDeviceVariable<indexT>(ptrRowIdx);
            cscValues = new CudaDeviceVariable<dataT>(ptrValues);
        }
        /// <summary>
        /// 
        /// </summary>
        public void CsrGet(out CudaDeviceVariable<indexT> csrRowOffsets,
            out CudaDeviceVariable<indexT> csrColInd,
            out CudaDeviceVariable<dataT> csrValues)
        {
            CUdeviceptr ptrRowOffsets = new CUdeviceptr();
            CUdeviceptr ptrColIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();
            IndexType indexTypeOffset = IndexType.Index32I;
            res = CudaSparseNativeMethods.cusparseConstCsrGet(descr, ref rows, ref cols, ref nnz, ref ptrRowOffsets,
                    ref ptrColIdx, ref ptrValues, ref indexTypeOffset, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstCsrGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            csrRowOffsets = new CudaDeviceVariable<indexT>(ptrRowOffsets);
            csrColInd = new CudaDeviceVariable<indexT>(ptrColIdx);
            csrValues = new CudaDeviceVariable<dataT>(ptrValues);
        }
        /// <summary>
        /// 
        /// </summary>
        public void CooGet(out CudaDeviceVariable<indexT> cooRowInd,
            out CudaDeviceVariable<indexT> cooColInd,
            out CudaDeviceVariable<dataT> cooValues)
        {
            CUdeviceptr ptrRowIdx = new CUdeviceptr();
            CUdeviceptr ptrColIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();
            res = CudaSparseNativeMethods.cusparseConstCooGet(descr, ref rows, ref cols, ref nnz, ref ptrRowIdx,
                    ref ptrColIdx, ref ptrValues, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstCooGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            cooRowInd = new CudaDeviceVariable<indexT>(ptrRowIdx);
            cooColInd = new CudaDeviceVariable<indexT>(ptrColIdx);
            cooValues = new CudaDeviceVariable<dataT>(ptrValues);
        }

        /// <summary>
        /// 
        /// </summary>
        public void BlockedEllGet(out CudaDeviceVariable<indexT> ellColInd,
            out CudaDeviceVariable<dataT> ellValue, out long ellBlockSize, out long ellCols)
        {
            ellBlockSize = 0;
            ellCols = 0;
            CUdeviceptr ptrIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();

            res = CudaSparseNativeMethods.cusparseConstBlockedEllGet(descr, ref rows, ref cols, ref ellBlockSize, ref ellCols, ref ptrIdx, ref ptrValues, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstBlockedEllGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            ellColInd = new CudaDeviceVariable<indexT>(ptrIdx);
            ellValue = new CudaDeviceVariable<dataT>(ptrValues);
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
                    res = CudaSparseNativeMethods.cusparseDestroySpMat(descr);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroySpMat", res));
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
        public cusparseConstSpMatDescr Descr
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
        public long Rows
        {
            get { return rows; }
        }
        /// <summary>
        /// 
        /// </summary>
        public long Cols
        {
            get { return cols; }
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
        public Format Format
        {
            get { return format; }
        }
        /// <summary>
        /// 
        /// </summary>
        public IndexBase IdxBase
        {
            get
            {
                res = CudaSparseNativeMethods.cusparseSpMatGetIndexBase(descr, ref idxBase);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetIndexBase", res));
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
            res = CudaSparseNativeMethods.cusparseConstSpMatGetValues(descr, ref devPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseConstSpMatGetValues", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return new CudaDeviceVariable<dataT>(devPtr);
        }

        /// <summary>
        /// 
        /// </summary>
        public int GetStridedBatch()
        {
            int batchCount = 0;
            res = CudaSparseNativeMethods.cusparseSpMatGetStridedBatch(descr, ref batchCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetStridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return batchCount;
        }

        /// <summary>
        /// 
        /// </summary>
        public void GetSize()
        {
            res = CudaSparseNativeMethods.cusparseSpMatGetSize(descr, ref rows, ref cols, ref nnz);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void GetAttribute(out cusparseFillMode fillMode)
        {
            fillMode = cusparseFillMode.Lower;
            res = CudaSparseNativeMethods.cusparseSpMatGetAttribute(descr, cusparseSpMatAttribute.FillMode, ref fillMode, sizeof(cusparseFillMode));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetAttribute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void GetAttribute(out cusparseDiagType diagType)
        {
            diagType = cusparseDiagType.NonUnit;
            res = CudaSparseNativeMethods.cusparseSpMatGetAttribute(descr, cusparseSpMatAttribute.DiagType, ref diagType, sizeof(cusparseDiagType));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetAttribute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
    }
    /// <summary>
    ///
    /// </summary>
    public class SparseMatrix<indexT, dataT> : IDisposable where indexT : struct where dataT : struct
    {
        private cusparseStatus res;
        private bool disposed;
        private cusparseSpMatDescr descr;
        private IndexType typeIndices;
        private cudaDataType typeData;
        private long rows;
        private long cols;
        private long nnz;
        private IndexBase idxBase;
        private Format format;



        #region Contructors
        /// <summary>
        /// </summary>
        private SparseMatrix(cusparseSpMatDescr aDescr, long aRows, long aCols, long aNnz, IndexBase aIdxBase, IndexType aTypeIndices, cudaDataType aTypeData)
        {
            rows = aRows;
            cols = aCols;
            nnz = aNnz;
            idxBase = aIdxBase;
            descr = aDescr;
            typeIndices = aTypeIndices;
            typeData = aTypeData;
            format = Format.COO;
            res = CudaSparseNativeMethods.cusparseSpMatGetFormat(descr, ref format);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetFormat", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~SparseMatrix()
        {
            Dispose(false);
        }
        #endregion

        /// <summary>
        ///
        /// </summary>
        public static implicit operator ConstSparseMatrix<indexT, dataT>(SparseMatrix<indexT, dataT> sparseMatrix)
        {
            return new ConstSparseMatrix<indexT, dataT>(sparseMatrix);
        }

        #region Create
        //CSR
        public static SparseMatrix<indexT1, dataT1> CreateCSR<indexT1, dataT1>(
            long rows,
            long cols,
            long nnz,
            CudaDeviceVariable<indexT1> csrRowOffsets,
            CudaDeviceVariable<indexT1> csrColInd,
            CudaDeviceVariable<dataT1> csrValues,
            IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseSpMatDescr descr = new cusparseSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateCsr(ref descr, rows, cols, nnz, csrRowOffsets.DevicePointer,
                csrColInd.DevicePointer, csrValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateCsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new SparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }
        //CSR
        public static SparseMatrix<indexT1, dataT1> CreateCSC<indexT1, dataT1>(
            long rows,
            long cols,
            long nnz,
            CudaDeviceVariable<indexT1> cscColOffsets,
            CudaDeviceVariable<indexT1> cscRowInd,
            CudaDeviceVariable<dataT1> cscValues,
            IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseSpMatDescr descr = new cusparseSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateCsc(ref descr, rows, cols, nnz, cscColOffsets.DevicePointer,
                cscRowInd.DevicePointer, cscValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateCsc", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new SparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }
        //COO
        public static SparseMatrix<indexT1, dataT1> CreateCOO<indexT1, dataT1>(
            long rows,
            long cols,
            long nnz,
            CudaDeviceVariable<indexT1> cooRowInd,
            CudaDeviceVariable<indexT1> cooColInd,
            CudaDeviceVariable<dataT1> cooValues,
            IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseSpMatDescr descr = new cusparseSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateCoo(ref descr, rows, cols, nnz, cooRowInd.DevicePointer,
                cooColInd.DevicePointer, cooValues.DevicePointer, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateCoo", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new SparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }

        //BlockedEll
        public static SparseMatrix<indexT1, dataT1> CreateBlockedEll<indexT1, dataT1>(
                         long rows,
                         long cols,
                         long ellBlockSize,
                         long ellCols,
                         CudaDeviceVariable<indexT1> ellColInd,
                         CudaDeviceVariable<dataT1> ellValue,
                         IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseSpMatDescr descr = new cusparseSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateBlockedEll(ref descr, rows, cols, ellBlockSize, ellCols, ellColInd.DevicePointer,
                ellValue.DevicePointer, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateBlockedEll", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new SparseMatrix<indexT1, dataT1>(descr, rows, cols, 0, idxBase, typeIndices, typeData);
        }

        //BSR
        public static SparseMatrix<indexT1, dataT1> CreateBSR<indexT1, dataT1>(
                        long brows,
                        long bcols,
                        long bnnz,
                        long rowBlockDim,
                        long colBlockDim,
                        CudaDeviceVariable<indexT1> bsrRowOffsets,
                        CudaDeviceVariable<indexT1> bsrColInd,
                        CudaDeviceVariable<dataT1> bsrValues,
                        IndexBase idxBase,
                        Order order) where indexT1 : struct where dataT1 : struct
        {
            cusparseSpMatDescr descr = new cusparseSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateBsr(ref descr, brows, bcols, bnnz, rowBlockDim, colBlockDim, bsrRowOffsets.DevicePointer,
                bsrColInd.DevicePointer, bsrValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData, order);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateBsr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new SparseMatrix<indexT1, dataT1>(descr, brows, bcols, 0, idxBase, typeIndices, typeData);
        }

        //SlicedEll
        public static SparseMatrix<indexT1, dataT1> CreateSlicedEll<indexT1, dataT1>(
                         long rows,
                         long cols,
                         long nnz,
                         long sellValuesSize,
                         long sliceSize,
                         CudaDeviceVariable<indexT1> sellSliceOffsets,
                         CudaDeviceVariable<indexT1> sellColInd,
                         CudaDeviceVariable<dataT1> sellValues,
                         IndexBase idxBase) where indexT1 : struct where dataT1 : struct
        {
            cusparseSpMatDescr descr = new cusparseSpMatDescr();
            IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
            cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
            cusparseStatus res = CudaSparseNativeMethods.cusparseCreateSlicedEll(ref descr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets.DevicePointer,
                sellColInd.DevicePointer, sellValues.DevicePointer, typeIndices, typeIndices, idxBase, typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateSlicedEll", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            return new SparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
        }
        #endregion

        #region Get
        /// <summary>
        /// 
        /// </summary>
        public void CscGet(out CudaDeviceVariable<indexT> cscColOffsets,
            out CudaDeviceVariable<indexT> cscRowInd,
            out CudaDeviceVariable<dataT> cscValues)
        {
            CUdeviceptr ptrColOffsets = new CUdeviceptr();
            CUdeviceptr ptrRowIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();
            IndexType indexTypeOffset = IndexType.Index32I;
            res = CudaSparseNativeMethods.cusparseCscGet(descr, ref rows, ref cols, ref nnz, ref ptrColOffsets,
                    ref ptrRowIdx, ref ptrValues, ref indexTypeOffset, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCscGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            cscColOffsets = new CudaDeviceVariable<indexT>(ptrColOffsets);
            cscRowInd = new CudaDeviceVariable<indexT>(ptrRowIdx);
            cscValues = new CudaDeviceVariable<dataT>(ptrValues);
        }
        /// <summary>
        /// 
        /// </summary>
        public void CsrGet(out CudaDeviceVariable<indexT> csrRowOffsets,
            out CudaDeviceVariable<indexT> csrColInd,
            out CudaDeviceVariable<dataT> csrValues)
        {
            CUdeviceptr ptrRowOffsets = new CUdeviceptr();
            CUdeviceptr ptrColIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();
            IndexType indexTypeOffset = IndexType.Index32I;
            res = CudaSparseNativeMethods.cusparseCsrGet(descr, ref rows, ref cols, ref nnz, ref ptrRowOffsets,
                    ref ptrColIdx, ref ptrValues, ref indexTypeOffset, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCsrGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            csrRowOffsets = new CudaDeviceVariable<indexT>(ptrRowOffsets);
            csrColInd = new CudaDeviceVariable<indexT>(ptrColIdx);
            csrValues = new CudaDeviceVariable<dataT>(ptrValues);
        }
        /// <summary>
        /// 
        /// </summary>
        public void CooGet(out CudaDeviceVariable<indexT> cooRowInd,
            out CudaDeviceVariable<indexT> cooColInd,
            out CudaDeviceVariable<dataT> cooValues)
        {
            CUdeviceptr ptrRowIdx = new CUdeviceptr();
            CUdeviceptr ptrColIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();
            res = CudaSparseNativeMethods.cusparseCooGet(descr, ref rows, ref cols, ref nnz, ref ptrRowIdx,
                    ref ptrColIdx, ref ptrValues, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCooGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            cooRowInd = new CudaDeviceVariable<indexT>(ptrRowIdx);
            cooColInd = new CudaDeviceVariable<indexT>(ptrColIdx);
            cooValues = new CudaDeviceVariable<dataT>(ptrValues);
        }

        /// <summary>
        /// 
        /// </summary>
        public void BlockedEllGet(out CudaDeviceVariable<indexT> ellColInd,
            out CudaDeviceVariable<dataT> ellValue, out long ellBlockSize, out long ellCols)
        {
            ellBlockSize = 0;
            ellCols = 0;
            CUdeviceptr ptrIdx = new CUdeviceptr();
            CUdeviceptr ptrValues = new CUdeviceptr();

            res = CudaSparseNativeMethods.cusparseBlockedEllGet(descr, ref rows, ref cols, ref ellBlockSize, ref ellCols, ref ptrIdx, ref ptrValues, ref typeIndices, ref idxBase, ref typeData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseBlockedEllGet", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);

            ellColInd = new CudaDeviceVariable<indexT>(ptrIdx);
            ellValue = new CudaDeviceVariable<dataT>(ptrValues);
        }
        #endregion

        #region Set
        /// <summary>
        /// 
        /// </summary>
        public void CsrSet(CudaDeviceVariable<indexT> csrRowOffsets,
            CudaDeviceVariable<indexT> csrColInd,
            CudaDeviceVariable<dataT> csrValues)
        {
            res = CudaSparseNativeMethods.cusparseCsrSetPointers(descr, csrRowOffsets.DevicePointer,
                    csrColInd.DevicePointer, csrValues.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCsrSetPointers", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        public void CscSet(CudaDeviceVariable<indexT> cscColOffsets,
            CudaDeviceVariable<indexT> cscRowInd,
            CudaDeviceVariable<dataT> cscValues)
        {
            res = CudaSparseNativeMethods.cusparseCscSetPointers(descr, cscColOffsets.DevicePointer,
                    cscRowInd.DevicePointer, cscValues.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCscSetPointers", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        public void CooSet(CudaDeviceVariable<indexT> cooRows,
            CudaDeviceVariable<indexT> cooColumns,
            CudaDeviceVariable<dataT> cooValues)
        {
            res = CudaSparseNativeMethods.cusparseCooSetPointers(descr, cooRows.DevicePointer,
                    cooColumns.DevicePointer, cooValues.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCooSetPointers", res));

            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
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
                res = CudaSparseNativeMethods.cusparseDestroySpMat(descr);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroySpMat", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cusparseSpMatDescr Descr
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
        public long Rows
        {
            get { return rows; }
        }
        /// <summary>
        /// 
        /// </summary>
        public long Cols
        {
            get { return cols; }
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
        public Format Format
        {
            get { return format; }
        }
        /// <summary>
        /// 
        /// </summary>
        public IndexBase IdxBase
        {
            get
            {
                res = CudaSparseNativeMethods.cusparseSpMatGetIndexBase(descr, ref idxBase);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetIndexBase", res));
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
            res = CudaSparseNativeMethods.cusparseSpMatGetValues(descr, ref devPtr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetValues", res));
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
            res = CudaSparseNativeMethods.cusparseSpMatSetValues(descr, data.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatSetValues", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        public int GetStridedBatch()
        {
            int batchCount = 0;
            res = CudaSparseNativeMethods.cusparseSpMatGetStridedBatch(descr, ref batchCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetStridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
            return batchCount;
        }

        /// <summary>
        /// 
        /// </summary>
        public void SetStridedBatch(int batchCount)
        {
            res = CudaSparseNativeMethods.cusparseSpMatSetStridedBatch(descr, batchCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatSetStridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        public void SetStridedBatchCoo(int batchCount,
                            long batchStride)
        {
            res = CudaSparseNativeMethods.cusparseCooSetStridedBatch(descr, batchCount, batchStride);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCooSetStridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        public void SetStridedBatchCsr(int batchCount,
                            long offsetsBatchStride, long columnsValuesBatchStride)
        {
            res = CudaSparseNativeMethods.cusparseCsrSetStridedBatch(descr, batchCount, offsetsBatchStride, columnsValuesBatchStride);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCsrSetStridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }


        /// <summary>
        /// 
        /// </summary>
        public void SetStridedBatchBsr(int batchCount,
                           long offsetsBatchStride,
                           long columnsValuesBatchStride,
                           long ValuesBatchStride)
        {
            res = CudaSparseNativeMethods.cusparseBsrSetStridedBatch(descr, batchCount, offsetsBatchStride, columnsValuesBatchStride, ValuesBatchStride);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseBsrSetStridedBatch", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        public void GetSize()
        {
            res = CudaSparseNativeMethods.cusparseSpMatGetSize(descr, ref rows, ref cols, ref nnz);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetSize", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void SetAttribute(cusparseFillMode fillMode)
        {
            res = CudaSparseNativeMethods.cusparseSpMatSetAttribute(descr, cusparseSpMatAttribute.FillMode, ref fillMode, sizeof(cusparseFillMode));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatSetAttribute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void GetAttribute(out cusparseFillMode fillMode)
        {
            fillMode = cusparseFillMode.Lower;
            res = CudaSparseNativeMethods.cusparseSpMatGetAttribute(descr, cusparseSpMatAttribute.FillMode, ref fillMode, sizeof(cusparseFillMode));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetAttribute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void SetAttribute(cusparseDiagType diagType)
        {
            res = CudaSparseNativeMethods.cusparseSpMatSetAttribute(descr, cusparseSpMatAttribute.DiagType, ref diagType, sizeof(cusparseDiagType));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatSetAttribute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }

        public void GetAttribute(out cusparseDiagType diagType)
        {
            diagType = cusparseDiagType.NonUnit;
            res = CudaSparseNativeMethods.cusparseSpMatGetAttribute(descr, cusparseSpMatAttribute.DiagType, ref diagType, sizeof(cusparseDiagType));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetAttribute", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
    }
}
