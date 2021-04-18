//	Copyright (c) 2012, Michael Kunz. All rights reserved.
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
using System.Net.NetworkInformation;

namespace ManagedCuda.CudaSparse
{
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
		//CSR
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
		//CSR
		public static SparseMatrix<indexT1, dataT1> CreateCooAoS<indexT1, dataT1>(
			long rows,
			long cols,
			long nnz,
			CudaDeviceVariable<indexT1> cooInd,
			CudaDeviceVariable<dataT1> cooValues,
			IndexBase idxBase) where indexT1 : struct where dataT1 : struct
		{
			cusparseSpMatDescr descr = new cusparseSpMatDescr();
			IndexType typeIndices = IndexTypeTranslator.GetType(typeof(indexT1));
			cudaDataType typeData = CudaDataTypeTranslator.GetType(typeof(dataT1));
			cusparseStatus res = CudaSparseNativeMethods.cusparseCreateCooAoS(ref descr, rows, cols, nnz, cooInd.DevicePointer,
				cooValues.DevicePointer, typeIndices, idxBase, typeData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateCooAoS", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);

			return new SparseMatrix<indexT1, dataT1>(descr, rows, cols, nnz, idxBase, typeIndices, typeData);
		}
		#endregion

		#region Get
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
		public void CooAoSGet(out CudaDeviceVariable<indexT> cooInd,
			out CudaDeviceVariable<dataT> cooValues)
		{
			CUdeviceptr ptrIdx = new CUdeviceptr();
			CUdeviceptr ptrValues = new CUdeviceptr();
			res = CudaSparseNativeMethods.cusparseCooAoSGet(descr, ref rows, ref cols, ref nnz, ref ptrIdx, ref ptrValues, ref typeIndices, ref idxBase, ref typeData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCooAoSGet", res));

			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);

			cooInd = new CudaDeviceVariable<indexT>(ptrIdx);
			cooValues = new CudaDeviceVariable<dataT>(ptrValues);
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
		public void GetSize()
		{
			res = CudaSparseNativeMethods.cusparseSpMatGetSize(descr, ref rows, ref cols, ref nnz);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSpMatGetSize", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}
	}
}
