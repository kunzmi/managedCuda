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

namespace ManagedCuda.CudaSparse
{
	/// <summary>
	/// Wrapper class for cusparseMatDescr handle.
	/// </summary>
	public class CudaSparseMatrixDescriptor : IDisposable
	{
		private cusparseMatDescr _descr;
		private cusparseStatus res;
		private bool disposed;
		
        #region Contructors
        /// <summary>
        /// When the matrix descriptor is created, its fields are initialized to: 
		/// CUSPARSE_MATRIXYPE_GENERAL
		/// CUSPARSE_INDEX_BASE_ZERO
		/// All other fields are uninitialized
        /// </summary>
        public CudaSparseMatrixDescriptor()
        {
            _descr = new cusparseMatDescr();
            res = CudaSparseNativeMethods.cusparseCreateMatDescr(ref _descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateMatDescr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
        }
		
        /// <summary>
        /// When the matrix descriptor is created, its fields are initialized to: 
		/// CUSPARSE_MATRIXYPE_GENERAL
		/// CUSPARSE_INDEX_BASE_ZERO
        /// </summary>
        public CudaSparseMatrixDescriptor(cusparseFillMode fillMode, cusparseDiagType diagType)
        {
            _descr = new cusparseMatDescr();
            res = CudaSparseNativeMethods.cusparseCreateMatDescr(ref _descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateMatDescr", res));
            if (res != cusparseStatus.Success)
                throw new CudaSparseException(res);
			SetMatFillMode(fillMode);
			SetMatDiagType(diagType);
        }
		
        /// <summary>
        /// Creates a new CudaSparseMatrixDescriptor
        /// </summary>
        public CudaSparseMatrixDescriptor(cusparseMatrixType matrixType, cusparseFillMode fillMode, cusparseDiagType diagType, cusparseIndexBase indexBase)
        {
            _descr = new cusparseMatDescr();
            res = CudaSparseNativeMethods.cusparseCreateMatDescr(ref _descr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateMatDescr", res));
            if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
			SetMatType(matrixType);
			SetMatFillMode(fillMode);
			SetMatDiagType(diagType);
			SetMatIndexBase(indexBase);
        }

        /// <summary>
        /// For dispose
        /// </summary>
		~CudaSparseMatrixDescriptor()
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
				res = CudaSparseNativeMethods.cusparseDestroyMatDescr(_descr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroyMatDescr", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

		#region Methods
		/// <summary>
		/// Sets the matrix type
		/// </summary>
		/// <param name="type"></param>
		public void SetMatType(cusparseMatrixType type)
		{
			res = CudaSparseNativeMethods.cusparseSetMatType(_descr, type);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSetMatType", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// Returns matrix type
		/// </summary>
		/// <returns></returns>
		public cusparseMatrixType GetMatType()
		{
			cusparseMatrixType type = CudaSparseNativeMethods.cusparseGetMatType(_descr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetMatType", res));
			return type;
		}

		/// <summary>
		/// Sets matrix fill mode
		/// </summary>
		/// <param name="fillMode"></param>
		public void SetMatFillMode(cusparseFillMode fillMode)
		{
			res = CudaSparseNativeMethods.cusparseSetMatFillMode(_descr, fillMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSetMatFillMode", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// Returns matrix fill mode
		/// </summary>
		/// <returns></returns>
		public cusparseFillMode GetMatFillMode()
		{
			cusparseFillMode fillMode = CudaSparseNativeMethods.cusparseGetMatFillMode(_descr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetMatFillMode", res));
			return fillMode;
		}

		/// <summary>
		/// Sets matrix diagonal type
		/// </summary>
		/// <param name="diagType"></param>
		public void SetMatDiagType(cusparseDiagType diagType)
		{
			res = CudaSparseNativeMethods.cusparseSetMatDiagType(_descr, diagType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSetMatDiagType", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// Returns matrix diagonal type
		/// </summary>
		/// <returns></returns>
		public cusparseDiagType GetMatDiagType()
		{
			cusparseDiagType diagType = CudaSparseNativeMethods.cusparseGetMatDiagType(_descr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetMatDiagType", res));
			return diagType;
		}

		/// <summary>
		/// Sets matrix index base
		/// </summary>
		/// <param name="indexBase"></param>
		public void SetMatIndexBase(cusparseIndexBase indexBase)
		{
			res = CudaSparseNativeMethods.cusparseSetMatIndexBase(_descr, indexBase);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseSetMatIndexBase", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// Returns matrix index base.
		/// </summary>
		/// <returns></returns>
		public cusparseIndexBase GetMatIndexBase()
		{
			cusparseIndexBase indexBase = CudaSparseNativeMethods.cusparseGetMatIndexBase(_descr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseGetMatIndexBase", res));
			return indexBase;
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cusparseMatDescr Descriptor
		{
			get { return _descr; }
		}
	}
}
