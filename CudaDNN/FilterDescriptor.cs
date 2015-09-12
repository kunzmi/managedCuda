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
	public class FilterDescriptor : IDisposable
	{
		private cudnnFilterDescriptor _desc;
		private cudnnStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public FilterDescriptor()
		{
			_desc = new cudnnFilterDescriptor();
			res = CudaDNNNativeMethods.cudnnCreateFilterDescriptor(ref _desc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateFilterDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~FilterDescriptor()
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
				res = CudaDNNNativeMethods.cudnnDestroyFilterDescriptor(_desc);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyFilterDescriptor", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cudnnFilterDescriptor Desc
		{
			get { return _desc; }
		}


		public void SetFilter4dDescriptor(cudnnDataType dataType, // image data type
												int k,        // number of output feature maps
												int c,        // number of input feature maps
												int h,        // height of each input filter
												int w         // width of  each input fitler
											)
		{
			res = CudaDNNNativeMethods.cudnnSetFilter4dDescriptor(_desc, dataType, k, c, h, w);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetFilter4dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void GetFilter4dDescriptor(ref cudnnDataType dataType, // image data type
											ref int k,        // number of output feature maps
											ref int c,        // number of input feature maps
											ref int h,        // height of each input filter
											ref int w         // width of  each input fitler
										)
		{
			res = CudaDNNNativeMethods.cudnnGetFilter4dDescriptor(_desc, ref dataType, ref k, ref c, ref h, ref w);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetFilter4dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void SetFilterNdDescriptor(cudnnDataType dataType, // image data type
											int nbDims,
											int[] filterDimA
											)
		{
			res = CudaDNNNativeMethods.cudnnSetFilterNdDescriptor(_desc, dataType, nbDims, filterDimA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetFilterNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void GetFilterNdDescriptor(int nbDimsRequested,
											ref cudnnDataType dataType, // image data type
											ref int nbDims,
											int[] filterDimA
										)
		{
			res = CudaDNNNativeMethods.cudnnGetFilterNdDescriptor(_desc, nbDimsRequested, ref dataType, ref nbDims, filterDimA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetFilterNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
	}
}
