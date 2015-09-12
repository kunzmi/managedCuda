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
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaFFT
{
    /// <summary>
    /// Creates a FFT plan configuration of dimension rank, with sizes
    /// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
    /// many transforms to configure in parallel. With this function, batched
    /// plans of any dimension may be created.<para/>
    /// Input parameters <c>inembed</c>, <c>istride</c>, and <c>idist</c> and output
    /// parameters <c>onembed</c>, <c>ostride</c>, and <c>odist</c> will allow setup of noncontiguous
    /// input data in a future version (Beta status in version 4.0). Note that for CUFFT 3.0,
    /// these parameters are ignored and the layout of batched data must be
    /// side‐by‐side and not interleaved.
    /// </summary>
    public class CudaFFTPlanMany64
    {
        private cufftHandle _handle;
        private cufftResult res;
        private bool disposed;
        private int _rank;
		private long[] _n;
		private long _batch;
        private cufftType _type;
        //optional:
		private long[] _inembed;
		private long _istride;
		private long _idist;
		private long[] _onembed;
		private long _ostride;
		private long _odist;

        #region Contructors

		#region newApiSimplified
		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, ref SizeT size)
		{
			_handle = handle;
			_rank = rank;
			_n = n;
			_batch = batch;
			_type = type;

			//optional:
			_inembed = null;
			_istride = 1;
			_idist = 0;
			_onembed = null;
			_ostride = 1;
			_odist = 0;
			res = CudaFFTNativeMethods.cufftMakePlanMany64(_handle, _rank, _n, _inembed, _istride, _idist, _onembed, _ostride, _odist, _type, _batch, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftMakePlanMany64", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type)
		{
			SizeT size = new SizeT();
			_handle = handle;
			_rank = rank;
			_n = n;
			_batch = batch;
			_type = type;

			//optional:
			_inembed = null;
			_istride = 1;
			_idist = 0;
			_onembed = null;
			_ostride = 1;
			_odist = 0;
			res = CudaFFTNativeMethods.cufftMakePlanMany64(_handle, _rank, _n, _inembed, _istride, _idist, _onembed, _ostride, _odist, _type, _batch, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftMakePlanMany64", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream)
			: this(handle, rank, n, batch, type)
		{
			SetStream(stream);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, Compatibility mode)
			: this(handle, rank, n, batch, type)
		{
			SetCompatibilityMode(mode);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream, Compatibility mode)
			: this(handle, rank, n, batch, type)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
		}


		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream, ref SizeT size)
			: this(handle, rank, n, batch, type, ref size)
		{
			SetStream(stream);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, Compatibility mode, ref SizeT size)
			: this(handle, rank, n, batch, type, ref size)
		{
			SetCompatibilityMode(mode);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream, Compatibility mode, ref SizeT size)
			: this(handle, rank, n, batch, type, ref size)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, bool autoAllocate)
			: this(handle, rank, n, batch, type)
		{
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, ref size)
		{
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream, bool autoAllocate)
			: this(handle, rank, n, batch, type)
		{
			SetStream(stream);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, Compatibility mode, bool autoAllocate)
			: this(handle, rank, n, batch, type)
		{
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream, Compatibility mode, bool autoAllocate)
			: this(handle, rank, n, batch, type)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}


		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, ref size)
		{
			SetStream(stream);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, Compatibility mode, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, ref size)
		{
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, CUstream stream, Compatibility mode, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, ref size)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}
		#endregion

		#region newApiComplete
		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, ref SizeT size)
		{
			_handle = handle;
			_rank = rank;
			_n = n;
			_batch = batch;
			_type = type;

			//optional:
			_inembed = inembed;
			_istride = istride;
			_idist = idist;
			_onembed = onembed;
			_ostride = ostride;
			_odist = odist;
			res = CudaFFTNativeMethods.cufftMakePlanMany64(_handle, _rank, _n, _inembed, _istride, _idist, _onembed, _ostride, _odist, _type, _batch, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftMakePlanMany64", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist)
		{
			SizeT size = new SizeT();
			_handle = handle;
			_rank = rank;
			_n = n;
			_batch = batch;
			_type = type;

			//optional:
			_inembed = inembed;
			_istride = istride;
			_idist = idist;
			_onembed = onembed;
			_ostride = ostride;
			_odist = odist;
			res = CudaFFTNativeMethods.cufftMakePlanMany64(_handle, _rank, _n, _inembed, _istride, _idist, _onembed, _ostride, _odist, _type, _batch, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftMakePlanMany64", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist)
		{
			SetStream(stream);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, Compatibility mode)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist)
		{
			SetCompatibilityMode(mode);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream, Compatibility mode)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
		}


		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream, ref SizeT size)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist, ref size)
		{
			SetStream(stream);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, Compatibility mode, ref SizeT size)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist, ref size)
		{
			SetCompatibilityMode(mode);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream, Compatibility mode, ref SizeT size)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist, ref size)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist)
		{
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist, ref size)
		{
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist)
		{
			SetStream(stream);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, Compatibility mode, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist)
		{
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream, Compatibility mode, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}


		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist, ref size)
		{
			SetStream(stream);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, Compatibility mode, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist, ref size)
		{
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes
		/// specified in the array <c>n</c>. The <c>batch</c> input parameter tells CUFFT how
		/// many transforms to configure in parallel. With this function, batched
		/// plans of any dimension may be created. (new API)
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="inembed">See CUFFT Manual</param>
		/// <param name="istride">See CUFFT Manual</param>
		/// <param name="idist">See CUFFT Manual</param>
		/// <param name="onembed">See CUFFT Manual</param>
		/// <param name="ostride">See CUFFT Manual</param>
		/// <param name="odist">See CUFFT Manual</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <param name="size"></param>
		/// <param name="autoAllocate">indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated.</param>
		public CudaFFTPlanMany64(cufftHandle handle, int rank, long[] n, long batch, cufftType type, long[] inembed, long istride, long idist, long[] onembed, long ostride, long odist, CUstream stream, Compatibility mode, ref SizeT size, bool autoAllocate)
			: this(handle, rank, n, batch, type, inembed, istride, idist, onembed, ostride, odist, ref size)
		{
			SetStream(stream);
			SetCompatibilityMode(mode);
			SetAutoAllocation(autoAllocate);
		}
		#endregion

		/// <summary>
        /// For dispose
        /// </summary>
        ~CudaFFTPlanMany64()
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
                res = CudaFFTNativeMethods.cufftDestroy(_handle);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

		#region Methods
		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than
		/// cufftEstimate1d(), given the specified parameters, and taking into account any plan
		/// settings that may have been made.
		/// </summary>
		/// <returns></returns>
		public SizeT GetSize()
		{
			SizeT size = new SizeT();
			res = CudaFFTNativeMethods.cufftGetSizeMany64(_handle, _rank, _n, _inembed, _istride, _idist, _onembed, _ostride, _odist, _type, _batch, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftGetSizeMany64", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
			return size;
		}

		///// <summary>
		///// During plan execution, CUFFT requires a work area for temporary storage of
		///// intermediate results. This call returns an estimate for the size of the work area required,
		///// given the specified parameters, and assuming default plan settings. Note that changing
		///// some plan settings, such as compatibility mode, may alter the size required for the work
		///// area.
		///// </summary>
		///// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		///// <param name="n">An array of size rank, describing the size of each dimension</param>
		///// <param name="batch">Batch size for this transform</param>
		///// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		///// <param name="inembed">See CUFFT Manual</param>
		///// <param name="istride">See CUFFT Manual</param>
		///// <param name="idist">See CUFFT Manual</param>
		///// <param name="onembed">See CUFFT Manual</param>
		///// <param name="ostride">See CUFFT Manual</param>
		///// <param name="odist">See CUFFT Manual</param>
		///// <returns></returns>
		//public static SizeT EstimateSize(int rank, int[] n, int batch, cufftType type, int[] inembed, int istride, int idist, int[] onembed, int ostride, int odist)
		//{
		//	SizeT size = new SizeT();
		//	cufftResult res = CudaFFTNativeMethods.cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, ref size);
		//	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftEstimateMany", res));
		//	if (res != cufftResult.Success)
		//		throw new CudaFFTException(res);
		//	return size;
		//}

		/// <summary>
		/// Once plan generation has been done, either with the original API or the extensible API,
		/// this call returns the actual size of the work area required to support the plan. Callers
		/// who choose to manage work area allocation within their application must use this call
		/// after plan generation, and after any cufftSet*() calls subsequent to plan generation, if
		/// those calls might alter the required work space size.
		/// </summary>
		/// <returns></returns>
		public SizeT GetActualSize()
		{
			SizeT size = new SizeT();
			res = CudaFFTNativeMethods.cufftGetSize(_handle, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftGetSize", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
			return size;
		}

		/// <summary>
		/// SetWorkArea() overrides the work area pointer associated with a plan.
		/// If the work area was auto-allocated, CUFFT frees the auto-allocated space. The
		/// cufftExecute*() calls assume that the work area pointer is valid and that it points to
		/// a contiguous region in device memory that does not overlap with any other work area. If
		/// this is not the case, results are indeterminate.
		/// </summary>
		/// <param name="workArea"></param>
		public void SetWorkArea(CUdeviceptr workArea)
		{
			res = CudaFFTNativeMethods.cufftSetWorkArea(_handle, workArea);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetWorkArea", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}

		/// <summary>
		/// SetAutoAllocation() indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated. CUFFT default behavior is to allocate
		/// the work area at plan generation time. If cufftSetAutoAllocation() has been called
		/// with autoAllocate set to "false" prior to one of the cufftMakePlan*() calls, CUFFT
		/// does not allocate the work area. This is the preferred sequence for callers wishing to
		/// manage work area allocation.
		/// </summary>
		/// <param name="autoAllocate"></param>
		public void SetAutoAllocation(bool autoAllocate)
		{
			int auto = 0;
			if (autoAllocate) auto = 1;
			res = CudaFFTNativeMethods.cufftSetAutoAllocation(_handle, auto);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetAutoAllocation", res));
			if (res != cufftResult.Success)
				throw new CudaFFTException(res);
		}


        /// <summary>
        /// Executes a CUFTT transorm as defined by the cufftType.
        /// If idata and odata are the
        /// same, this method does an in‐place transform.
        /// </summary>
        /// <param name="idata"></param>
        /// <param name="odata"></param>
        /// <param name="direction">Only unsed for transformations where direction is not implicitly given by type</param>
        public void Exec(CUdeviceptr idata, CUdeviceptr odata, TransformDirection direction)
        {
            switch (_type)
            {
                case cufftType.R2C:
                    res = CudaFFTNativeMethods.cufftExecR2C(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecR2C", res));
                    break;
                case cufftType.C2R:
                    res = CudaFFTNativeMethods.cufftExecC2R(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecC2R", res));
                    break;
                case cufftType.C2C:
                    res = CudaFFTNativeMethods.cufftExecC2C(_handle, idata, odata, direction);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecC2C", res));
                    break;
                case cufftType.D2Z:
                    res = CudaFFTNativeMethods.cufftExecD2Z(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecD2Z", res));
                    break;
                case cufftType.Z2D:
                    res = CudaFFTNativeMethods.cufftExecZ2D(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecZ2D", res));
                    break;
                case cufftType.Z2Z:
                    res = CudaFFTNativeMethods.cufftExecZ2Z(_handle, idata, odata, direction);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecZ2Z", res));
                    break;
                default:
                    break;
            }
            if (res != cufftResult.Success)
                throw new CudaFFTException(res);
        }

        /// <summary>
        /// Executes a CUFTT transorm as defined by the cufftType.
        /// This method does an in‐place transform.
        /// </summary>
        /// <param name="iodata"></param>
        /// <param name="direction">Only unsed for transformations where direction is not implicitly given by type</param>
        public void Exec(CUdeviceptr iodata, TransformDirection direction)
        {
            switch (_type)
            {
                case cufftType.R2C:
                    res = CudaFFTNativeMethods.cufftExecR2C(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecR2C", res));
                    break;
                case cufftType.C2R:
                    res = CudaFFTNativeMethods.cufftExecC2R(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecC2R", res));
                    break;
                case cufftType.C2C:
                    res = CudaFFTNativeMethods.cufftExecC2C(_handle, iodata, iodata, direction);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecC2C", res));
                    break;
                case cufftType.D2Z:
                    res = CudaFFTNativeMethods.cufftExecD2Z(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecD2Z", res));
                    break;
                case cufftType.Z2D:
                    res = CudaFFTNativeMethods.cufftExecZ2D(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecZ2D", res));
                    break;
                case cufftType.Z2Z:
                    res = CudaFFTNativeMethods.cufftExecZ2Z(_handle, iodata, iodata, direction);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecZ2Z", res));
                    break;
                default:
                    break;
            }
            if (res != cufftResult.Success)
                throw new CudaFFTException(res);
        }

        /// <summary>
        /// Executes a CUFTT transorm as defined by the cufftType.
        /// If idata and odata are the
        /// same, this method does an in‐place transform.<para/>
        /// This method is only valid for transform types where transorm direction is implicitly 
        /// given by the type (i.e. not C2C and not Z2Z)
        /// </summary>
        /// <param name="idata"></param>
        /// <param name="odata"></param>
        public void Exec(CUdeviceptr idata, CUdeviceptr odata)
        {
            switch (_type)
            {
                case cufftType.R2C:
                    res = CudaFFTNativeMethods.cufftExecR2C(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecR2C", res));
                    break;
                case cufftType.C2R:
                    res = CudaFFTNativeMethods.cufftExecC2R(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecC2R", res));
                    break;
                case cufftType.D2Z:
                    res = CudaFFTNativeMethods.cufftExecD2Z(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecD2Z", res));
                    break;
                case cufftType.Z2D:
                    res = CudaFFTNativeMethods.cufftExecZ2D(_handle, idata, odata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecZ2D", res));
                    break;
                default:
                    throw new ArgumentException("For transformation not of type R2C, C2R, D2Z or Z2D, you must specify a transform direction.");
            }
            if (res != cufftResult.Success)
                throw new CudaFFTException(res);
        }

        /// <summary>
        /// Executes a CUFTT transorm as defined by the cufftType.
        /// This method does an in‐place transform.<para/>
        /// This method is only valid for transform types where transorm direction is implicitly 
        /// given by the type (i.e. not C2C and not Z2Z)
        /// </summary>
        /// <param name="iodata"></param>
        public void Exec(CUdeviceptr iodata)
        {
            switch (_type)
            {
                case cufftType.R2C:
                    res = CudaFFTNativeMethods.cufftExecR2C(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecR2C", res));
                    break;
                case cufftType.C2R:
                    res = CudaFFTNativeMethods.cufftExecC2R(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecC2R", res));
                    break;
                case cufftType.D2Z:
                    res = CudaFFTNativeMethods.cufftExecD2Z(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecD2Z", res));
                    break;
                case cufftType.Z2D:
                    res = CudaFFTNativeMethods.cufftExecZ2D(_handle, iodata, iodata);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftExecZ2D", res));
                    break;
                default:
                    throw new ArgumentException("For transformation not of type R2C, C2R, D2Z or Z2D, you must specify a transform direction.");
            }
            if (res != cufftResult.Success)
                throw new CudaFFTException(res);
        }

        /// <summary>
        /// Associates a CUDA stream with a CUFFT plan. All kernel launches
        /// made during plan execution are now done through the associated
        /// stream, enabling overlap with activity in other streams (for example,
        /// data copying). The association remains until the plan is destroyed or
        /// the stream is changed with another call to SetStream().
        /// </summary>
        /// <param name="stream"></param>
        public void SetStream(CUstream stream)
        {
            res = CudaFFTNativeMethods.cufftSetStream(_handle, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetStream", res));
            if (res != cufftResult.Success)
                throw new CudaFFTException(res);
        }

        /// <summary>
        /// configures the layout of CUFFT output in FFTW‐compatible modes.
        /// When FFTW compatibility is desired, it can be configured for padding
        /// only, for asymmetric complex inputs only, or to be fully compatible.
        /// </summary>
        /// <param name="mode"></param>
        public void SetCompatibilityMode(Compatibility mode)
        {
            res = CudaFFTNativeMethods.cufftSetCompatibilityMode(_handle, mode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cufftSetCompatibilityMode", res));
            if (res != cufftResult.Success)
                throw new CudaFFTException(res);
        }
        #endregion

        #region Properties
        /// <summary>
        /// Dimensionality of the transform (1, 2, or 3)
        /// </summary>
        public int Rank
        {
            get { return _rank; }
        }

        /// <summary>
        /// An array of size rank, describing the size of each dimension
        /// </summary>
        public long[] N
        {
            get { return _n; }
        }

        /// <summary>
        /// Batch size for this transform
        /// </summary>
		public long Batch
        {
            get { return _batch; }
        }

        /// <summary>
        /// Transform data type (e.g., C2C, as per other CUFFT calls)
        /// </summary>
        public cufftType Type
        {
            get { return _type; }
        }

        /// <summary>
        /// See CUFFT Manual
        /// </summary>
		public long[] Inembed
        {
            get { return _inembed; }
        }

        /// <summary>
        /// See CUFFT Manual
        /// </summary>
		public long Istride
        {
            get { return _istride; }
        }

        /// <summary>
        /// See CUFFT Manual
        /// </summary>
		public long Idist
        {
            get { return _idist; }
        }

        /// <summary>
        /// See CUFFT Manual
        /// </summary>
		public long[] Onembed
        {
            get { return _onembed; }
        }

        /// <summary>
        /// See CUFFT Manual
        /// </summary>
		public long Ostride
        {
            get { return _ostride; }
        }

        /// <summary>
        /// See CUFFT Manual
        /// </summary>
		public long Odist
        {
            get { return _odist; }
        }

		/// <summary>
		/// Handle
		/// </summary>
		public cufftHandle Handle
		{
			get { return _handle; }
		}
        #endregion
    }
}
