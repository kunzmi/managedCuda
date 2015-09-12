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
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.CudaFFT
{
	/// <summary>
	/// C# wrapper for the NVIDIA CUFFT API (--> cufft.h)
	/// </summary>
	public static class CudaFFTNativeMethods
	{        
		//unfortunately Nvidia provides different dll-names for x86 and x64. Use preprocessor macro to switch names:
#if _x64
		internal const string CUFFT_API_DLL_NAME = "cufft64_75";
#else
		internal const string CUFFT_API_DLL_NAME = "cufft32_75";
#endif


		/// <summary>
		/// Creates a 1D FFT plan configuration for a specified signal size and data
		/// type. The <c>batch</c> input parameter tells CUFFT how many 1D
		/// transforms to configure.
		/// </summary>
		/// <param name="plan">Pointer to a <see cref="cufftHandle"/> object</param>
		/// <param name="nx">The transform size (e.g., 256 for a 256-point FFT)</param>
		/// <param name="type">The transform data type (e.g., C2C for complex to complex)</param>
		/// <param name="batch">Number of transforms of size nx</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.Success"/>, <see cref="cufftResult.AllocFailed"/>, 
		/// <see cref="cufftResult.InvalidType"/>, <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.InternalError"/>, 
		/// <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidSize"/>, 
		/// </returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftPlan1d([In, Out] ref cufftHandle plan, [In] int nx, [In] cufftType type, [In] int batch);

		/// <summary>
		/// Creates a 2D FFT plan configuration according to specified signal sizes
		/// and data type. This function is the same as <see cref="cufftPlan1d"/> except that
		/// it takes a second size parameter, <c>ny</c>, and does not support batching.
		/// </summary>
		/// <param name="plan">Pointer to a <see cref="cufftHandle"/> object</param>
		/// <param name="nx">The transform size in the X dimension (number of rows)</param>
		/// <param name="ny">The transform size in the Y dimension (number of columns)</param>
		/// <param name="type">The transform data type (e.g., C2R for complex to real)</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.Success"/>, <see cref="cufftResult.AllocFailed"/>, 
		/// <see cref="cufftResult.InvalidType"/>, <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.InternalError"/>, 
		/// <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidSize"/>, 
		/// </returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftPlan2d([In, Out] ref cufftHandle plan, [In] int nx, [In] int ny, [In] cufftType type);

		/// <summary>
		/// Creates a 3D FFT plan configuration according to specified signal sizes
		/// and data type. This function is the same as <see cref="cufftPlan2d"/> except that
		/// it takes a third size parameter <c>nz</c>.
		/// </summary>
		/// <param name="plan">Pointer to a <see cref="cufftHandle"/> object</param>
		/// <param name="nx">The transform size in the X dimension</param>
		/// <param name="ny">The transform size in the Y dimension</param>
		/// <param name="nz">The transform size in the Z dimension</param>
		/// <param name="type">The transform data type (e.g., R2C for real to complex)</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.Success"/>, <see cref="cufftResult.AllocFailed"/>, 
		/// <see cref="cufftResult.InvalidType"/>, <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.InternalError"/>, 
		/// <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidSize"/>, 
		/// </returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftPlan3d([In, Out] ref cufftHandle plan, [In] int nx, [In] int ny, [In] int nz, [In] cufftType type);

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
		/// <param name="plan">Pointer to a <see cref="cufftHandle"/> object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">An array of size rank, describing the size of each dimension</param>
		/// <param name="inembed">Pointer of size rank that indicates the storage dimensions of the input data in memory</param>
		/// <param name="istride">Defines the distance between two successive input elements in the least significant (i.e., innermost) dimension</param>
		/// <param name="idist">Indicates the distance between the first element of two consecutive batches in the input data</param>
		/// <param name="onembed">Pointer of size rank that indicates the storage dimensions of the output data in memory</param>
		/// <param name="ostride">Defines the distance between two successive output elements in the output array in the least significant (i.e., innermost) dimension</param>
		/// <param name="odist">Indicates the distance between the first element of two consecutive batches in the output data</param>
		/// <param name="type">Transform data type (e.g., C2C, as per other CUFFT calls)</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.Success"/>, <see cref="cufftResult.AllocFailed"/>, 
		/// <see cref="cufftResult.InvalidType"/>, <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.InternalError"/>, 
		/// <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidSize"/>, 
		/// </returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftPlanMany([In, Out] ref cufftHandle plan, [In] int rank, [In] int[] n, [In] int[] inembed, [In] int istride, [In] int idist,
								   [In] int[] onembed, [In] int ostride, [In] int odist, [In] cufftType type, [In] int batch);



		/// <summary>
		/// Following a call to cufftCreate() makes a 1D FFT plan configuration for a specified
		/// signal size and data type. The batch input parameter tells CUFFT how many 1D 
		/// transforms to configure.
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="nx">The transform size (e.g. 256 for a 256-point FFT)</param>
		/// <param name="type">The transform data type (e.g., CUFFT_C2C for single precision complex to complex)</param>
		/// <param name="batch">Number of transforms of size nx</param>
		/// <param name="workSize"></param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftMakePlan1d(cufftHandle plan, 
                                     int nx, 
                                     cufftType type, 
                                     int batch, /* deprecated - use cufftPlanMany */
                                     ref SizeT workSize);

		/// <summary>
		/// Following a call to cufftCreate() makes a 2D FFT plan configuration according to specified signal sizes and data type.
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="nx">The transform size in the x dimension (number of rows)</param>
		/// <param name="ny">The transform size in the y dimension (number of columns)</param>
		/// <param name="type">The transform data type (e.g., CUFFT_C2R for single precision complex to real)</param>
		/// <param name="workSize"></param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftMakePlan2d(cufftHandle plan, 
                                     int nx, int ny,
                                     cufftType type,
									 ref SizeT workSize);

		/// <summary>
		/// Following a call to cufftCreate() makes a 3D FFT plan configuration according to
		/// specified signal sizes and data type. This function is the same as cufftPlan2d() except
		/// that it takes a third size parameter nz.
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="nx">The transform size in the x dimension</param>
		/// <param name="ny">The transform size in the y dimension</param>
		/// <param name="nz">The transform size in the z dimension</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single precision real to complex)</param>
		/// <param name="workSize"></param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftMakePlan3d(cufftHandle plan, 
                                     int nx, int ny, int nz, 
                                     cufftType type,
									 ref SizeT workSize);

		/// <summary>
		/// Following a call to cufftCreate() makes a FFT plan configuration of dimension rank,
		/// with sizes specified in the array n. The batch input parameter tells CUFFT how many
		/// transforms to configure. With this function, batched plans of 1, 2, or 3 dimensions may
		/// be created.<para/>
		/// The cufftPlanMany() API supports more complicated input and output data layouts
		/// via the advanced data layout parameters: inembed, istride, idist, onembed,
		/// ostride, and odist.<para/>
		/// All arrays are assumed to be in CPU memory.
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">Array of size rank, describing the size of each dimension</param>
		/// <param name="inembed">Pointer of size rank that indicates the storage dimensions of
		/// the input data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="istride">Indicates the distance between two successive input
		/// elements in the least significant (i.e., innermost) dimension</param>
		/// <param name="idist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the input data</param>
		/// <param name="onembed">Pointer of size rank that indicates the storage dimensions of
		/// the output data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="ostride">Indicates the distance between two successive output
		/// elements in the output array in the least significant (i.e.,
		/// innermost) dimension</param>
		/// <param name="odist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the output data</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single
		/// precision real to complex)</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="workSize"></param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftMakePlanMany(cufftHandle plan,
                                       int rank,
                                       [In] int[] n, [In] int[] inembed, [In] int istride, [In] int idist,
									   [In] int[] onembed, [In] int ostride, [In] int odist,
                                       cufftType type,
                                       int batch,
									   ref SizeT workSize);


		/// <summary>
		/// Following a call to cufftCreate() makes a FFT plan configuration of dimension rank,
		/// with sizes specified in the array n. The batch input parameter tells CUFFT how many
		/// transforms to configure. With this function, batched plans of 1, 2, or 3 dimensions may
		/// be created.<para/>
		/// The cufftPlanMany() API supports more complicated input and output data layouts
		/// via the advanced data layout parameters: inembed, istride, idist, onembed,
		/// ostride, and odist.<para/>
		/// All arrays are assumed to be in CPU memory.
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">Array of size rank, describing the size of each dimension</param>
		/// <param name="inembed">Pointer of size rank that indicates the storage dimensions of
		/// the input data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="istride">Indicates the distance between two successive input
		/// elements in the least significant (i.e., innermost) dimension</param>
		/// <param name="idist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the input data</param>
		/// <param name="onembed">Pointer of size rank that indicates the storage dimensions of
		/// the output data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="ostride">Indicates the distance between two successive output
		/// elements in the output array in the least significant (i.e.,
		/// innermost) dimension</param>
		/// <param name="odist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the output data</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single
		/// precision real to complex)</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="workSize"></param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftMakePlanMany64(cufftHandle plan, 
                                         int rank, 
                                         [In] long[] n, [In] long[] inembed, [In] long istride, [In] long idist,
										 [In] long[] onembed, [In] long ostride, [In] long odist,
										 cufftType type,
										 long batch,
										 ref SizeT workSize);





                     
		/// <summary>
		/// During plan execution, CUFFT requires a work area for temporary storage of
		/// intermediate results. This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings. Note that changing
		/// some plan settings, such as compatibility mode, may alter the size required for the work
		/// area.
		/// </summary>
		/// <param name="nx">The transform size (e.g. 256 for a 256-point FFT)</param>
		/// <param name="type">The transform data type (e.g., CUFFT_C2C for single
		/// precision complex to complex)</param>
		/// <param name="batch">Number of transforms of size nx</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftEstimate1d(int nx, 
                                     cufftType type, 
                                     int batch, /* deprecated - use cufftPlanMany */
									 ref SizeT workSize);

		/// <summary>
		/// During plan execution, CUFFT requires a work area for temporary storage of
		/// intermediate results. This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings. Note that changing
		/// some plan settings, such as compatibility mode, may alter the size required for the workarea.
		/// </summary>
		/// <param name="nx">The transform size in the x dimension (number of rows)</param>
		/// <param name="ny">The transform size in the y dimension (number of columns)</param>
		/// <param name="type">The transform data type (e.g., CUFFT_C2R for single
		/// precision complex to real)</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftEstimate2d(int nx, int ny,
                                     cufftType type,
									 ref SizeT workSize);

		/// <summary>
		/// During plan execution, CUFFT requires a work area for temporary storage of
		/// intermediate results. This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings. Note that changing
		/// some plan settings, such as compatibility mode, may alter the size required for the workarea.
		/// </summary>
		/// <param name="nx">The transform size in the x dimension</param>
		/// <param name="ny">The transform size in the y dimension</param>
		/// <param name="nz">The transform size in the z dimension</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single precision real to complex)</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftEstimate3d(int nx, int ny, int nz, 
                                     cufftType type,
									 ref SizeT workSize);

		/// <summary>
		/// During plan execution, CUFFT requires a work area for temporary storage of
		/// intermediate results. This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings. Note that changing
		/// some plan settings, such as compatibility mode, may alter the size required for the work
		/// area.<para/>
		/// The cufftPlanMany() API supports more complicated input and output data layouts
		/// via the advanced data layout parameters: inembed, istride, idist, onembed,
		/// ostride, and odist.<para/>
		/// All arrays are assumed to be in CPU memory.
		/// </summary>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">Array of size rank, describing the size of each dimension</param>
		/// <param name="inembed">Pointer of size rank that indicates the storage dimensions of
		/// the input data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="istride">Indicates the distance between two successive input
		/// elements in the least significant (i.e., innermost) dimension</param>
		/// <param name="idist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the input data</param>
		/// <param name="onembed">Pointer of size rank that indicates the storage dimensions of
		/// the output data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="ostride">Indicates the distance between two successive output
		/// elements in the output array in the least significant (i.e.,
		/// innermost) dimension</param>
		/// <param name="odist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the output data</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single
		/// precision real to complex)</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftEstimateMany(int rank,
									   [In] int[] n, [In] int[] inembed, [In] int istride, [In] int idist,
									   [In] int[] onembed, [In] int ostride, [In] int odist,
                                       cufftType type,
                                       int batch,
									   ref SizeT workSize);
                     
		/// <summary>
		/// Creates only an opaque handle, and allocates small data structures on the host. The
		/// cufftMakePlan*() calls actually do the plan generation. It is recommended that
		/// cufftSet*() calls, such as cufftSetCompatibilityMode(), that may require a plan
		/// to be broken down and re-generated, should be made after cufftCreate() and before
		/// one of the cufftMakePlan*() calls.
		/// </summary>
		/// <param name="cufftHandle">Pointer to a cufftHandle object</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.Success"/>, <see cref="cufftResult.AllocFailed"/>, 
		/// <see cref="cufftResult.InvalidPlan"/>, <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.InternalError"/>, 
		/// <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidSize"/>, 
		/// </returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftCreate(ref cufftHandle cufftHandle);

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than
		/// cufftEstimate1d(), given the specified parameters, and taking into account any plan
		/// settings that may have been made.
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="nx">The transform size (e.g. 256 for a 256-point FFT)</param>
		/// <param name="type">The transform data type (e.g., CUFFT_C2C for single
		/// precision complex to complex)</param>
		/// <param name="batch">Number of transforms of size nx</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftGetSize1d(cufftHandle handle, 
                                    int nx, 
                                    cufftType type, 
                                    int batch,
									ref SizeT workSize);

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than
		/// cufftEstimate2d(), given the specified parameters, and taking into account any plan
		/// settings that may have been made.
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="nx">The transform size in the x dimension (number of rows)</param>
		/// <param name="ny">The transform size in the y dimension (number of columns)</param>
		/// <param name="type">The transform data type (e.g., CUFFT_C2R for single
		/// precision complex to real)</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftGetSize2d(cufftHandle handle, 
                                    int nx, int ny,
                                    cufftType type,
									ref SizeT workSize);

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than
		/// cufftEstimate3d(), given the specified parameters, and taking into account any plan
		/// settings that may have been made.
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="nx">The transform size in the x dimension</param>
		/// <param name="ny">The transform size in the y dimension</param>
		/// <param name="nz">The transform size in the z dimension</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single precision real to complex)</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftGetSize3d(cufftHandle handle,
                                    int nx, int ny, int nz, 
                                    cufftType type,
									ref SizeT workSize);

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than
		/// cufftEstimateSizeMany(), given the specified parameters, and taking into account
		/// any plan settings that may have been made.<para/>
		/// The batch input parameter tells CUFFT how many transforms to configure. With this
		/// function, batched plans of 1, 2, or 3 dimensions may be created.<para/>
		/// The cufftPlanMany() API supports more complicated input and output data layouts
		/// via the advanced data layout parameters: inembed, istride, idist, onembed,
		/// ostride, and odist.<para/>
		/// All arrays are assumed to be in CPU memory
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">Array of size rank, describing the size of each dimension</param>
		/// <param name="inembed">Pointer of size rank that indicates the storage dimensions of
		/// the input data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="istride">Indicates the distance between two successive input
		/// elements in the least significant (i.e., innermost) dimension</param>
		/// <param name="idist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the input data</param>
		/// <param name="onembed">Pointer of size rank that indicates the storage dimensions of
		/// the output data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="ostride">Indicates the distance between two successive output
		/// elements in the output array in the least significant (i.e.,
		/// innermost) dimension</param>
		/// <param name="odist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the output data</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single
		/// precision real to complex)</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="workArea">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftGetSizeMany(cufftHandle handle,
									  int rank, [In] int[] n, [In] int[] inembed, [In] int istride, [In] int idist,
								      [In] int[] onembed, [In] int ostride, [In] int odist,
									  cufftType type, int batch, ref SizeT workArea);



		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than
		/// cufftEstimateSizeMany(), given the specified parameters, and taking into account
		/// any plan settings that may have been made.<para/>
		/// The batch input parameter tells CUFFT how many transforms to configure. With this
		/// function, batched plans of 1, 2, or 3 dimensions may be created.<para/>
		/// The cufftPlanMany() API supports more complicated input and output data layouts
		/// via the advanced data layout parameters: inembed, istride, idist, onembed,
		/// ostride, and odist.<para/>
		/// All arrays are assumed to be in CPU memory
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="rank">Dimensionality of the transform (1, 2, or 3)</param>
		/// <param name="n">Array of size rank, describing the size of each dimension</param>
		/// <param name="inembed">Pointer of size rank that indicates the storage dimensions of
		/// the input data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="istride">Indicates the distance between two successive input
		/// elements in the least significant (i.e., innermost) dimension</param>
		/// <param name="idist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the input data</param>
		/// <param name="onembed">Pointer of size rank that indicates the storage dimensions of
		/// the output data in memory. If set to NULL all other advanced
		/// data layout parameters are ignored.</param>
		/// <param name="ostride">Indicates the distance between two successive output
		/// elements in the output array in the least significant (i.e.,
		/// innermost) dimension</param>
		/// <param name="odist">Indicates the distance between the first element of two
		/// consecutive signals in a batch of the output data</param>
		/// <param name="type">The transform data type (e.g., CUFFT_R2C for single
		/// precision real to complex)</param>
		/// <param name="batch">Batch size for this transform</param>
		/// <param name="workArea">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftGetSizeMany64(cufftHandle plan,
									   long rank, [In] long[] n, [In] long[] inembed, [In] long istride, [In] long idist,
									   [In] long[] onembed, [In] long ostride, [In] long odist,
									   cufftType type, long batch, ref SizeT workArea);
                     
		/// <summary>
		/// Once plan generation has been done, either with the original API or the extensible API,
		/// this call returns the actual size of the work area required to support the plan. Callers
		/// who choose to manage work area allocation within their application must use this call
		/// after plan generation, and after any cufftSet*() calls subsequent to plan generation, if
		/// those calls might alter the required work space size.
		/// </summary>
		/// <param name="handle">cufftHandle object</param>
		/// <param name="workSize">Pointer to the size of the work space</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftGetSize(cufftHandle handle, ref SizeT workSize);
           
		/// <summary>
		/// cufftSetWorkArea() overrides the work area pointer associated with a plan.
		/// If the work area was auto-allocated, CUFFT frees the auto-allocated space. The
		/// cufftExecute*() calls assume that the work area pointer is valid and that it points to
		/// a contiguous region in device memory that does not overlap with any other work area. If
		/// this is not the case, results are indeterminate.
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="workArea">Pointer to workArea</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftSetWorkArea(cufftHandle plan, CUdeviceptr workArea);

		/// <summary>
		/// cufftSetAutoAllocation() indicates that the caller intends to allocate and manage
		/// work areas for plans that have been generated. CUFFT default behavior is to allocate
		/// the work area at plan generation time. If cufftSetAutoAllocation() has been called
		/// with autoAllocate set to "false" prior to one of the cufftMakePlan*() calls, CUFFT
		/// does not allocate the work area. This is the preferred sequence for callers wishing to
		/// manage work area allocation.
		/// </summary>
		/// <param name="plan">cufftHandle object</param>
		/// <param name="autoAllocate">Boolean to indicate whether to allocate work area.</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);




		/// <summary>
		/// Frees all GPU resources associated with a CUFFT plan and destroys the
		/// internal plan data structure. This function should be called once a plan
		/// is no longer needed to avoid wasting GPU memory.
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <returns></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftDestroy([In] cufftHandle plan);

		/// <summary>
		/// Executes a CUFFT single‐precision complex‐to‐complex transform
		/// plan as specified by direction. CUFFT uses as input data the GPU
		/// memory pointed to by the idata parameter. This function stores the
		/// Fourier coefficients in the odata array. If idata and odata are the
		/// same, this method does an in‐place transform.
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="idata">cuFloatComplex: Pointer to the single-precision complex input data (in GPU memory) to transform</param>
		/// <param name="odata">cuFloatComplex: Pointer to the single-precision complex output data (in GPU memory)</param>
		/// <param name="direction">The transform direction: Forward or Inverse</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InternalError"/>, 
		/// <see cref="cufftResult.InvalidPlan"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftExecC2C([In] cufftHandle plan, [In] CUdeviceptr idata, [Out] CUdeviceptr odata, [In] TransformDirection direction);

		/// <summary>
		/// executes a CUFFT single‐precision real‐to‐complex (implicitly
		/// forward) transform plan. CUFFT uses as input data the GPU memory
		/// pointed to by the <c>idata</c> parameter. This function stores the nonredundant
		/// Fourier coefficients in the <c>odata</c> array. If <c>idata</c> and <c>odata</c>
		/// are the same, this method does an in‐place transform (See “CUFFT
		/// Transform Types” on page 4 for details on real data FFTs.)
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="idata">cuFloatReal: Pointer to the single-precision real input data (in GPU memory) to transform</param>
		/// <param name="odata">cuFloatComplex: Pointer to the single-precision complex output data (in GPU memory)</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidPlan"/>, 
		/// <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.ExecFailed"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftExecR2C([In] cufftHandle plan, [In] CUdeviceptr idata, [Out] CUdeviceptr odata);

		/// <summary>
		/// Executes a CUFFT single‐precision complex‐to‐real (implicitly inverse)
		/// transform plan. CUFFT uses as input data the GPU memory pointed to
		/// by the idata parameter. The input array holds only the nonredundant
		/// complex Fourier coefficients. This function stores the real
		/// output values in the odata array. If idata and odata are the same, this
		/// method does an in‐place transform. (See “CUFFT Transform Types”
		/// on page 4 for details on real data FFTs.)
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="idata">cuFloatComplex: Pointer to the single-precision complex input data (in GPU memory) to transform</param>
		/// <param name="odata">cuFloatReal: Pointer to the single-precision real output data (in GPU memory)</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidPlan"/>, 
		/// <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.ExecFailed"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftExecC2R([In] cufftHandle plan, [In]  CUdeviceptr idata, [Out] CUdeviceptr odata);

		/// <summary>
		/// Executes a CUFFT double‐precision complex‐to‐complex transform
		/// plan as specified by direction. CUFFT uses as input data the GPU
		/// memory pointed to by the <c>idata</c> parameter. This function stores the
		/// Fourier coefficients in the <c>odata</c> array. If <c>idata</c> and <c>odata</c> are the
		/// same, this method does an in‐place transform.
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="idata">cuDoubleComplex: Pointer to the double-precision complex input data (in GPU memory) to transform</param>
		/// <param name="odata">cuDoubleComplex: Pointer to the double-precision complex output data (in GPU memory)</param>
		/// <param name="direction">The transform direction: Forward or Inverse</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidPlan"/>, 
		/// <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.ExecFailed"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftExecZ2Z([In] cufftHandle plan, [In] CUdeviceptr idata, [Out] CUdeviceptr odata, [In] TransformDirection direction);

		/// <summary>
		/// Executes a CUFFT double‐precision real‐to‐complex (implicitly
		/// forward) transform plan. CUFFT uses as input data the GPU memory
		/// pointed to by the <c>idata</c> parameter. This function stores the nonredundant
		/// Fourier coefficients in the <c>odata</c> array. If <c>idata</c> and <c>odata</c>
		/// are the same, this method does an in‐place transform (See “CUFFT
		/// Transform Types” on page 4 for details on real data FFTs.)
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="idata">cuDoubleReal: Pointer to the double-precision real input data (in GPU memory) to transform</param>
		/// <param name="odata">cuDoubleComplex: Pointer to the double-precision complex output data (in GPU memory)</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidPlan"/>, 
		/// <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.ExecFailed"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftExecD2Z([In] cufftHandle plan, [In] CUdeviceptr idata, [Out] CUdeviceptr odata);

		/// <summary>
		/// Executes a CUFFT double‐precision complex‐to‐real (implicitly
		/// inverse) transform plan. CUFFT uses as input data the GPU memory
		/// pointed to by the <c>idata</c> parameter. The input array holds only the
		/// non‐redundant complex Fourier coefficients. This function stores the
		/// real output values in the <c>odata</c> array. If <c>idata</c> and <c>odata</c> are the same,
		/// this method does an in‐place transform. (See “CUFFT Transform
		/// Types” on page 4 for details on real data FFTs.)
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="idata">cuDoubleComplex: Pointer to the double-precision complex input data (in GPU memory) to transform</param>
		/// <param name="odata">cuDoubleReal: Pointer to the double-precision real output data (in GPU memory)</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidPlan"/>, 
		/// <see cref="cufftResult.InvalidValue"/>, <see cref="cufftResult.ExecFailed"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftExecZ2D([In] cufftHandle plan, [In] CUdeviceptr idata, [In] CUdeviceptr odata);

		/// <summary>
		/// Associates a CUDA stream with a CUFFT plan. All kernel launches
		/// made during plan execution are now done through the associated
		/// stream, enabling overlap with activity in other streams (for example,
		/// data copying). The association remains until the plan is destroyed or
		/// the stream is changed with another call to cufftSetStream().
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="stream">A valid CUDA stream created with cudaStreamCreate() (or 0 for the default stream)</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.InvalidPlan"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftSetStream([In] cufftHandle plan, [In] CUstream stream);

		/// <summary>
		/// configures the layout of CUFFT output in FFTW‐compatible modes.
		/// When FFTW compatibility is desired, it can be configured for padding
		/// only, for asymmetric complex inputs only, or to be fully compatible.
		/// </summary>
		/// <param name="plan">The <see cref="cufftHandle"/> object of the plan to be destroyed.</param>
		/// <param name="mode">The <see cref="Compatibility"/> option to be used</param>
		/// <returns>cufftResult Error Codes: <see cref="cufftResult.SetupFailed"/>, <see cref="cufftResult.InvalidPlan"/>, <see cref="cufftResult.Success"/></returns>
		[DllImport(CUFFT_API_DLL_NAME)]
		public static extern cufftResult cufftSetCompatibilityMode([In] cufftHandle plan, [In] Compatibility mode);

	}
}
