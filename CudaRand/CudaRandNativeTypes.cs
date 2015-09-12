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
using System.Globalization;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.CudaRand
{
	#region Enums
	/// <summary>
	/// CURAND function call status types
	/// </summary>
	public enum CurandStatus
	{
		/// <summary>
		/// No errors.
		/// </summary>
		Success = 0,
		/// <summary>
		/// Header file and linked library version do not match.
		/// </summary>
		VersionMismatch = 100,
		/// <summary>
		/// Generator not initialized.
		/// </summary>
		NotInitialized = 101,
		/// <summary>
		/// Memory allocation failed.
		/// </summary>
		AllocationFailed = 102,
		/// <summary>
		/// Generator is wrong type.
		/// </summary>
		TypeError = 103,
		/// <summary>
		/// Argument out of range.
		/// </summary>
		OutOfRange = 104,
		/// <summary>
		/// Length requested is not a multple of dimension.
		/// </summary>
		LengthNotMultiple = 105,
		/// <summary>
		/// GPU does not have double precision required by MRG32k3a.
		/// </summary>
		DoublePrecisionRequired = 106,
		/// <summary>
		/// Kernel launch failure.
		/// </summary>
		LaunchFailure = 201,
		/// <summary>
		/// Preexisting failure on library entry.
		/// </summary>
		PreexistingFailure = 202,
		/// <summary>
		/// Initialization of CUDA failed.
		/// </summary>
		InitializationFailed = 203,
		/// <summary>
		/// Architecture mismatch, GPU does not support requested feature.
		/// </summary>
		ArchMismatch = 204,
		/// <summary>
		/// Internal library error.
		/// </summary>
		InternalError = 999
	}

	/// <summary>
	/// CURAND generator types
	/// </summary>
	public enum GeneratorType
	{
		/// <summary>
		/// 
		/// </summary>
		Test = 0,
		/// <summary>
		/// Default pseudorandom generator.
		/// </summary>
		PseudoDefault = 100,
		/// <summary>
		/// XORWOW pseudorandom generator.
		/// </summary>
		PseudoXORWOW = 101,
		/// <summary>
		/// MRG32k3a pseudorandom generator.
		/// </summary>
		PseudoMRG32K3A = 121,
		/// <summary>
		/// Mersenne Twister pseudorandom generator.
		/// </summary>
		PseudoMTGP32 = 141,
		/// <summary>
		/// Mersenne Twister MT19937 pseudorandom generator.
		/// </summary>
		PseudoMT19937 = 142,
		/// <summary>
		/// PseudoPhilox4_32_10 quasirandom generator.
		/// </summary>
		PseudoPhilox4_32_10 = 161,
		/// <summary>
		/// Default quasirandom generator.
		/// </summary>
		QuasiDefault = 200,
		/// <summary>
		/// Sobol32 quasirandom generator.
		/// </summary>
		QuasiSobol32 = 201,
		/// <summary>
		/// Scrambled Sobol32 quasirandom generator.
		/// </summary>
		QuasiScrambledSobol32 = 202,
		/// <summary>
		/// Sobol64 quasirandom generator.
		/// </summary>
		QuasiSobol64 = 203,
		/// <summary>
		/// Scrambled Sobol64 quasirandom generator.
		/// </summary>
		QuasiScrambledSobol64 = 204
	}

	/// <summary>
	/// CURAND orderings of results in memory
	/// </summary>
	public enum Ordering
	{
		/// <summary>
		/// Best ordering for pseudorandom results.
		/// </summary>
		PseudoBest = 100,
		/// <summary>
		/// Specific default 4096 thread sequence for pseudorandom results.
		/// </summary>
		PseudoDefault = 101,
		/// <summary>
		/// Specific seeding pattern for fast lower quality pseudorandom results.
		/// </summary>
		PseudoSeeded = 102,
		/// <summary>
		/// Specific n-dimensional ordering for quasirandom results.
		/// </summary>
		QuasiDefault = 201
	}

	/// <summary>
	/// CURAND choice of direction vector set
	/// </summary>
	public enum DirectionVectorSet
	{
		/// <summary>
		/// Specific set of 32-bit direction vectors generated from polynomials 
		/// recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions.
		/// </summary>
		JoeKuo6_32 = 101,
		/// <summary>
		/// Specific set of 32-bit direction vectors generated from polynomials 
		/// recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled.
		/// </summary>
		ScrambledJoeKuo6_32 = 102,
		/// <summary>
		/// Specific set of 64-bit direction vectors generated from polynomials 
		/// recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions.
		/// </summary>
		JoeKuo6_64 = 103,
		/// <summary>
		/// Specific set of 64-bit direction vectors generated from polynomials 
		/// recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled.
		/// </summary>
		ScrambledJoeKuo6_64 = 104
	}

	/// <summary>
	/// CURAND method
	/// </summary>
	public enum curandMethod
	{
		/// <summary>
		/// choose best depends on args
		/// </summary>
		ChooseBest = 0, 
		/// <summary></summary>
		ITR = 1,
		/// <summary></summary>
		Knuth = 2,
		/// <summary></summary>
		HITR = 3,
		/// <summary></summary>
		M1 = 4,
		/// <summary></summary>
		M2 = 5,
		/// <summary></summary>
		BinarySearch = 6,
		/// <summary></summary>
		DiscreteGauss = 7,
		/// <summary></summary>
		Rejection = 8,
		/// <summary></summary>
		DeviceAPI = 9,
		/// <summary></summary>
		FastRejection = 10,
		/// <summary></summary>
		Third = 11,
		/// <summary></summary>
		Definition = 12,
		/// <summary></summary>
		Poisson = 13,
	};

	#endregion

	#region Structs
	/// <summary>
	/// CURAND generator
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CurandGenerator
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	/// <summary>
	/// Array of 32-bit direction vectors
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct DirectionVectors32
	{
		/// <summary>
		/// Inner data array
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
		public uint[] Array;
	}

	/// <summary>
	/// Array of 64-bit direction vectors
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct DirectionVectors64
	{
		/// <summary>
		/// Inner data array
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
		public ulong[] Array;
	}

	/// <summary>
	/// Discrete Distribution
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct DiscreteDistribution
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}
	#endregion
	

}
