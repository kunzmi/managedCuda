//	Copyright (c) 2014, Michael Kunz. All rights reserved.
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
using System.Linq;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ManagedCuda
{
	/// <summary>
	/// Cuda occupancy from CudaOccupancy.h
	/// </summary>
	public class CudaOccupancy
	{
		const int MIN_SHARED_MEM_PER_SM = 16384;
		const int MIN_SHARED_MEM_PER_SM_GK210 = 81920;

		/// <summary>
		/// mirror the type and spelling of cudaDeviceProp's members keep these alphabetized
		/// </summary>
		public class cudaOccDeviceProp {
			/// <summary/>
			public int major;
			/// <summary/>
			public int minor;
			/// <summary/>
			public int maxThreadsPerBlock;
			/// <summary/>
			public int maxThreadsPerMultiProcessor;
			/// <summary/>
			public int regsPerBlock;
			/// <summary/>
			public int regsPerMultiprocessor;
			/// <summary/>
			public int warpSize;
			/// <summary/>
			public SizeT sharedMemPerBlock;
			/// <summary/>
			public SizeT sharedMemPerMultiprocessor;

			/// <summary/>
			public cudaOccDeviceProp()
			{

			}

			/// <summary/>
			public cudaOccDeviceProp(int deviceID)
				: this(CudaContext.GetDeviceInfo(deviceID))
			{

			}

			/// <summary/>
			public cudaOccDeviceProp(CudaDeviceProperties props)
			{
				major = props.ComputeCapabilityMajor;
				minor = props.ComputeCapabilityMinor;
				maxThreadsPerBlock = props.MaxThreadsPerBlock;
				maxThreadsPerMultiProcessor = props.MaxThreadsPerMultiProcessor;
				regsPerBlock = props.RegistersPerBlock;
				regsPerMultiprocessor = props.MaxRegistersPerMultiprocessor;
				warpSize = props.WarpSize;
				sharedMemPerBlock = props.SharedMemoryPerBlock;
				sharedMemPerMultiprocessor = props.MaxSharedMemoryPerMultiprocessor;
			}
		}

		
		/// <summary>
		/// define our own cudaOccFuncAttributes to stay consistent with the original header file
		/// </summary>
		public class cudaOccFuncAttributes
		{
			/// <summary/>
			public int maxThreadsPerBlock;
			/// <summary/>
			public int numRegs;
			/// <summary/>
			public SizeT sharedSizeBytes;

			/// <summary>
			/// 
			/// </summary>
			public cudaOccFuncAttributes()
			{ 
			
			}

			/// <summary>
			/// cudaOccFuncAttributes
			/// </summary>
			/// <param name="aMaxThreadsPerBlock"></param>
			/// <param name="aNumRegs"></param>
			/// <param name="aSharedSizeBytes">Only the static part shared memory (without dynamic allocations)</param>
			public cudaOccFuncAttributes(int aMaxThreadsPerBlock, int aNumRegs, SizeT aSharedSizeBytes)
			{
				maxThreadsPerBlock = aMaxThreadsPerBlock;
				numRegs = aNumRegs;
				sharedSizeBytes = aSharedSizeBytes;
			}

			/// <summary>
			/// 
			/// </summary>
			/// <param name="aKernel"></param>
			public cudaOccFuncAttributes(CudaKernel aKernel)
				: this(aKernel.MaxThreadsPerBlock, aKernel.Registers, aKernel.SharedMemory)
			{ 
			
			}
		}

		/// <summary>
		/// Occupancy Error types
		/// </summary>
		public enum cudaOccError
		{
			/// <summary/>
			None = 0,
			/// <summary>
			/// input parameter is invalid
			/// </summary>
			ErrorInvalidInput = -1,
			/// <summary>
			/// requested device is not supported in current implementation or device is invalid
			/// </summary>
			ErrorUnknownDevice  = -2, 
		}

		
		/// <summary>
		/// Function cache configurations
		/// </summary>
		public enum cudaOccCacheConfig
		{
			/// <summary>
			/// no preference for shared memory or L1 (default) 
			/// </summary>
			PreferNone    = 0x00, 
			/// <summary>
			/// prefer larger shared memory and smaller L1 cache
			/// </summary>
			PreferShared  = 0x01, 
			/// <summary>
			/// prefer larger L1 cache and smaller shared memory
			/// </summary>
			PreferL1      = 0x02,
			/// <summary>
			/// prefer equal sized L1 cache and shared memory
			/// </summary>
			PreferEqual   = 0x03,
		}


		/// <summary>
		/// Occupancy Limiting Factors 
		/// </summary>
		[Flags]
		public enum cudaOccLimitingFactors
		{
			/// <summary>
			/// occupancy limited due to warps available
			/// </summary>
			Warps = 0x01, 
			/// <summary>
			/// occupancy limited due to registers available
			/// </summary>
			Registers = 0x02, 
			/// <summary>
			/// occupancy limited due to shared memory available
			/// </summary>
			SharedMemory = 0x04, 
			/// <summary>
			/// occupancy limited due to blocks available
			/// </summary>
			Blocks         = 0x08 
		};

		/// <summary>
		/// 
		/// </summary>
		public class cudaOccResult
		{
			/// <summary>
			/// Active Thread Blocks per Multiprocessor
			/// </summary>
			public int ActiveBlocksPerMultiProcessor;
			/// <summary>
			/// Active Warps per Multiprocessor
			/// </summary>
			public int ActiveWarpsPerMultiProcessor;
			/// <summary>
			/// Occupancy of each Multiprocessor
			/// </summary>
			public int OccupancyOfEachMultiProcessor;
			/// <summary>
			/// Active Threads per Multiprocessor
			/// </summary>
			public int ActiceThreadsPerMultiProcessor;
			/// <summary/>
			public cudaOccLimitingFactors LimitingFactors;
			/// <summary/>
			public int BlockLimitRegs;
			/// <summary/>
			public int BlockLimitSharedMem;
			/// <summary/>
			public int BlockLimitWarps;
			/// <summary/>
			public int BlockLimitBlocks;
			/// <summary/>
			public int BllocatedRegistersPerBlock;
			/// <summary/>
			public int AllocatedSharedMemPerBlock;
		}

		/// <summary>
		/// define cudaOccDeviceState to include any device property needed to be passed
		/// in future GPUs so that user interfaces don't change ; hence users are encouraged
		/// to declare the struct zero in order to handle the assignments of any field
		/// that might be added for later GPUs.
		/// </summary>
		public struct cudaOccDeviceState
		{
			/// <summary/>
			public cudaOccCacheConfig cacheConfig;
		}

		// get the minimum out of two parameters
		private static int min_(int lhs, int rhs)
		{
			return rhs < lhs ? rhs : lhs;
		}
		
		// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
		private static int divide_ri(int x, int y)
		{
			return (x + (y - 1)) / y;
		}

		// round x towards infinity to the next multiple of y
		private static int round_i(int x, int y)
		{
			return y * divide_ri(x, y);
		}

		
		//////////////////////////////////////////
		//    Occupancy Helper Functions        //
		//////////////////////////////////////////

		/*!
		 * Granularity of shared memory allocation
		 */
		private static int cudaOccSMemAllocationUnit(cudaOccDeviceProp properties)
		{
			switch(properties.major)
			{
				case 1:  return 512;
				case 2:  return 128;
				case 3:
				case 5:  return 256;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}


		/*!
		 * Granularity of register allocation
		 */
		private static int cudaOccRegAllocationUnit(cudaOccDeviceProp properties, int regsPerThread)
		{
			switch(properties.major)
			{
				case 1:  return (properties.minor <= 1) ? 256 : 512;
				case 2:  switch(regsPerThread)
						 {
							case 21:
							case 22:
							case 29:
							case 30:
							case 37:
							case 38:
							case 45:
							case 46:
								return 128;
							default:
								return 64;
						 }
				case 3:
				case 5:  return 256;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}


		/*!
		 * Granularity of warp allocation
		 */
		private static int cudaOccWarpAllocationMultiple(cudaOccDeviceProp properties)
		{
			return (properties.major <= 1) ? 2 : 1;
		}

		/*!
		 * Number of "sides" into which the multiprocessor is partitioned
		 */
		private static int cudaOccSidesPerMultiprocessor(cudaOccDeviceProp properties)
		{
			switch(properties.major)
			{
				case 1:  return 1;
				case 2:  return 2;
				case 3:  return 4;
				case 5:  return 4;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}

		/*!
		 * Maximum blocks that can run simultaneously on a multiprocessor
		 */
		private static int cudaOccMaxBlocksPerMultiprocessor(cudaOccDeviceProp properties)
		{
			switch(properties.major)
			{
				case 1:  return 8;
				case 2:  return 8;
				case 3:  return 16;
				case 5:  return 32;
				default: throw new CudaOccupancyException(cudaOccError.ErrorUnknownDevice);
			}
		}

		///*!
		// * Map int to cudaOccCacheConfig
		// */
		//private static cudaOccCacheConfig cudaOccGetCacheConfig(cudaOccDeviceState state)
		//{
		//    switch(state.cacheConfig)
		//    {
		//        case 0:  return cudaOccCacheConfig.PreferNone;
		//        case 1:  return cudaOccCacheConfig.PreferShared;
		//        case 2:  return cudaOccCacheConfig.PreferL1;
		//        case 3:  return cudaOccCacheConfig.PreferEqual;
		//        default: return cudaOccCacheConfig.PreferNone;
		//    }
		//}

		/*!
		 * Shared memory based on config requested by User
		 */
		private static int cudaOccSMemPerMultiprocessor(cudaOccDeviceProp properties, cudaOccCacheConfig cacheConfig)
		{
			int bytes = 0;
			int sharedMemPerMultiprocessorHigh = (int) properties.sharedMemPerMultiprocessor;
			int sharedMemPerMultiprocessorLow  = (properties.major==3 && properties.minor==7)
				? MIN_SHARED_MEM_PER_SM_GK210
				: MIN_SHARED_MEM_PER_SM ;

			switch(properties.major)
			{
				case 1:
				case 2: bytes = (cacheConfig == cudaOccCacheConfig.PreferL1)? sharedMemPerMultiprocessorLow : sharedMemPerMultiprocessorHigh;
						break;
				case 3: switch (cacheConfig)
						{
							default :
							case cudaOccCacheConfig.PreferNone:
							case cudaOccCacheConfig.PreferShared:
									bytes = sharedMemPerMultiprocessorHigh;
									break;
							case cudaOccCacheConfig.PreferL1:
									bytes = sharedMemPerMultiprocessorLow;
									break;
							case cudaOccCacheConfig.PreferEqual:
									bytes = (sharedMemPerMultiprocessorHigh + sharedMemPerMultiprocessorLow) / 2;
									break;
						}
						break;
				case 5:
				default: bytes = sharedMemPerMultiprocessorHigh;
						 break;
			}

			return bytes;
		}


		
		///////////////////////////////////////////////
		//    Occupancy calculation Functions        //
		///////////////////////////////////////////////

		/// <summary>
		/// Determine the maximum number of CTAs that can be run simultaneously per SM.<para/>
		/// This is equivalent to the calculation done in the CUDA Occupancy Calculator
		/// spreadsheet
		/// </summary>
		/// <param name="properties"></param>
		/// <param name="kernel"></param>
		/// <param name="state"></param>
		/// <returns></returns>
		public static cudaOccResult cudaOccMaxActiveBlocksPerMultiprocessor(
			CudaDeviceProperties properties,
			CudaKernel kernel,
			cudaOccDeviceState state)
		{
			cudaOccDeviceProp props = new cudaOccDeviceProp(properties);
			cudaOccFuncAttributes attributes = new cudaOccFuncAttributes(kernel);

			return cudaOccMaxActiveBlocksPerMultiprocessor(props, attributes, (int)kernel.BlockDimensions.x * (int)kernel.BlockDimensions.y * (int)kernel.BlockDimensions.z, kernel.DynamicSharedMemory, state);			
		}

		/// <summary>
		/// Determine the maximum number of CTAs that can be run simultaneously per SM.<para/>
		/// This is equivalent to the calculation done in the CUDA Occupancy Calculator
		/// spreadsheet
		/// </summary>
		/// <param name="properties"></param>
		/// <param name="attributes"></param>
		/// <param name="blockSize"></param>
		/// <param name="dynamic_smem_bytes"></param>
		/// <param name="state"></param>
		/// <returns></returns>
		public static cudaOccResult cudaOccMaxActiveBlocksPerMultiprocessor(
			cudaOccDeviceProp properties,
			cudaOccFuncAttributes attributes,
			int blockSize,
			SizeT dynamic_smem_bytes,
			cudaOccDeviceState state)
		{
			int regAllocationUnit = 0, warpAllocationMultiple = 0, maxBlocksPerSM=0;
			int ctaLimitWarps = 0, ctaLimitBlocks = 0, smemPerCTA = 0, smemBytes = 0, smemAllocationUnit = 0;
			int cacheConfigSMem = 0, sharedMemPerMultiprocessor = 0, ctaLimitRegs = 0, regsPerCTA=0;
			int regsPerWarp = 0, numSides = 0, numRegsPerSide = 0, ctaLimit=0;
			int maxWarpsPerSm = 0, warpsPerCTA = 0, ctaLimitSMem=0;
			cudaOccLimitingFactors limitingFactors = 0;
			cudaOccResult result = new cudaOccResult();

			if(properties == null || attributes == null || blockSize <= 0)
			{
				throw new CudaOccupancyException(cudaOccError.ErrorInvalidInput);
			}

			//////////////////////////////////////////
			// Limits due to warps/SM or blocks/SM
			//////////////////////////////////////////
			CudaOccupancyException.CheckZero(properties.warpSize);
			maxWarpsPerSm   = properties.maxThreadsPerMultiProcessor / properties.warpSize;
			warpAllocationMultiple = cudaOccWarpAllocationMultiple(properties);

			CudaOccupancyException.CheckZero(warpAllocationMultiple);
			warpsPerCTA = round_i(divide_ri(blockSize, properties.warpSize), warpAllocationMultiple);

			maxBlocksPerSM  = cudaOccMaxBlocksPerMultiprocessor(properties);

			// Calc limits
			CudaOccupancyException.CheckZero(warpsPerCTA);
			ctaLimitWarps  = (blockSize <= properties.maxThreadsPerBlock) ? maxWarpsPerSm / warpsPerCTA : 0;
			ctaLimitBlocks = maxBlocksPerSM;

			//////////////////////////////////////////
			// Limits due to shared memory/SM
			//////////////////////////////////////////
			smemAllocationUnit     = cudaOccSMemAllocationUnit(properties);
			smemBytes  = (int)(attributes.sharedSizeBytes + dynamic_smem_bytes);
			CudaOccupancyException.CheckZero(smemAllocationUnit);
			smemPerCTA = round_i(smemBytes, smemAllocationUnit);

			// Calc limit
			cacheConfigSMem = cudaOccSMemPerMultiprocessor(properties,state.cacheConfig);

			// sharedMemoryPerMultiprocessor is by default limit set in hardware but user requested shared memory
			// limit is used instead if it is greater than total shared memory used by function .
			sharedMemPerMultiprocessor = (cacheConfigSMem >= smemPerCTA)
				? cacheConfigSMem
				: (int)properties.sharedMemPerMultiprocessor;
			// Limit on blocks launched should be calculated with shared memory per SM but total shared memory
			// used by function should be limited by shared memory per block
			ctaLimitSMem = 0;
			if(properties.sharedMemPerBlock >= (SizeT)smemPerCTA)
			{
				ctaLimitSMem = smemPerCTA > 0 ? sharedMemPerMultiprocessor / smemPerCTA : maxBlocksPerSM;
			}

			//////////////////////////////////////////
			// Limits due to registers/SM
			//////////////////////////////////////////
			regAllocationUnit      = cudaOccRegAllocationUnit(properties, attributes.numRegs);
			CudaOccupancyException.CheckZero(regAllocationUnit);

			// Calc limit
			ctaLimitRegs = 0;
			if(properties.major <= 1)
			{
				// GPUs of compute capability 1.x allocate registers to CTAs
				// Number of regs per block is regs per thread times number of warps times warp size, rounded up to allocation unit
				regsPerCTA = round_i(attributes.numRegs * properties.warpSize * warpsPerCTA, regAllocationUnit);
				ctaLimitRegs = regsPerCTA > 0 ? properties.regsPerMultiprocessor / regsPerCTA : maxBlocksPerSM;
			}
			else
			{
				// GPUs of compute capability 2.x and higher allocate registers to warps
				// Number of regs per warp is regs per thread times number of warps times warp size, rounded up to allocation unit
				regsPerWarp = round_i(attributes.numRegs * properties.warpSize, regAllocationUnit);
				regsPerCTA = regsPerWarp * warpsPerCTA;
				if(properties.regsPerBlock >= regsPerCTA)
				{
					numSides = cudaOccSidesPerMultiprocessor(properties);
					CudaOccupancyException.CheckZero(numSides);
					numRegsPerSide = properties.regsPerMultiprocessor / numSides;
					ctaLimitRegs = regsPerWarp > 0 ? ((numRegsPerSide / regsPerWarp) * numSides) / warpsPerCTA : maxBlocksPerSM;
				}
			}

			//////////////////////////////////////////
			// Overall limit is min() of limits due to above reasons
			//////////////////////////////////////////
			ctaLimit = min_(ctaLimitRegs, min_(ctaLimitSMem, min_(ctaLimitWarps, ctaLimitBlocks)));
			// Determine occupancy limiting factors
			
			
			result.ActiveBlocksPerMultiProcessor = ctaLimit;

			if(ctaLimit==ctaLimitWarps)
			{
				limitingFactors |= cudaOccLimitingFactors.Warps;
			}
			if(ctaLimit==ctaLimitRegs && regsPerCTA > 0)
			{
				limitingFactors |= cudaOccLimitingFactors.Registers;
			}
			if(ctaLimit==ctaLimitSMem && smemPerCTA > 0)
			{
				limitingFactors |= cudaOccLimitingFactors.SharedMemory;
			}
			if(ctaLimit==ctaLimitBlocks)
			{
				limitingFactors |= cudaOccLimitingFactors.Blocks;
			}
			result.LimitingFactors = limitingFactors;

			result.BlockLimitRegs = ctaLimitRegs;
			result.BlockLimitSharedMem = ctaLimitSMem;
			result.BlockLimitWarps = ctaLimitWarps;
			result.BlockLimitBlocks = ctaLimitBlocks;

			result.BllocatedRegistersPerBlock = regsPerCTA;
			result.AllocatedSharedMemPerBlock = smemPerCTA;

			result.ActiveWarpsPerMultiProcessor = ctaLimit * ((int)Math.Ceiling(blockSize / (double)properties.warpSize));
			result.ActiceThreadsPerMultiProcessor = result.ActiveWarpsPerMultiProcessor * properties.warpSize;
			result.OccupancyOfEachMultiProcessor = (int)Math.Round(result.ActiveWarpsPerMultiProcessor / (double)maxWarpsPerSm * 100);
			return result;
		}

		/// <summary>
		/// A function to convert from block size to dynamic shared memory size.<para/>
		/// e.g.:
		/// If no dynamic shared memory is used: x => 0<para/>
		/// If 4 bytes shared memory per thread is used: x = 4 * x
		/// </summary>
		/// <param name="aBlockSize">block size</param>
		/// <returns>size of dynamic shared memory</returns>
		public delegate SizeT del_blockSizeToDynamicSMemSize(int aBlockSize);
		

		/// <summary>
		/// Determine the potential block size that allows maximum number of CTAs that can run on multiprocessor simultaneously 
		/// </summary>
		/// <param name="properties"></param>
		/// <param name="kernel"></param>
		/// <param name="state"></param>
		/// <param name="blockSizeToSMem">
		/// A function to convert from block size to dynamic shared memory size.<para/>
		/// e.g.:
		/// If no dynamic shared memory is used: x => 0<para/>
		/// If 4 bytes shared memory per thread is used: x = 4 * x</param>
		/// <returns>maxBlockSize</returns>
		public static int cudaOccMaxPotentialOccupancyBlockSize(
			CudaDeviceProperties properties,
			CudaKernel kernel,
			cudaOccDeviceState state,
			del_blockSizeToDynamicSMemSize blockSizeToSMem)
		{
			cudaOccDeviceProp props = new cudaOccDeviceProp(properties);
			cudaOccFuncAttributes attributes = new cudaOccFuncAttributes(kernel);
			return cudaOccMaxPotentialOccupancyBlockSize(props, attributes, state, blockSizeToSMem);
		}

		/// <summary>
		/// Determine the potential block size that allows maximum number of CTAs that can run on multiprocessor simultaneously 
		/// </summary>
		/// <param name="properties"></param>
		/// <param name="attributes"></param>
		/// <param name="state"></param>
		/// <param name="blockSizeToSMem">
		/// A function to convert from block size to dynamic shared memory size.<para/>
		/// e.g.:
		/// If no dynamic shared memory is used: x => 0<para/>
		/// If 4 bytes shared memory per thread is used: x = 4 * x</param>
		/// <returns>maxBlockSize</returns>
		public static int cudaOccMaxPotentialOccupancyBlockSize(
		    cudaOccDeviceProp properties,
		    cudaOccFuncAttributes attributes,
		    cudaOccDeviceState state,
		    del_blockSizeToDynamicSMemSize blockSizeToSMem)
		{
		    int maxOccupancy       = properties.maxThreadsPerMultiProcessor;
		    int largestBlockSize   = min_(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
		    int granularity        = properties.warpSize;
		    int maxBlockSize  = 0;
		    int blockSize     = 0;
		    int highestOccupancy   = 0;

		    for(blockSize = largestBlockSize; blockSize > 0; blockSize -= granularity)
		    {
				cudaOccResult res = cudaOccMaxActiveBlocksPerMultiprocessor(properties, attributes, blockSize, blockSizeToSMem(blockSize), state);
				int occupancy = res.ActiveBlocksPerMultiProcessor;
		        occupancy = blockSize*occupancy;

		        if(occupancy > highestOccupancy)
		        {
		            maxBlockSize = blockSize;
		            highestOccupancy = occupancy;
		        }

		        // can not get higher occupancy
		        if(highestOccupancy == maxOccupancy)
		            break;
		    }

		    return maxBlockSize;
		}




	}
}
