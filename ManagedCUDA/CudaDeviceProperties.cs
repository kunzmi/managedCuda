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
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;

namespace ManagedCuda
{
	/// <summary>
	/// CUDA device properties
	/// </summary>
	public class CudaDeviceProperties
	{
		// Fields
		private int _clockRate;
		private dim3 _maxBlockDim;
		private dim3 _maxGridDim;
		private int _maxThreadsPerBlock;
		private int _memPitch;
		private int _regsPerBlock;
		private int _sharedMemPerBlock;
		private int _textureAlign;
		private int _totalConstantMemory;
		private string _deviceName;
		//private Version _computeCapability;
		private Version _driverVersion;
		private SizeT _totalGlobalMemory;
		private int _multiProcessorCount;
		private int _warpSize;
		private bool _gpuOverlap;
		private bool _kernelExecTimeoutEnabled;
		private bool _integrated;
		private bool _canMapHostMemory;
		private BasicTypes.CUComputeMode _computeMode;
		private int _maximumTexture1DWidth;
		private int _maximumTexture2DWidth;
		private int _maximumTexture2DHeight;
		private int _maximumTexture3DWidth;
		private int _maximumTexture3DHeight;
		private int _maximumTexture3DDepth;
		private int _maximumTexture2DArrayWidth;
		private int _maximumTexture2DArrayHeight;
		private int _maximumTexture2DArrayNumSlices;
		private int _surfaceAllignment;
		private bool _concurrentKernels;
		private bool _ECCEnabled;
		private int _PCIBusID;
		private int _PCIDeviceID;
		private bool _TCCDriver;
		private int _memoryClockRate;
		private int _globalMemoryBusWidth;
		private int _L2CacheSize;
		private int _maxThreadsPerMultiProcessor;
		private int _asyncEngineCount;
		private bool _unifiedAddressing;
		private int _maximumTexture1DLayeredWidth;
		private int _maximumTexture1DLayeredLayers;
		private int _PCIDomainID;
		
		private int _texturePitchAlignment;
		private int _maximumTextureCubeMapWidth;     
		private int _maximumTextureCubeMapLayeredWidth;  
		private int _maximumTextureCubeMapLayeredLayers;
		private int _maximumSurface1DWidth;        
		private int _maximumSurface2DWidth;         
		private int _maximumSurface2DHeight;      
		private int _maximumSurface3DWidth;          
		private int _maximumSurface3DHeight;      
		private int _maximumSurface3DDepth;      
		private int _maximumSurface1DLayeredWidth;
		private int _maximumSurface1DLayeredLayers;
		private int _maximumSurface2DLayeredWidth;
		private int _maximumSurface2DLayeredHeight;
		private int _maximumSurface2DLayeredLayers;
		private int _maximumSurfaceCubemapWidth;  
		private int _maximumSurfaceCubemapLayeredWidth;
		private int _maximumSurfaceCubemapLayeredLayers;
		private int _maximumTexture1DLinearWidth; 
		private int _maximumTexture2DLinearWidth;
		private int _maximumTexture2DLinearHeight;
		private int _maximumTexture2DLinearPitch; 
		private int _maximumTexture2DMipmappedWidth;
		private int _maximumTexture2DMipmappedHeight;
		private int _computeCapabilityMajor;
		private int _computeCapabilityMinor;
		private int _maximumTexture1DMipmappedWidth;
		private bool _deviceSupportsStreamPriorities;
		private bool _globalL1CacheSupported;
		private bool _localL1CacheSupported;
		private int _maxSharedMemoryPerMultiprocessor;
		private int _maxRegistersPerMultiprocessor;
		private bool _managedMemory;
		private bool _multiGPUBoard;
		private int _multiGPUBoardGroupID;
		private bool _hostNativeAtomicSupported;
		private int _singleToDoublePrecisionPerfRatio;
		private bool _pageableMemoryAccess;
		private bool _concurrentManagedAccess;
		private bool _computePreemptionSupported;
		private bool _canUseHostPointerForRegisteredMem;
        private bool _canUseStreamMemOps;
        private bool _canUse64BitStreamMemOps;
        private bool _canUseStreamWaitValueNOr;
        private bool _cooperativeLaunch;
        private bool _cooperativeMultiDeviceLaunch;
        private int _maxSharedMemoryPerBlockOptin;
        private bool _canFlushRemoteWrites;
        private bool _hostRegisterSupported;
        private bool _pageableMemoryAccessUsesHostPageTables;
        private bool _directManagedMemoryAccessFromHost;
		private bool _virtualAddressManagementSupported;
		private bool _handleTypePosixFileDescriptorSupported;
		private bool _handleTypeWin32HandleSupported;
		private bool _handleTypeWin32KMTHandleSupported;

		// Properties
		/// <summary>
		/// Typical clock frequency in kilohertz
		/// </summary>
		public int ClockRate
		{
			get
			{
				return this._clockRate;
			}
			internal set
			{
				this._clockRate = value;
			}
		}

		/// <summary>
		/// Maximum block dimensions
		/// </summary>
		public dim3 MaxBlockDim
		{
			get
			{
				return this._maxBlockDim;
			}
			internal set
			{
				this._maxBlockDim = value;
			}
		}

		/// <summary>
		/// Maximum grid dimensions
		/// </summary>
		public dim3 MaxGridDim
		{
			get
			{
				return this._maxGridDim;
			}
			internal set
			{
				this._maxGridDim = value;
			}
		}

		/// <summary>
		/// Maximum number of threads per block
		/// </summary>
		public int MaxThreadsPerBlock
		{
			get
			{
				return this._maxThreadsPerBlock;
			}
			internal set
			{
				this._maxThreadsPerBlock = value;
			}
		}

		/// <summary>
		/// Maximum pitch in bytes allowed by memory copies
		/// </summary>
		public int MemoryPitch
		{
			get
			{
				return this._memPitch;
			}
			internal set
			{
				this._memPitch = value;
			}
		}

		/// <summary>
		/// Maximum number of 32-bit registers available per block
		/// </summary>
		public int RegistersPerBlock
		{
			get
			{
				return this._regsPerBlock;
			}
			internal set
			{
				this._regsPerBlock = value;
			}
		}

		/// <summary>
		/// Maximum shared memory available per block in bytes
		/// </summary>
		public int SharedMemoryPerBlock
		{
			get
			{
				return this._sharedMemPerBlock;
			}
			internal set
			{
				this._sharedMemPerBlock = value;
			}
		}

		/// <summary>
		/// Alignment requirement for textures
		/// </summary>
		public int TextureAlign
		{
			get
			{
				return this._textureAlign;
			}
			set
			{
				this._textureAlign = value;
			}
		}

		/// <summary>
		/// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
		/// </summary>
		public int TotalConstantMemory
		{
			get
			{
				return this._totalConstantMemory;
			}
			internal set
			{
				this._totalConstantMemory = value;
			}
		}

		/// <summary>
		/// Name of the device
		/// </summary>
		public string DeviceName
		{
			get { return this._deviceName; }
			internal set { this._deviceName = value; }
		}

		///// <summary>
		///// Compute capability version
		///// </summary>
		//public Version ComputeCapability
		//{
		//    get { return this._computeCapability; }
		//    internal set { this._computeCapability = value; }
		//}

		/// <summary>
		/// Driver version
		/// </summary>
		public Version DriverVersion
		{
			get { return this._driverVersion; }
			internal set { this._driverVersion = value; }
		}

		/// <summary>
		/// Total amount of global memory on the device
		/// </summary>
		public SizeT TotalGlobalMemory
		{
			get { return this._totalGlobalMemory; }
			internal set { this._totalGlobalMemory = value; }
		}

		/// <summary>
		/// Number of multiprocessors on device
		/// </summary>
		public int MultiProcessorCount
		{
			get { return this._multiProcessorCount; }
			internal set { this._multiProcessorCount = value; }
		}

		/// <summary>
		/// Warp size in threads (also called SIMDWith)
		/// </summary>
		public int WarpSize
		{
			get { return this._warpSize; }
			internal set { this._warpSize = value; }
		}

		/// <summary>
		/// Device can possibly copy memory and execute a kernel concurrently
		/// </summary>
		public bool GpuOverlap
		{
			get { return this._gpuOverlap; }
			internal set { this._gpuOverlap = value; }
		}

		/// <summary>
		/// Specifies whether there is a run time limit on kernels
		/// </summary>
		public bool KernelExecTimeoutEnabled
		{
			get { return this._kernelExecTimeoutEnabled; }
			internal set { this._kernelExecTimeoutEnabled = value; }
		}

		/// <summary>
		/// Device is integrated with host memory
		/// </summary>
		public bool Integrated
		{
			get { return this._integrated; }
			internal set { this._integrated = value; }
		}

		/// <summary>
		/// Device can map host memory into CUDA address space
		/// </summary>
		public bool CanMapHostMemory
		{
			get { return this._canMapHostMemory; }
			internal set { this._canMapHostMemory = value; }
		}

		/// <summary>
		/// Compute mode (See CUComputeMode for details)
		/// </summary>
		public BasicTypes.CUComputeMode ComputeMode
		{
			get { return this._computeMode; }
			internal set { this._computeMode = value; }
		}


		/// <summary>
		/// Maximum 1D texture width
		/// </summary>
		public int MaximumTexture1DWidth
		{
			get { return this._maximumTexture1DWidth; }
			internal set { this._maximumTexture1DWidth = value; }
		}

		/// <summary>
		/// Maximum 2D texture width
		/// </summary>
		public int MaximumTexture2DWidth
		{
			get { return this._maximumTexture2DWidth; }
			internal set { this._maximumTexture2DWidth = value; }
		}

		/// <summary>
		/// Maximum 2D texture height
		/// </summary>
		public int MaximumTexture2DHeight
		{
			get { return this._maximumTexture2DHeight; }
			internal set { this._maximumTexture2DHeight = value; }
		}

		/// <summary>
		/// Maximum 3D texture width
		/// </summary>
		public int MaximumTexture3DWidth
		{
			get { return this._maximumTexture3DWidth; }
			internal set { this._maximumTexture3DWidth = value; }
		}

		/// <summary>
		/// Maximum 3D texture height
		/// </summary>
		public int MaximumTexture3DHeight
		{
			get { return this._maximumTexture3DHeight; }
			internal set { this._maximumTexture3DHeight = value; }
		}

		/// <summary>
		/// Maximum 3D texture depth
		/// </summary>
		public int MaximumTexture3DDepth
		{
			get { return this._maximumTexture3DDepth; }
			internal set { this._maximumTexture3DDepth = value; }
		}

		/// <summary>
		/// Maximum texture array width
		/// </summary>
		public int MaximumTexture2DArrayWidth
		{
			get { return this._maximumTexture2DArrayWidth; }
			internal set { this._maximumTexture2DArrayWidth = value; }
		}

		/// <summary>
		/// Maximum texture array height
		/// </summary>
		public int MaximumTexture2DArrayHeight
		{
			get { return this._maximumTexture2DArrayHeight; }
			internal set { this._maximumTexture2DArrayHeight = value; }
		}

		/// <summary>
		/// Maximum slices in a texture array
		/// </summary>
		public int MaximumTexture2DArrayNumSlices
		{
			get { return this._maximumTexture2DArrayNumSlices; }
			internal set { this._maximumTexture2DArrayNumSlices = value; }
		}

		/// <summary>
		/// Alignment requirement for surfaces
		/// </summary>
		public int SurfaceAllignment
		{
			get { return this._surfaceAllignment; }
			internal set { this._surfaceAllignment = value; }
		}

		/// <summary>
		/// Device can possibly execute multiple kernels concurrently
		/// </summary>
		public bool ConcurrentKernels
		{
			get { return this._concurrentKernels; }
			internal set { this._concurrentKernels = value; }
		}

		/// <summary>
		/// Device has ECC support enabled
		/// </summary>
		public bool EccEnabled
		{
			get { return this._ECCEnabled; }
			internal set { this._ECCEnabled = value; }
		}

		/// <summary>
		/// PCI bus ID of the device
		/// </summary>
		public int PciBusId
		{
			get { return this._PCIBusID; }
			internal set { this._PCIBusID = value; }
		}

		/// <summary>
		/// PCI device ID of the device
		/// </summary>
		public int PciDeviceId
		{
			get { return this._PCIDeviceID; }
			internal set { this._PCIDeviceID = value; }
		}

		/// <summary>
		/// Device is using TCC driver model
		/// </summary>
		public bool TccDrivelModel
		{
			get { return this._TCCDriver; }
			internal set { this._TCCDriver = value; }
		}

		/// <summary>
		/// Peak memory clock frequency in kilohertz
		/// </summary>
		public int MemoryClockRate
		{
			get { return this._memoryClockRate; }
			internal set { this._memoryClockRate = value; }
		}

		/// <summary>
		/// Global memory bus width in bits
		/// </summary>
		public int GlobalMemoryBusWidth
		{
			get { return this._globalMemoryBusWidth; }
			internal set { this._globalMemoryBusWidth = value; }
		}

		/// <summary>
		/// Size of L2 cache in bytes
		/// </summary>
		public int L2CacheSize
		{
			get { return this._L2CacheSize; }
			internal set { this._L2CacheSize = value; }
		}

		/// <summary>
		/// Maximum resident threads per multiprocessor
		/// </summary>
		public int MaxThreadsPerMultiProcessor
		{
			get { return this._maxThreadsPerMultiProcessor; }
			internal set { this._maxThreadsPerMultiProcessor = value; }
		}

		/// <summary>
		/// Number of asynchronous engines
		/// </summary>
		public int AsyncEngineCount
		{
			get { return this._asyncEngineCount; }
			internal set { this._asyncEngineCount = value; }
		}

		/// <summary>
		/// Device shares a unified address space with the host
		/// </summary>
		public bool UnifiedAddressing
		{
			get { return this._unifiedAddressing; }
			internal set { this._unifiedAddressing = value; }
		}

		/// <summary>
		/// Maximum 1D layered texture width
		/// </summary>
		public int MaximumTexture1DLayeredWidth
		{
			get { return this._maximumTexture1DLayeredWidth; }
			internal set { this._maximumTexture1DLayeredWidth = value; }
		}

		/// <summary>
		/// Maximum layers in a 1D layered texture
		/// </summary>
		public int MaximumTexture1DLayeredLayers
		{
			get { return this._maximumTexture1DLayeredLayers; }
			internal set { this._maximumTexture1DLayeredLayers = value; }
		}

		/// <summary>
		/// PCI domain ID of the device
		/// </summary>
		public int PCIDomainID
		{
			get { return this._PCIDomainID; }
			internal set { this._PCIDomainID = value; }
		}

		
		/// <summary>
		/// Pitch alignment requirement for textures
		/// </summary>
		public int TexturePitchAlignment
		{
			get { return this._texturePitchAlignment; }
			internal set { this._texturePitchAlignment = value; }
		}
		/// <summary>
		/// Maximum cubemap texture width/height
		/// </summary>
		public int MaximumTextureCubeMapWidth
		{
			get { return this._maximumTextureCubeMapWidth; }
			internal set { this._maximumTextureCubeMapWidth = value; }
		}    
		/// <summary>
		/// Maximum cubemap layered texture width/height
		/// </summary>
		public int MaximumTextureCubeMapLayeredWidth
		{
			get { return this._maximumTextureCubeMapLayeredWidth; }
			internal set { this._maximumTextureCubeMapLayeredWidth = value; }
		}
		/// <summary>
		/// Maximum layers in a cubemap layered texture
		/// </summary>
		public int MaximumTextureCubeMapLayeredLayers
		{
			get { return this._maximumTextureCubeMapLayeredLayers; }
			internal set { this._maximumTextureCubeMapLayeredLayers = value; }
		} 
		/// <summary>
		/// Maximum 1D surface width
		/// </summary>
		public int MaximumSurface1DWidth
		{
			get { return this._maximumSurface1DWidth; }
			internal set { this._maximumSurface1DWidth = value; }
		}          
		/// <summary>
		/// Maximum 2D surface width
		/// </summary>
		public int MaximumSurface2DWidth 
		{
			get { return this._maximumSurface2DWidth; }
			internal set { this._maximumSurface2DWidth = value; }
		}      
		/// <summary>
		/// Maximum 2D surface height
		/// </summary>
		public int MaximumSurface2DHeight
		{
			get { return this._maximumSurface2DHeight; }
			internal set { this._maximumSurface2DHeight = value; }
		}    
		/// <summary>
		/// Maximum 3D surface width
		/// </summary>
		public int MaximumSurface3DWidth
		{
			get { return this._maximumSurface3DWidth; }
			internal set { this._maximumSurface3DWidth = value; }
		}        
		/// <summary>
		/// Maximum 3D surface height
		/// </summary>
		public int MaximumSurface3DHeight
		{
			get { return this._maximumSurface3DHeight; }
			internal set { this._maximumSurface3DHeight = value; }
		}      
		/// <summary>
		/// Maximum 3D surface depth
		/// </summary>
		public int MaximumSurface3DDepth 
		{
			get { return this._maximumSurface3DDepth; }
			internal set { this._maximumSurface3DDepth = value; }
		}          
		/// <summary>
		/// Maximum 1D layered surface width
		/// </summary>
		public int MaximumSurface1DLayeredWidth 
		{
			get { return this._maximumSurface1DLayeredWidth; }
			internal set { this._maximumSurface1DLayeredWidth = value; }
		}
		/// <summary>
		/// Maximum layers in a 1D layered surface
		/// </summary>
		public int MaximumSurface1DLayeredLayers
		{
			get { return this._maximumSurface1DLayeredLayers; }
			internal set { this._maximumSurface1DLayeredLayers = value; }
		}
		/// <summary>
		/// Maximum 2D layered surface width
		/// </summary>
		public int MaximumSurface2DLayeredWidth 
		{
			get { return this._maximumSurface2DLayeredWidth; }
			internal set { this._maximumSurface2DLayeredWidth = value; }
		}
		/// <summary>
		/// Maximum 2D layered surface height
		/// </summary>
		public int MaximumSurface2DLayeredHeight
		{
			get { return this._maximumSurface2DLayeredHeight; }
			internal set { this._maximumSurface2DLayeredHeight = value; }
		}  
		/// <summary>
		/// Maximum layers in a 2D layered surface
		/// </summary>
		public int MaximumSurface2DLayeredLayers
		{
			get { return this._maximumSurface2DLayeredLayers; }
			internal set { this._maximumSurface2DLayeredLayers = value; }
		} 
		/// <summary>
		/// Maximum cubemap surface width
		/// </summary>
		public int MaximumSurfaceCubemapWidth
		{
			get { return this._maximumSurfaceCubemapWidth; }
			internal set { this._maximumSurfaceCubemapWidth = value; }
		}   
		/// <summary>
		/// Maximum cubemap layered surface width
		/// </summary>
		public int MaximumSurfaceCubemapLayeredWidth
		{
			get { return this._maximumSurfaceCubemapLayeredWidth; }
			internal set { this._maximumSurfaceCubemapLayeredWidth = value; }
		}
		/// <summary>
		/// Maximum layers in a cubemap layered surface
		/// </summary>
		public int MaximumSurfaceCubemapLayeredLayers
		{
			get { return this._maximumSurfaceCubemapLayeredLayers; }
			internal set { this._maximumSurfaceCubemapLayeredLayers = value; }
		}
		/// <summary>
		/// Maximum 1D linear texture width
		/// </summary>
		public int MaximumTexture1DLinearWidth
		{
			get { return this._maximumTexture1DLinearWidth; }
			internal set { this._maximumTexture1DLinearWidth = value; }
		}  
		/// <summary>
		/// Maximum 2D linear texture width
		/// </summary>
		public int MaximumTexture2DLinearWidth
		{
			get { return this._maximumTexture2DLinearWidth; }
			internal set { this._maximumTexture2DLinearWidth = value; }
		}  
		/// <summary>
		/// Maximum 2D linear texture height
		/// </summary>
		public int MaximumTexture2DLinearHeight
		{
			get { return this._maximumTexture2DLinearHeight; }
			internal set { this._maximumTexture2DLinearHeight = value; }
		} 
		/// <summary>
		/// Maximum 2D linear texture pitch in bytes
		/// </summary>
		public int MaximumTexture2DLinearPitch
		{
			get { return this._maximumTexture2DLinearPitch; }
			internal set { this._maximumTexture2DLinearPitch = value; }
		} 
  		/// <summary>
		/// Maximum mipmapped 2D texture width
		/// </summary>
		public int MaximumTexture2DMipmappedWidth
		{
			get { return this._maximumTexture2DMipmappedWidth; }
			internal set { this._maximumTexture2DMipmappedWidth = value; }
		} 
		/// <summary>
		/// Maximum mipmapped 2D texture height
		/// </summary>
		public int MaximumTexture2DMipmappedHeight
		{
			get { return this._maximumTexture2DMipmappedHeight; }
			internal set { this._maximumTexture2DMipmappedHeight = value; }
		} 
		/// <summary>
		/// Major compute capability version number
		/// </summary>
		internal int ComputeCapabilityMajor
		{
			get { return this._computeCapabilityMajor; }
			set { this._computeCapabilityMajor = value; }
		} 
		/// <summary>
		/// Minor compute capability version number
		/// </summary>
		internal int ComputeCapabilityMinor
		{
			get { return this._computeCapabilityMinor; }
			set { this._computeCapabilityMinor = value; }
		} 
		/// <summary>
		/// Compute capability version number
		/// </summary>
		public Version ComputeCapability
		{
			get { return new Version(this._computeCapabilityMajor, this._computeCapabilityMinor); }
		} 
		/// <summary>
		/// Maximum mipmapped 1D texture width
		/// </summary>
		public int MaximumTexture1DMipmappedWidth
		{
			get { return this._maximumTexture1DMipmappedWidth; }
			internal set { this._maximumTexture1DMipmappedWidth = value; }
		}

		/// <summary>
		/// Device supports stream priorities
		/// </summary>
		public bool SupportsStreamPriorities
		{
			get { return this._deviceSupportsStreamPriorities; }
			internal set { this._deviceSupportsStreamPriorities = value; }
		}

		/// <summary>
		/// Device supports caching globals in L1
		/// </summary>
		public bool GlobalL1CacheSupported
		{
			get { return this._globalL1CacheSupported; }
			internal set { this._globalL1CacheSupported = value; }
		}

		/// <summary>
		/// Device supports caching locals in L1
		/// </summary>
		public bool LocalL1CacheSupported
		{
			get { return this._localL1CacheSupported; }
			internal set { this._localL1CacheSupported = value; }
		}

		/// <summary>
		/// Maximum shared memory available per multiprocessor in bytes
		/// </summary>
		public int MaxSharedMemoryPerMultiprocessor
		{
			get { return this._maxSharedMemoryPerMultiprocessor; }
			internal set { this._maxSharedMemoryPerMultiprocessor = value; }
		}

		/// <summary>
		/// Maximum number of 32-bit registers available per multiprocessor
		/// </summary>
		public int MaxRegistersPerMultiprocessor
		{
			get { return this._maxRegistersPerMultiprocessor; }
			internal set { this._maxRegistersPerMultiprocessor = value; }
		}

		/// <summary>
		/// Device can allocate managed memory on this system
		/// </summary>
		public bool ManagedMemory
		{
			get { return this._managedMemory; }
			internal set { this._managedMemory = value; }
		}

		/// <summary>
		/// Device is on a multi-GPU board
		/// </summary>
		public bool MultiGPUBoard
		{
			get { return this._multiGPUBoard; }
			internal set { this._multiGPUBoard = value; }
		}

		/// <summary>
		/// Unique id for a group of devices on the same multi-GPU board
		/// </summary>
		public int MultiGPUBoardGroupID
		{
			get { return this._multiGPUBoardGroupID; }
			internal set { this._multiGPUBoardGroupID = value; }
		}

		/// <summary>
		/// Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
		/// </summary>
		public bool HostNativeAtomicSupported
		{
			get { return this._hostNativeAtomicSupported; }
			internal set { this._hostNativeAtomicSupported = value; }
		}

		/// <summary>
		/// Ratio of single precision performance (in floating-point operations per second) to double precision performance
		/// </summary>
		public int SingleToDoublePrecisionPerfRatio
		{
			get { return this._singleToDoublePrecisionPerfRatio; }
			internal set { this._singleToDoublePrecisionPerfRatio = value; }
		}
		/// <summary>
		/// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
		/// </summary>
		public bool PageableMemoryAccess
		{
			get { return this._pageableMemoryAccess; }
			internal set { this._pageableMemoryAccess = value; }
		}
		/// <summary>
		/// Device can coherently access managed memory concurrently with the CPU
		/// </summary>
		public bool ConcurrentManagedAccess
		{
			get { return this._concurrentManagedAccess; }
			internal set { this._concurrentManagedAccess = value; }
		}
		/// <summary>
		/// Device supports compute preemption.
		/// </summary>
		public bool ComputePreemptionSupported
		{
			get { return this._computePreemptionSupported; }
			internal set { this._computePreemptionSupported = value; }
		}
        /// <summary>
        /// Device can access host registered memory at the same virtual address as the CPU.
        /// </summary>
        public bool CanUseHostPointerForRegisteredMem
        {
			get { return this._canUseHostPointerForRegisteredMem; }
			internal set { this._canUseHostPointerForRegisteredMem = value; }
		}

        /// <summary>
        /// cuStreamBatchMemOp and related APIs are supported.
        /// </summary>
        public bool CanUseStreamMemOps
        {
            get { return this._canUseStreamMemOps; }
            internal set { this._canUseStreamMemOps = value; }
        }

        /// <summary>
        /// 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs.
        /// </summary>
        public bool CanUse64BitStreamMemOps
        {
            get { return this._canUse64BitStreamMemOps; }
            internal set { this._canUse64BitStreamMemOps = value; }
        }
        /// <summary>
        /// CU_STREAM_WAIT_VALUE_NOR is supported.
        /// </summary>
        public bool CanUseStreamWaitValueNOr
        {
            get { return this._canUseStreamWaitValueNOr; }
            internal set { this._canUseStreamWaitValueNOr = value; }
        }
        /// <summary>
        /// Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel
        /// </summary>
        public bool CooperativeLaunch
        {
            get { return this._cooperativeLaunch; }
            internal set { this._cooperativeLaunch = value; }
        }
        /// <summary>
        /// Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice
        /// </summary>
        public bool CooperativeMultiDeviceLaunch
        {
            get { return this._cooperativeMultiDeviceLaunch; }
            internal set { this._cooperativeMultiDeviceLaunch = value; }
        }
        /// <summary>
        /// Maximum optin shared memory per block
        /// </summary>
        public int MaxSharedMemoryPerBlockOptin
        {
            get { return this._maxSharedMemoryPerBlockOptin; }
            internal set { this._maxSharedMemoryPerBlockOptin = value; }
        }
        /// <summary>
        /// Both the ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details.
        /// </summary>
        public bool CanFlushRemoteWrites
        {
            get { return this._canFlushRemoteWrites; }
            internal set { this._canFlushRemoteWrites = value; }
        }
        /// <summary>
        /// Device supports host memory registration via ::cudaHostRegister.
        /// </summary>
        public bool HostRegisterSupported
        {
            get { return this._hostRegisterSupported; }
            internal set { this._hostRegisterSupported = value; }
        }
        /// <summary>
        /// Device accesses pageable memory via the host's page tables.
        /// </summary>
        public bool PageableMemoryAccessUsesHostPageTables
        {
            get { return this._pageableMemoryAccessUsesHostPageTables; }
            internal set { this._pageableMemoryAccessUsesHostPageTables = value; }
        }
        /// <summary>
        /// The host can directly access managed memory on the device without migration.
        /// </summary>
        public bool DirectManagedMemoryAccessFromHost
        {
            get { return this._directManagedMemoryAccessFromHost; }
            internal set { this._directManagedMemoryAccessFromHost = value; }
		}
		/// <summary>
		/// Device supports virtual address management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
		/// </summary>
		public bool VirtualAddressManagementSupported
		{
			get { return this._virtualAddressManagementSupported; }
			internal set { this._virtualAddressManagementSupported = value; }
		}
		/// <summary>
		/// Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
		/// </summary>
		public bool HandleTypePosixFileDescriptorSupported
		{
			get { return this._handleTypePosixFileDescriptorSupported; }
			internal set { this._handleTypePosixFileDescriptorSupported = value; }
		}
		/// <summary>
		/// Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
		/// </summary>
		public bool HandleTypeWin32HandleSupported
		{
			get { return this._handleTypeWin32HandleSupported; }
			internal set { this._handleTypeWin32HandleSupported = value; }
		}
		/// <summary>
		/// Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested ::cuMemCreate
		/// </summary>
		public bool HandleTypeWin32KMTHandleSupported
		{
			get { return this._handleTypeWin32KMTHandleSupported; }
			internal set { this._handleTypeWin32KMTHandleSupported = value; }
		}
	}
}
