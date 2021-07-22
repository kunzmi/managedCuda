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
using System.Diagnostics;

namespace ManagedCuda.BasicTypes
{


    #region Enums
    /// <summary>
    /// Texture reference addressing modes
    /// </summary>
    public enum CUAddressMode
    {
        /// <summary>
        /// Wrapping address mode
        /// </summary>
        Wrap = 0,

        /// <summary>
        /// Clamp to edge address mode
        /// </summary>
        Clamp = 1,

        /// <summary>
        /// Mirror address mode
        /// </summary>
        Mirror = 2,

        /// <summary>
        /// Border address mode
        /// </summary>
        Border = 3
    }

    /// <summary>
    /// Array formats
    /// </summary>
    public enum CUArrayFormat
    {
        /// <summary>
        /// Unsigned 8-bit integers
        /// </summary>
        UnsignedInt8 = 0x01,

        /// <summary>
        /// Unsigned 16-bit integers
        /// </summary>
        UnsignedInt16 = 0x02,

        /// <summary>
        /// Unsigned 32-bit integers
        /// </summary>
        UnsignedInt32 = 0x03,

        /// <summary>
        /// Signed 8-bit integers
        /// </summary>
        SignedInt8 = 0x08,

        /// <summary>
        /// Signed 16-bit integers
        /// </summary>
        SignedInt16 = 0x09,

        /// <summary>
        /// Signed 32-bit integers
        /// </summary>
        SignedInt32 = 0x0a,

        /// <summary>
        /// 16-bit floating point
        /// </summary>
        Half = 0x10,

        /// <summary>
        /// 32-bit floating point
        /// </summary>
        Float = 0x20,

        /// <summary>
        /// 8-bit YUV planar format, with 4:2:0 sampling
        /// </summary>
        NV12 = 0xb0
    }

    /// <summary>
    /// Compute mode that device is currently in.
    /// </summary>
    public enum CUComputeMode
    {
        /// <summary>
        /// Default mode - Device is not restricted and can have multiple CUDA
        /// contexts present at a single time.
        /// </summary>
        Default = 0,

        ///// <summary>
        ///// Compute-exclusive mode - Device can have only one CUDA context
        ///// present on it at a time.
        ///// </summary>
        //Exclusive = 1,

        /// <summary>
        /// Compute-prohibited mode - Device is prohibited from creating
        /// new CUDA contexts.
        /// </summary>
        Prohibited = 2,

        /// <summary>
        /// Compute-exclusive-process mode (Only one context used by a 
        /// single process can be present on this device at a time)
        /// </summary>
        ExclusiveProcess = 2
    }

    /// <summary>
    /// Memory advise values
    /// </summary>
    public enum CUmemAdvise
    {
        /// <summary>
        /// Data will mostly be read and only occassionally be written to
        /// </summary>
        SetReadMostly = 1,
        /// <summary>
        /// Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY
        /// </summary>
        UnsetReadMostly = 2,
        /// <summary>
        /// Set the preferred location for the data as the specified device
        /// </summary>
        SetPreferredLocation = 3,
        /// <summary>
        /// Clear the preferred location for the data
        /// </summary>
        UnsetPreferredLocation = 4,
        /// <summary>
        /// Data will be accessed by the specified device, so prevent page faults as much as possible
        /// </summary>
        SetAccessedBy = 5,
        /// <summary>
        /// Let the Unified Memory subsystem decide on the page faulting policy for the specified device
        /// </summary>
        UnsetAccessedBy = 6
    }

    /// <summary>
    /// Context Attach flags
    /// </summary>
    public enum CUCtxAttachFlags
    {
        /// <summary>
        /// None
        /// </summary>
        None = 0
    }

    /// <summary>
    /// Device properties
    /// </summary>
    public enum CUDeviceAttribute
    {
        /// <summary>
        /// Maximum number of threads per block
        /// </summary>
        MaxThreadsPerBlock = 1,

        /// <summary>
        /// Maximum block dimension X
        /// </summary>
        MaxBlockDimX = 2,

        /// <summary>
        /// Maximum block dimension Y
        /// </summary>
        MaxBlockDimY = 3,

        /// <summary>
        /// Maximum block dimension Z
        /// </summary>
        MaxBlockDimZ = 4,

        /// <summary>
        /// Maximum grid dimension X
        /// </summary>
        MaxGridDimX = 5,

        /// <summary>
        /// Maximum grid dimension Y
        /// </summary>
        MaxGridDimY = 6,

        /// <summary>
        /// Maximum grid dimension Z
        /// </summary>
        MaxGridDimZ = 7,

        /// <summary>
        /// Maximum amount of shared memory
        /// available to a thread block in bytes; this amount is shared by all thread blocks simultaneously resident on a
        /// multiprocessor
        /// </summary>
        MaxSharedMemoryPerBlock = 8,

        /// <summary>
        /// Deprecated, use MaxSharedMemoryPerBlock
        /// </summary>
        [Obsolete("Use MaxSharedMemoryPerBlock")]
        SharedMemoryPerBlock = 8,

        /// <summary>
        /// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
        /// </summary>
        TotalConstantMemory = 9,

        /// <summary>
        /// Warp size in threads
        /// </summary>
        WarpSize = 10,

        /// <summary>
        /// Maximum pitch in bytes allowed by the memory copy functions
        /// that involve memory regions allocated through <see cref="DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2"/>
        /// </summary>
        MaxPitch = 11,

        /// <summary>
        /// Deprecated, use MaxRegistersPerBlock
        /// </summary>
        [Obsolete("Use MaxRegistersPerBlock")]
        RegistersPerBlock = 12,

        /// <summary>
        /// Maximum number of 32-bit registers available
        /// to a thread block; this number is shared by all thread blocks simultaneously resident on a multiprocessor
        /// </summary>
        MaxRegistersPerBlock = 12,

        /// <summary>
        /// Typical clock frequency in kilohertz
        /// </summary>
        ClockRate = 13,

        /// <summary>
        /// Alignment requirement; texture base addresses
        /// aligned to textureAlign bytes do not need an offset applied to texture fetches
        /// </summary>
        TextureAlignment = 14,

        /// <summary>
        /// 1 if the device can concurrently copy memory between host
        /// and device while executing a kernel, or 0 if not
        /// </summary>
        GPUOverlap = 15,

        /// <summary>
        /// Number of multiprocessors on device
        /// </summary>
        MultiProcessorCount = 0x10,

        /// <summary>
        /// Specifies whether there is a run time limit on kernels. <para/>
        /// 1 if there is a run time limit for kernels executed on the device, or 0 if not
        /// </summary>
        KernelExecTimeout = 0x11,

        /// <summary>
        /// Device is integrated with host memory. 1 if the device is integrated with the memory subsystem, or 0 if not
        /// </summary>
        Integrated = 0x12,

        /// <summary>
        /// Device can map host memory into CUDA address space. 1 if the device can map host memory into the
        /// CUDA address space, or 0 if not
        /// </summary>
        CanMapHostMemory = 0x13,

        /// <summary>
        /// Compute mode (See <see cref="CUComputeMode"/> for details)
        /// </summary>
        ComputeMode = 20,


        /// <summary>
        /// Maximum 1D texture width
        /// </summary>
        MaximumTexture1DWidth = 21,

        /// <summary>
        /// Maximum 2D texture width
        /// </summary>
        MaximumTexture2DWidth = 22,

        /// <summary>
        /// Maximum 2D texture height
        /// </summary>
        MaximumTexture2DHeight = 23,

        /// <summary>
        /// Maximum 3D texture width
        /// </summary>
        MaximumTexture3DWidth = 24,

        /// <summary>
        /// Maximum 3D texture height
        /// </summary>
        MaximumTexture3DHeight = 25,

        /// <summary>
        /// Maximum 3D texture depth
        /// </summary>
        MaximumTexture3DDepth = 26,

        /// <summary>
        /// Maximum texture array width
        /// </summary>
        MaximumTexture2DArray_Width = 27,

        /// <summary>
        /// Maximum texture array height
        /// </summary>
        MaximumTexture2DArray_Height = 28,

        /// <summary>
        /// Maximum slices in a texture array
        /// </summary>
        MaximumTexture2DArray_NumSlices = 29,

        /// <summary>
        /// Alignment requirement for surfaces
        /// </summary>
        SurfaceAllignment = 30,

        /// <summary>
        /// Device can possibly execute multiple kernels concurrently. <para/>
        /// 1 if the device supports executing multiple kernels
        /// within the same context simultaneously, or 0 if not. It is not guaranteed that multiple kernels will be resident on
        /// the device concurrently so this feature should not be relied upon for correctness.
        /// </summary>
        ConcurrentKernels = 31,

        /// <summary>
        /// Device has ECC support enabled. 1 if error correction is enabled on the device, 0 if error correction
        /// is disabled or not supported by the device.
        /// </summary>
        ECCEnabled = 32,

        /// <summary>
        /// PCI bus ID of the device
        /// </summary>
        PCIBusID = 33,

        /// <summary>
        /// PCI device ID of the device
        /// </summary>
        PCIDeviceID = 34,

        /// <summary>
        /// Device is using TCC driver model
        /// </summary>
        TCCDriver = 35,

        /// <summary>
        /// Peak memory clock frequency in kilohertz
        /// </summary>
        MemoryClockRate = 36,

        /// <summary>
        /// Global memory bus width in bits
        /// </summary>
        GlobalMemoryBusWidth = 37,

        /// <summary>
        /// Size of L2 cache in bytes
        /// </summary>
        L2CacheSize = 38,

        /// <summary>
        /// Maximum resident threads per multiprocessor
        /// </summary>
        MaxThreadsPerMultiProcessor = 39,

        /// <summary>
        /// Number of asynchronous engines
        /// </summary>
        AsyncEngineCount = 40,

        /// <summary>
        /// Device shares a unified address space with the host
        /// </summary>
        UnifiedAddressing = 41,

        /// <summary>
        /// Maximum 1D layered texture width
        /// </summary>
        MaximumTexture1DLayeredWidth = 42,

        /// <summary>
        /// Maximum layers in a 1D layered texture
        /// </summary>
        MaximumTexture1DLayeredLayers = 43,

        /// <summary>
        /// PCI domain ID of the device
        /// </summary>
        PCIDomainID = 50,

        /// <summary>
        /// Pitch alignment requirement for textures
        /// </summary>
        TexturePitchAlignment = 51,
        /// <summary>
        /// Maximum cubemap texture width/height
        /// </summary>
        MaximumTextureCubeMapWidth = 52,
        /// <summary>
        /// Maximum cubemap layered texture width/height
        /// </summary>
        MaximumTextureCubeMapLayeredWidth = 53,
        /// <summary>
        /// Maximum layers in a cubemap layered texture
        /// </summary>
        MaximumTextureCubeMapLayeredLayers = 54,
        /// <summary>
        /// Maximum 1D surface width
        /// </summary>
        MaximumSurface1DWidth = 55,
        /// <summary>
        /// Maximum 2D surface width
        /// </summary>
        MaximumSurface2DWidth = 56,
        /// <summary>
        /// Maximum 2D surface height
        /// </summary>
        MaximumSurface2DHeight = 57,
        /// <summary>
        /// Maximum 3D surface width
        /// </summary>
        MaximumSurface3DWidth = 58,
        /// <summary>
        /// Maximum 3D surface height
        /// </summary>
        MaximumSurface3DHeight = 59,
        /// <summary>
        /// Maximum 3D surface depth
        /// </summary>
        MaximumSurface3DDepth = 60,
        /// <summary>
        /// Maximum 1D layered surface width
        /// </summary>
        MaximumSurface1DLayeredWidth = 61,
        /// <summary>
        /// Maximum layers in a 1D layered surface
        /// </summary>
        MaximumSurface1DLayeredLayers = 62,
        /// <summary>
        /// Maximum 2D layered surface width
        /// </summary>
        MaximumSurface2DLayeredWidth = 63,
        /// <summary>
        /// Maximum 2D layered surface height
        /// </summary>
        MaximumSurface2DLayeredHeight = 64,
        /// <summary>
        /// Maximum layers in a 2D layered surface
        /// </summary>
        MaximumSurface2DLayeredLayers = 65,
        /// <summary>
        /// Maximum cubemap surface width
        /// </summary>
        MaximumSurfaceCubemapWidth = 66,
        /// <summary>
        /// Maximum cubemap layered surface width
        /// </summary>
        MaximumSurfaceCubemapLayeredWidth = 67,
        /// <summary>
        /// Maximum layers in a cubemap layered surface
        /// </summary>
        MaximumSurfaceCubemapLayeredLayers = 68,
        /// <summary>
        /// Maximum 1D linear texture width
        /// </summary>
        [Obsolete("Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.")]
        MaximumTexture1DLinearWidth = 69,
        /// <summary>
        /// Maximum 2D linear texture width
        /// </summary>
        MaximumTexture2DLinearWidth = 70,
        /// <summary>
        /// Maximum 2D linear texture height
        /// </summary>
        MaximumTexture2DLinearHeight = 71,
        /// <summary>
        /// Maximum 2D linear texture pitch in bytes
        /// </summary>
        MaximumTexture2DLinearPitch = 72,
        /// <summary>
        /// Maximum mipmapped 2D texture width
        /// </summary>
        MaximumTexture2DMipmappedWidth = 73,
        /// <summary>
        /// Maximum mipmapped 2D texture height
        /// </summary>
        MaximumTexture2DMipmappedHeight = 74,
        /// <summary>
        /// Major compute capability version number
        /// </summary>
        ComputeCapabilityMajor = 75,
        /// <summary>
        /// Minor compute capability version number
        /// </summary>
        ComputeCapabilityMinor = 76,
        /// <summary>
        /// Maximum mipmapped 1D texture width
        /// </summary>
        MaximumTexture1DMipmappedWidth = 77,
        /// <summary>
        /// Device supports stream priorities
        /// </summary>
        StreamPrioritiesSupported = 78,
        /// <summary>
        /// Device supports caching globals in L1
        /// </summary>
        GlobalL1CacheSupported = 79,
        /// <summary>
        /// Device supports caching locals in L1
        /// </summary>
        LocalL1CacheSupported = 80,
        /// <summary>
        /// Maximum shared memory available per multiprocessor in bytes
        /// </summary>
        MaxSharedMemoryPerMultiprocessor = 81,
        /// <summary>
        /// Maximum number of 32-bit registers available per multiprocessor
        /// </summary>
        MaxRegistersPerMultiprocessor = 82,
        /// <summary>
        /// Device can allocate managed memory on this system
        /// </summary>
        ManagedMemory = 83,
        /// <summary>
        /// Device is on a multi-GPU board
        /// </summary>
        MultiGpuBoard = 84,
        /// <summary>
        /// Unique id for a group of devices on the same multi-GPU board
        /// </summary>
        MultiGpuBoardGroupID = 85,
        /// <summary>
        /// Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
        /// </summary>
        HostNativeAtomicSupported = 86,
        /// <summary>
        /// Ratio of single precision performance (in floating-point operations per second) to double precision performance
        /// </summary>
        SingleToDoublePrecisionPerfRatio = 87,
        /// <summary>
        /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
        /// </summary>
        PageableMemoryAccess = 88,
        /// <summary>
        /// Device can coherently access managed memory concurrently with the CPU
        /// </summary>
        ConcurrentManagedAccess = 89,
        /// <summary>
        /// Device supports compute preemption.
        /// </summary>
        ComputePreemptionSupported = 90,
        /// <summary>
        /// Device can access host registered memory at the same virtual address as the CPU.
        /// </summary>
        CanUseHostPointerForRegisteredMem = 91,
        /// <summary>
        /// ::cuStreamBatchMemOp and related APIs are supported.
        /// </summary>
        CanUseStreamMemOps = 92,
        /// <summary>
        /// 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs.
        /// </summary>
        CanUse64BitStreamMemOps = 93,
        /// <summary>
        /// ::CU_STREAM_WAIT_VALUE_NOR is supported.
        /// </summary>
        CanUseStreamWaitValueNOr = 94,
        /// <summary>
        /// Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel
        /// </summary>
        CooperativeLaunch = 95,
        /// <summary>
        /// Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice
        /// </summary>
        CooperativeMultiDeviceLaunch = 96,
        /// <summary>
        /// Maximum optin shared memory per block
        /// </summary>
        MaxSharedMemoryPerBlockOptin = 97,
        /// <summary>
        /// Both the ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details.
        /// </summary>
        CanFlushRemoteWrites = 98,
        /// <summary>
        /// Device supports host memory registration via ::cudaHostRegister.
        /// </summary>
        HostRegisterSupported = 99,
        /// <summary>
        /// Device accesses pageable memory via the host's page tables.
        /// </summary>
        PageableMemoryAccessUsesHostPageTables = 100,
        /// <summary>
        /// The host can directly access managed memory on the device without migration.
        /// </summary>
        DirectManagedMemoryAccessFromHost = 101,
        /// <summary>
        /// Deprecated, Use VirtualMemoryManagementSupported
        /// </summary>
        [Obsolete("Deprecated, Use VirtualMemoryManagementSupported")]
        VirtualAddressManagementSupported = 102,
        /// <summary>
        /// Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
        /// </summary>
        VirtualMemoryManagementSupported = 102,
        /// <summary>
        /// Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
        /// </summary>
        HandleTypePosixFileDescriptorSupported = 103,
        /// <summary>
        /// Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
        /// </summary>
        HandleTypeWin32HandleSupported = 104,
        /// <summary>
        /// Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested ::cuMemCreate
        /// </summary>
        HandleTypeWin32KMTHandleSupported = 105,
        /// <summary>
        /// Maximum number of blocks per multiprocessor
        /// </summary>
        MaxBlocksPerMultiProcessor = 106,
        /// <summary>
        /// Device supports compression of memory
        /// </summary>
        GenericCompressionSupported = 107,
        /// <summary>
        /// Device's maximum L2 persisting lines capacity setting in bytes
        /// </summary>
        MaxPersistingL2CacheSize = 108,
        /// <summary>
        /// The maximum value of CUaccessPolicyWindow::num_bytes.
        /// </summary>
        MaxAccessPolicyWindowSize = 109,
        /// <summary>
        /// Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
        /// </summary>
        GPUDirectRDMAWithCudaVMMSupported = 110,
        /// <summary>
        /// Shared memory reserved by CUDA driver per block in bytes
        /// </summary>
        ReservedSharedMemoryPerBlock = 111,
        /// <summary>
        /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
        /// </summary>
        SparseCudaArraySupported = 112,
        /// <summary>
        /// Device supports using the ::cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU
        /// </summary>
        ReadOnlyHostRegisterSupported = 113,
        /// <summary>
        /// External timeline semaphore interop is supported on the device
        /// </summary>
        TimelineSemaphoreInteropSupported = 114,
        /// <summary>
        /// Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs
        /// </summary>
        MemoryPoolsSupported = 115,
        /// <summary>
        /// Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
        /// </summary>
        GpuDirectRDMASupported = 116,
        /// <summary>
        /// The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum
        /// </summary>
        GpuDirectRDMAFlushWritesOptions = 117,
        /// <summary>
        /// GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
        /// </summary>
        GpuDirectRDMAWritesOrdering = 118,
        /// <summary>
        /// Handle types supported with mempool based IPC
        /// </summary>
        MempoolSupportedHandleTypes = 119,


        /// <summary>
        /// Max elems...
        /// </summary>
        MAX
    }

    /// <summary>
    /// Texture reference filtering modes
    /// </summary>
    public enum CUFilterMode
    {
        /// <summary>
        /// Point filter mode
        /// </summary>
        Point = 0,

        /// <summary>
        /// Linear filter mode
        /// </summary>
        Linear = 1
    }

    /// <summary>
    /// Function properties
    /// </summary>
    public enum CUFunctionAttribute
    {
        /// <summary>
        /// <para>The number of threads beyond which a launch of the function would fail.</para>
        /// <para>This number depends on both the function and the device on which the
        /// function is currently loaded.</para>
        /// </summary>
        MaxThreadsPerBlock = 0,

        /// <summary>
        /// <para>The size in bytes of statically-allocated shared memory required by
        /// this function. </para><para>This does not include dynamically-allocated shared
        /// memory requested by the user at runtime.</para>
        /// </summary>
        SharedSizeBytes = 1,

        /// <summary>
        /// <para>The size in bytes of statically-allocated shared memory required by
        /// this function. </para><para>This does not include dynamically-allocated shared
        /// memory requested by the user at runtime.</para>
        /// </summary>
        ConstSizeBytes = 2,

        /// <summary>
        /// The size in bytes of thread local memory used by this function.
        /// </summary>
        LocalSizeBytes = 3,

        /// <summary>
        /// The number of registers used by each thread of this function.
        /// </summary>
        NumRegs = 4,

        /// <summary>
        /// The PTX virtual architecture version for which the function was
        /// compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function
        /// would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA
        /// 3.0.
        /// </summary>
        PTXVersion = 5,

        /// <summary>
        /// The binary version for which the function was compiled. This
        /// value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return
        /// the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary
        /// architecture version.
        /// </summary>
        BinaryVersion = 6,

        /// <summary>
        /// The attribute to indicate whether the function has been compiled with 
        /// user specified option "-Xptxas --dlcm=ca" set.
        /// </summary>
        CacheModeCA = 7,

        /// <summary>
        /// The maximum size in bytes of dynamically-allocated shared memory that can be used by
        /// this function. If the user-specified dynamic shared memory size is larger than this
        /// value, the launch will fail.
        /// </summary>
        MaxDynamicSharedSizeBytes = 8,

        /// <summary>
        /// On devices where the L1 cache and shared memory use the same hardware resources, 
        /// this sets the shared memory carveout preference, in percent of the total resources. 
        /// This is only a hint, and the driver can choose a different ratio if required to execute the function.
        /// </summary>
        PreferredSharedMemoryCarveout = 9,

        /// <summary>
        /// No descritption found...
        /// </summary>
        Max
    }

    /// <summary>
    /// Function cache configurations
    /// </summary>
    public enum CUFuncCache
    {
        /// <summary>
        /// No preference for shared memory or L1 (default)
        /// </summary>
        PreferNone = 0x00,
        /// <summary>
        /// Function prefers larger shared memory and smaller L1 cache.
        /// </summary>
        PreferShared = 0x01,
        /// <summary>
        /// Function prefers larger L1 cache and smaller shared memory.
        /// </summary>
        PreferL1 = 0x02,
        /// <summary>
        /// Function prefers equal sized L1 cache and shared memory.
        /// </summary>
        PreferEqual = 0x03
    }

    /// <summary>
    /// Cubin matching fallback strategies
    /// </summary>
    public enum CUJITFallback
    {
        /// <summary>
        /// Prefer to compile ptx if exact binary match not found
        /// </summary>
        PTX = 0,

        /// <summary>
        /// Prefer to fall back to compatible binary code if exact binary match not found
        /// </summary>
        Binary
    }

    /// <summary>
    /// Online compiler options
    /// </summary>
    public enum CUJITOption
    {
        /// <summary>
        /// <para>Max number of registers that a thread may use.</para>
        /// <para>Option type: unsigned int</para>
        /// <para>Applies to: compiler only</para>
        /// </summary>
        MaxRegisters = 0,

        /// <summary>
        /// <para>IN: Specifies minimum number of threads per block to target compilation
        /// for</para>
        /// <para>OUT: Returns the number of threads the compiler actually targeted.
        /// This restricts the resource utilization fo the compiler (e.g. max
        /// registers) such that a block with the given number of threads should be
        /// able to launch based on register limitations. Note, this option does not
        /// currently take into account any other resource limitations, such as
        /// shared memory utilization.</para>
        /// <para>Option type: unsigned int</para>
        /// <para>Applies to: compiler only</para>
        /// </summary>
        ThreadsPerBlock,

        /// <summary>
        /// Returns a float value in the option of the wall clock time, in
        /// milliseconds, spent creating the cubin<para/>
        /// Option type: float
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        WallTime,

        /// <summary>
        /// <para>Pointer to a buffer in which to print any log messsages from PTXAS
        /// that are informational in nature (the buffer size is specified via
        /// option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)</para>
        /// <para>Option type: char*</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        InfoLogBuffer,

        /// <summary>
        /// <para>IN: Log buffer size in bytes.  Log messages will be capped at this size
        /// (including null terminator)</para>
        /// <para>OUT: Amount of log buffer filled with messages</para>
        /// <para>Option type: unsigned int</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        InfoLogBufferSizeBytes,

        /// <summary>
        /// <para>Pointer to a buffer in which to print any log messages from PTXAS that
        /// reflect errors (the buffer size is specified via option
        /// ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)</para>
        /// <para>Option type: char*</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        ErrorLogBuffer,

        /// <summary>
        /// <para>IN: Log buffer size in bytes.  Log messages will be capped at this size
        /// (including null terminator)</para>
        /// <para>OUT: Amount of log buffer filled with messages</para>
        /// <para>Option type: unsigned int</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        ErrorLogBufferSizeBytes,


        /// <summary>
        /// <para>Level of optimizations to apply to generated code (0 - 4), with 4
        /// being the default and highest level of optimizations.</para>
        /// <para>Option type: unsigned int</para>
        /// <para>Applies to: compiler only</para>
        /// </summary>
        OptimizationLevel,

        /// <summary>
        /// <para>No option value required. Determines the target based on the current
        /// attached context (default)</para>
        /// <para>Option type: No option value needed</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        TargetFromContext,

        /// <summary>
        /// <para>Target is chosen based on supplied ::CUjit_target_enum. This option cannot be
        /// used with cuLink* APIs as the linker requires exact matches.</para>
        /// <para>Option type: unsigned int for enumerated type ::CUjit_target_enum</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        Target,

        /// <summary>
        /// <para>Specifies choice of fallback strategy if matching cubin is not found.
        /// Choice is based on supplied ::CUjit_fallback_enum.</para>
        /// <para>Option type: unsigned int for enumerated type ::CUjit_fallback_enum</para>
        /// <para>Applies to: compiler only</para>
        /// </summary>
        FallbackStrategy,

        /// <summary>
        /// Specifies whether to create debug information in output (-g) <para/> (0: false, default)
        /// <para>Option type: int</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        GenerateDebugInfo,

        /// <summary>
        /// Generate verbose log messages <para/> (0: false, default)
        /// <para>Option type: int</para>
        /// <para>Applies to: compiler and linker</para>
        /// </summary>
        LogVerbose,

        /// <summary>
        /// Generate line number information (-lineinfo) <para/> (0: false, default)
        /// <para>Option type: int</para>
        /// <para>Applies to: compiler only</para>
        /// </summary>
        GenerateLineInfo,

        /// <summary>
        /// Specifies whether to enable caching explicitly (-dlcm)<para/>
        /// Choice is based on supplied ::CUjit_cacheMode_enum.
        /// <para>Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum</para>
        /// <para>Applies to: compiler only</para>
        /// </summary>
        JITCacheMode,

        /// <summary>
        /// The below jit options are used for internal purposes only, in this version of CUDA
        /// </summary>
        NewSM3XOpt,

        /// <summary>
        /// The below jit options are used for internal purposes only, in this version of CUDA
        /// </summary>
        FastCompile,

        /// <summary>
        /// Array of device symbol names that will be relocated to the corresponing
        /// host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.<para/>
        /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.<para/>
        /// When loding a device module, driver will relocate all encountered
        /// unresolved symbols to the host addresses.<para/>
        /// It is only allowed to register symbols that correspond to unresolved
        /// global variables.<para/>
        /// It is illegal to register the same device symbol at multiple addresses.<para/>
        /// Option type: const char **<para/>
        /// Applies to: dynamic linker only
        /// </summary>
        GlobalSymbolNames,

        /// <summary>
        /// Array of host addresses that will be used to relocate corresponding
        /// device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.<para/>
        /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.<para/>
        /// Option type: void **<para/>
        /// Applies to: dynamic linker only
        /// </summary>
        GlobalSymbolAddresses,

        /// <summary>
        /// Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
        /// ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.<para/>
        /// Option type: unsigned int<para/>
        /// Applies to: dynamic linker only
        /// </summary>
        GlobalSymbolCount
    }

    /// <summary>
    /// Online compilation targets
    /// </summary>
    public enum CUJITTarget
    {
        /// <summary>
        /// Compute device class 2.0
        /// </summary>
        Compute_20 = 20,

        /// <summary>
        /// Compute device class 2.1
        /// </summary>
        Compute_21 = 21,

        /// <summary>
        /// Compute device class 3.0
        /// </summary>
        Compute_30 = 30,

        /// <summary>
        /// Compute device class 3.2
        /// </summary>
        Compute_32 = 32,

        /// <summary>
        /// Compute device class 3.5
        /// </summary>
        Compute_35 = 35,

        /// <summary>
        /// Compute device class 3.7
        /// </summary>
        Compute_37 = 37,

        /// <summary>
        /// Compute device class 5.0
        /// </summary>
        Compute_50 = 50,

        /// <summary>
        /// Compute device class 5.2
        /// </summary>
        Compute_52 = 52,

        /// <summary>
        /// Compute device class 5.3
        /// </summary>
        /// 
        Compute_53 = 53,

        /// <summary>
        /// Compute device class 6.0
        /// </summary>
        Compute_60 = 60,

        /// <summary>
        /// Compute device class 6.1
        /// </summary>
        Compute_61 = 61,

        /// <summary>
        /// Compute device class 6.2.
        /// </summary>
        Compute_62 = 62,

        /// <summary>
        /// Compute device class 7.0.
        /// </summary>
        Compute_70 = 70,

        /// <summary>
        /// Compute device class 7.0.
        /// </summary>
        Compute_72 = 72,

        /// <summary>
        /// Compute device class 7.5.
        /// </summary>
        Compute_75 = 75,

        /// <summary>
        /// Compute device class 8.0.
        /// </summary>
        Compute_80 = 80,

        /// <summary>
        /// Compute device class 8.6.
        /// </summary>
        Compute_86 = 86
    }

    /// <summary>
    /// Online compilation optimization levels
    /// </summary>
    public enum CUJITOptimizationLevel
    {
        /// <summary>
        /// No optimization
        /// </summary>
        ZERO = 0,

        /// <summary>
        /// Optimization level 1
        /// </summary>
        ONE = 1,

        /// <summary>
        /// Optimization level 2
        /// </summary>
        TWO = 2,

        /// <summary>
        /// Optimization level 3
        /// </summary>
        THREE = 3,
        /// <summary>
        /// Best, Default
        /// </summary>
        FOUR = 4,
    }


    /// <summary>
    /// Caching modes for dlcm 
    /// </summary>
    public enum CUJITCacheMode
    {
        /// <summary>
        /// Compile with no -dlcm flag specified
        /// </summary>
        None = 0,
        /// <summary>
        /// Compile with L1 cache disabled
        /// </summary>
        Cg,
        /// <summary>
        /// Compile with L1 cache enabled
        /// </summary>
        Ca
    }

    /// <summary>
    /// Device code formats
    /// </summary>
    public enum CUJITInputType
    {
        /// <summary>
        /// Compiled device-class-specific device code
        /// <para>Applicable options: none</para>
        /// </summary>
        Cubin = 0,

        /// <summary>
        /// PTX source code
        /// <para>Applicable options: PTX compiler options</para>
        /// </summary>
        PTX,

        /// <summary>
        /// Bundle of multiple cubins and/or PTX of some device code
        /// <para>Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY</para>
        /// </summary>
        FatBinary,

        /// <summary>
        /// Host object with embedded device code
        /// <para>Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY</para>
        /// </summary>
        Object,

        /// <summary>
        /// Archive of host objects with embedded device code
        /// <para>Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY</para>
        /// </summary>
        Library
    }

    /// <summary>
    /// Array indices for cube faces
    /// </summary>
    public enum CUArrayCubemapFace
    {
        /// <summary>
        /// Positive X face of cubemap
        /// </summary>
        PositiveX = 0x00,

        /// <summary>
        /// Negative X face of cubemap
        /// </summary>
        NegativeX = 0x01,

        /// <summary>
        /// Positive Y face of cubemap 
        /// </summary>
        PositiveY = 0x02,

        /// <summary>
        /// Negative Y face of cubemap
        /// </summary>
        NegativeY = 0x03,

        /// <summary>
        /// Positive Z face of cubemap
        /// </summary>
        PositiveZ = 0x04,

        /// <summary>
        /// Negative Z face of cubemap
        /// </summary>
        NegativeZ = 0x05
    }

    /// <summary>
    /// Limits
    /// </summary>
    public enum CULimit
    {
        /// <summary>
        /// GPU thread stack size
        /// </summary>
        StackSize = 0,

        /// <summary>
        /// GPU printf FIFO size
        /// </summary>
        PrintfFIFOSize = 1,

        /// <summary>
        /// GPU malloc heap size
        /// </summary>
        MallocHeapSize = 2,

        /// <summary>
        /// GPU device runtime launch synchronize depth
        /// </summary>
        DevRuntimeSyncDepth = 3,

        /// <summary>
        /// GPU device runtime pending launch count
        /// </summary>
        DevRuntimePendingLaunchCount = 4,

        /// <summary>
        /// A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
        /// </summary>
        MaxL2FetchGranularity = 0x05,

        /// <summary>
        /// A size in bytes for L2 persisting lines cache size
        /// </summary>
        PersistingL2CacheSize = 0x06
    }

    /// <summary>
    /// Memory types
    /// </summary>
    public enum CUMemoryType : uint
    {
        /// <summary>
        /// Host memory
        /// </summary>
        Host = 0x01,

        /// <summary>
        /// Device memory
        /// </summary>
        Device = 0x02,

        /// <summary>
        /// Array memory
        /// </summary>
        Array = 0x03,

        /// <summary>
        /// Unified device or host memory
        /// </summary>
        Unified = 4
    }


    /// <summary>
    /// Resource types
    /// </summary>
    public enum CUResourceType
    {
        /// <summary>
        /// Array resoure
        /// </summary>
        Array = 0x00,
        /// <summary>
        /// Mipmapped array resource
        /// </summary>
        MipmappedArray = 0x01,
        /// <summary>
        /// Linear resource
        /// </summary>
        Linear = 0x02,
        /// <summary>
        /// Pitch 2D resource
        /// </summary>
        Pitch2D = 0x03
    }

    /// <summary>
    /// Error codes returned by CUDA driver API calls
    /// </summary>
    public enum CUResult
    {
        /// <summary>
        /// No errors
        /// </summary>
        Success = 0,

        /// <summary>
        /// Invalid value
        /// </summary>
        ErrorInvalidValue = 1,

        /// <summary>
        /// Out of memory
        /// </summary>
        ErrorOutOfMemory = 2,

        /// <summary>
        /// Driver not initialized
        /// </summary>
        ErrorNotInitialized = 3,

        /// <summary>
        /// Driver deinitialized
        /// </summary>
        ErrorDeinitialized = 4,

        /// <summary>
        /// This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools
        /// like visual profiler.
        /// </summary>
        ErrorProfilerDisabled = 5,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to attempt to enable/disable the profiling via ::cuProfilerStart or
        /// ::cuProfilerStop without initialization.
        /// </summary>
        [Obsolete("deprecated as of CUDA 5.0")]
        ErrorProfilerNotInitialized = 6,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to call cuProfilerStart() when profiling is already enabled.
        /// </summary>
        [Obsolete("deprecated as of CUDA 5.0")]
        ErrorProfilerAlreadyStarted = 7,

        /// <summary>
        /// This error return is deprecated as of CUDA 5.0. It is no longer an error
        /// to call cuProfilerStop() when profiling is already disabled.
        /// </summary>
        [Obsolete("deprecated as of CUDA 5.0")]
        ErrorProfilerAlreadyStopped = 8,

        /// <summary>
        /// This indicates that the CUDA driver that the application has loaded is a
        /// stub library. Applications that run with the stub rather than a real
        /// driver loaded will result in CUDA API returning this error.
        /// </summary>
        ErrorStubLibrary = 34,


        /// <summary>
        /// No CUDA-capable device available
        /// </summary>
        ErrorNoDevice = 100,

        /// <summary>
        /// Invalid device
        /// </summary>
        ErrorInvalidDevice = 101,

        /// <summary>
        /// This error indicates that the Grid license is not applied.
        /// </summary>
        DeviceNotLicensed = 102,



        /// <summary>
        /// Invalid kernel image
        /// </summary>
        ErrorInvalidImage = 200,

        /// <summary>
        /// Invalid context
        /// </summary>
        ErrorInvalidContext = 201,

        /// <summary>
        /// Context already current
        /// </summary>
        [Obsolete("This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to push the active context via cuCtxPushCurrent().")]
        ErrorContextAlreadyCurrent = 202,

        /// <summary>
        /// Map failed
        /// </summary>
        ErrorMapFailed = 205,

        /// <summary>
        /// Unmap failed
        /// </summary>
        ErrorUnmapFailed = 206,

        /// <summary>
        /// Array is mapped
        /// </summary>
        ErrorArrayIsMapped = 207,

        /// <summary>
        /// Already mapped
        /// </summary>
        ErrorAlreadyMapped = 208,

        /// <summary>
        /// No binary for GPU
        /// </summary>
        ErrorNoBinaryForGPU = 209,

        /// <summary>
        /// Already acquired
        /// </summary>
        ErrorAlreadyAcquired = 210,

        /// <summary>
        /// Not mapped
        /// </summary>
        ErrorNotMapped = 211,

        /// <summary>
        /// Mapped resource not available for access as an array
        /// </summary>
        ErrorNotMappedAsArray = 212,

        /// <summary>
        /// Mapped resource not available for access as a pointer
        /// </summary>
        ErrorNotMappedAsPointer = 213,

        /// <summary>
        /// Uncorrectable ECC error detected
        /// </summary>
        ErrorECCUncorrectable = 214,

        /// <summary>
        /// CULimit not supported by device
        /// </summary>
        ErrorUnsupportedLimit = 215,

        /// <summary>
        /// This indicates that the <see cref="CUcontext"/> passed to the API call can
        /// only be bound to a single CPU thread at a time but is already 
        /// bound to a CPU thread.
        /// </summary>
        ErrorContextAlreadyInUse = 216,

        /// <summary>
        /// This indicates that peer access is not supported across the given devices.
        /// </summary>
        ErrorPeerAccessUnsupported = 217,

        /// <summary>
        /// This indicates that a PTX JIT compilation failed.
        /// </summary>
        ErrorInvalidPtx = 218,

        /// <summary>
        /// This indicates an error with OpenGL or DirectX context.
        /// </summary>
        ErrorInvalidGraphicsContext = 219,

        /// <summary>
        /// This indicates that an uncorrectable NVLink error was detected during the execution.
        /// </summary>
        NVLinkUncorrectable = 220,

        /// <summary>
        /// This indicates that the PTX JIT compiler library was not found.
        /// </summary>
        JITCompilerNotFound = 221,

        /// <summary>
        /// This indicates that the provided PTX was compiled with an unsupported toolchain.
        /// </summary>
        UnsupportedPTXVersion = 222,

        /// <summary>
        /// This indicates that the PTX JIT compilation was disabled.
        /// </summary>
        JITCompilationDisabled = 223,

        /// <summary>
        /// Invalid source
        /// </summary>
        ErrorInvalidSource = 300,

        /// <summary>
        /// File not found
        /// </summary>
        ErrorFileNotFound = 301,

        /// <summary>
        /// Link to a shared object failed to resolve
        /// </summary>
        ErrorSharedObjectSymbolNotFound = 302,

        /// <summary>
        /// Shared object initialization failed
        /// </summary>
        ErrorSharedObjectInitFailed = 303,

        /// <summary>
        /// OS call failed
        /// </summary>
        ErrorOperatingSystem = 304,

        /// <summary>
        /// Invalid handle
        /// </summary>
        ErrorInvalidHandle = 400,

        /// <summary>
        /// This indicates that a resource required by the API call is not in a
        /// valid state to perform the requested operation.
        /// </summary>
        ErrorIllegalState = 401,

        /// <summary>
        /// Not found
        /// </summary>
        ErrorNotFound = 500,


        /// <summary>
        /// CUDA not ready
        /// </summary>
        ErrorNotReady = 600,


        /// <summary>
        /// While executing a kernel, the device encountered a
        /// load or store instruction on an invalid memory address.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorIllegalAddress = 700,

        /// <summary>
        /// Launch exceeded resources
        /// </summary>
        ErrorLaunchOutOfResources = 701,

        /// <summary>
        /// This indicates that the device kernel took too long to execute. This can
        /// only occur if timeouts are enabled - see the device attribute
        /// ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorLaunchTimeout = 702,

        /// <summary>
        /// Launch with incompatible texturing
        /// </summary>
        ErrorLaunchIncompatibleTexturing = 703,

        /// <summary>
        /// This error indicates that a call to <see cref="DriverAPINativeMethods.CudaPeerAccess.cuCtxEnablePeerAccess"/> is
        /// trying to re-enable peer access to a context which has already
        /// had peer access to it enabled.
        /// </summary>
        ErrorPeerAccessAlreadyEnabled = 704,

        /// <summary>
        /// This error indicates that <see cref="DriverAPINativeMethods.CudaPeerAccess.cuCtxDisablePeerAccess"/> is 
        /// trying to disable peer access which has not been enabled yet 
        /// via <see cref="DriverAPINativeMethods.CudaPeerAccess.cuCtxEnablePeerAccess"/>. 
        /// </summary>
        ErrorPeerAccessNotEnabled = 705,

        /// <summary>
        /// This error indicates that the primary context for the specified device
        /// has already been initialized.
        /// </summary>
        ErrorPrimaryContextActice = 708,

        /// <summary>
        /// This error indicates that the context current to the calling thread
        /// has been destroyed using <see cref="DriverAPINativeMethods.ContextManagement.cuCtxDestroy_v2"/>, or is a primary context which
        /// has not yet been initialized. 
        /// </summary>
        ErrorContextIsDestroyed = 709,

        /// <summary>
        /// A device-side assert triggered during kernel execution. The context
        /// cannot be used anymore, and must be destroyed. All existing device 
        /// memory allocations from this context are invalid and must be 
        /// reconstructed if the program is to continue using CUDA.
        /// </summary>
        ErrorAssert = 710,

        /// <summary>
        /// This error indicates that the hardware resources required to enable
        /// peer access have been exhausted for one or more of the devices 
        /// passed to ::cuCtxEnablePeerAccess().
        /// </summary>
        ErrorTooManyPeers = 711,

        /// <summary>
        /// This error indicates that the memory range passed to ::cuMemHostRegister()
        /// has already been registered.
        /// </summary>
        ErrorHostMemoryAlreadyRegistered = 712,

        /// <summary>
        /// This error indicates that the pointer passed to ::cuMemHostUnregister()
        /// does not correspond to any currently registered memory region.
        /// </summary>
        ErrorHostMemoryNotRegistered = 713,

        /// <summary>
        /// While executing a kernel, the device encountered a stack error.
        /// This can be due to stack corruption or exceeding the stack size limit.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorHardwareStackError = 714,

        /// <summary>
        /// While executing a kernel, the device encountered an illegal instruction.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorIllegalInstruction = 715,

        /// <summary>
        /// While executing a kernel, the device encountered a load or store instruction
        /// on a memory address which is not aligned.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorMisalignedAddress = 716,

        /// <summary>
        /// While executing a kernel, the device encountered an instruction
        /// which can only operate on memory locations in certain address spaces
        /// (global, shared, or local), but was supplied a memory address not
        /// belonging to an allowed address space.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorInvalidAddressSpace = 717,

        /// <summary>
        /// While executing a kernel, the device program counter wrapped its address space.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorInvalidPC = 718,

        /// <summary>
        /// An exception occurred on the device while executing a kernel. Common
        /// causes include dereferencing an invalid device pointer and accessing
        /// out of bounds shared memory. This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorLaunchFailed = 719,

        /// <summary>
        /// This error indicates that the number of blocks launched per grid for a kernel that was
        /// launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
        /// exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
        /// or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
        /// as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
        /// </summary>
        ErrorCooperativeLaunchTooLarge = 720,




        //Removed in update CUDA version 3.1 -> 3.2
        ///// <summary>
        ///// Attempted to retrieve 64-bit pointer via 32-bit API function
        ///// </summary>
        //ErrorPointerIs64Bit = 800,

        ///// <summary>
        ///// Attempted to retrieve 64-bit size via 32-bit API function
        ///// </summary>
        //ErrorSizeIs64Bit = 801, 

        /// <summary>
        /// This error indicates that the attempted operation is not permitted.
        /// </summary>
        ErrorNotPermitted = 800,

        /// <summary>
        /// This error indicates that the attempted operation is not supported
        /// on the current system or device.
        /// </summary>
        ErrorNotSupported = 801,

        /// <summary>
        /// This error indicates that the system is not yet ready to start any CUDA
        /// work.  To continue using CUDA, verify the system configuration is in a
        /// valid state and all required driver daemons are actively running.
        /// </summary>
        ErrorSystemNotReady = 802,

        /// <summary>
        /// This error indicates that there is a mismatch between the versions of
        /// the display driver and the CUDA driver. Refer to the compatibility documentation
        /// for supported versions.
        /// </summary>
        ErrorSystemDriverMismatch = 803,

        /// <summary>
        /// This error indicates that the system was upgraded to run with forward compatibility
        /// but the visible hardware detected by CUDA does not support this configuration.
        /// Refer to the compatibility documentation for the supported hardware matrix or ensure
        /// that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
        /// environment variable.
        /// </summary>
        ErrorCompatNotSupportedOnDevice = 804,

        /// <summary>
        /// This error indicates that the operation is not permitted when the stream is capturing.
        /// </summary>
        ErrorStreamCaptureUnsupported = 900,

        /// <summary>
        /// This error indicates that the current capture sequence on the stream
        /// has been invalidated due to a previous error.
        /// </summary>
        ErrorStreamCaptureInvalidated = 901,

        /// <summary>
        /// This error indicates that the operation would have resulted in a merge of two independent capture sequences.
        /// </summary>
        ErrorStreamCaptureMerge = 902,

        /// <summary>
        /// This error indicates that the capture was not initiated in this stream.
        /// </summary>
        ErrorStreamCaptureUnmatched = 903,

        /// <summary>
        /// This error indicates that the capture sequence contains a fork that was not joined to the primary stream.
        /// </summary>
        ErrorStreamCaptureUnjoined = 904,

        /// <summary>
        /// This error indicates that a dependency would have been created which
        /// crosses the capture sequence boundary. Only implicit in-stream ordering
        /// dependencies are allowed to cross the boundary.
        /// </summary>
        ErrorStreamCaptureIsolation = 905,

        /// <summary>
        /// This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.
        /// </summary>
        ErrorStreamCaptureImplicit = 906,

        /// <summary>
        /// This error indicates that the operation is not permitted on an event which
        /// was last recorded in a capturing stream.
        /// </summary>
        ErrorCapturedEvent = 907,

        /// <summary>
        /// A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED
        /// argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a
        /// different thread.
        /// </summary>
        ErrorStreamCaptureWrongThread = 908,

        /// <summary>
        /// This error indicates that the timeout specified for the wait operation has lapsed.
        /// </summary>
        ErrorTimeOut = 909,

        /// <summary>
        /// This error indicates that the graph update was not performed because it included 
        /// changes which violated constraints specific to instantiated graph update.
        /// </summary>
        ErrorGraphExecUpdateFailure = 910,

        /// <summary>
        /// Unknown error
        /// </summary>
        ErrorUnknown = 999,
    }

    /// <summary>
    /// P2P Attributes
    /// </summary>
    public enum CUdevice_P2PAttribute
    {
        /// <summary>
        /// A relative value indicating the performance of the link between two devices
        /// </summary>
        PerformanceRank = 0x01,
        /// <summary>
        /// P2P Access is enable
        /// </summary>
        AccessSupported = 0x02,
        /// <summary>
        /// Atomic operation over the link supported
        /// </summary>
        NativeAtomicSupported = 0x03,
        /// <summary>
        /// \deprecated use CudaArrayAccessAccessSupported instead
        /// </summary>
        [Obsolete("use CudaArrayAccessAccessSupported instead")]
        AccessAccessSupported = 0x04,
        /// <summary>
        /// Accessing CUDA arrays over the link supported
        /// </summary>
        CudaArrayAccessAccessSupported = 0x04

    }

    /// <summary>
    /// CUTexRefSetArrayFlags
    /// </summary>
    public enum CUTexRefSetArrayFlags
    {
        /// <summary>
        /// 
        /// </summary>
        None = 0,
        /// <summary>
        /// Override the texref format with a format inferred from the array.
        /// <para/>Flag for <see cref="DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray"/>.
        /// </summary>
        OverrideFormat = 1
    }

    /// <summary>
    /// CUParameterTexRef
    /// </summary>
    public enum CUParameterTexRef
    {
        /// <summary>
        /// For texture references loaded into the module, use default texunit from texture reference.
        /// </summary>
        Default = -1
    }

    /// <summary>
    /// CUSurfRefSetFlags
    /// </summary>
    public enum CUSurfRefSetFlags
    {
        /// <summary>
        /// Currently no CUSurfRefSetFlags flags are defined.
        /// </summary>
        None = 0
    }

    /// <summary>
    /// Pointer information
    /// </summary>
    public enum CUPointerAttribute
    {
        /// <summary>
        /// The <see cref="CUcontext"/> on which a pointer was allocated or registered
        /// </summary>
        Context = 1,

        /// <summary>
        /// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
        /// </summary>
        MemoryType = 2,

        /// <summary>
        /// The address at which a pointer's memory may be accessed on the device 
        /// </summary>
        DevicePointer = 3,

        /// <summary>
        /// The address at which a pointer's memory may be accessed on the host 
        /// </summary>
        HostPointer = 4,

        /// <summary>
        /// A pair of tokens for use with the nv-p2p.h Linux kernel interface
        /// </summary>
        P2PTokens = 5,

        /// <summary>
        /// Synchronize every synchronous memory operation initiated on this region
        /// </summary>
        SyncMemops = 6,

        /// <summary>
        /// A process-wide unique ID for an allocated memory region
        /// </summary>
        BufferID = 7,

        /// <summary>
        /// Indicates if the pointer points to managed memory
        /// </summary>
        IsManaged = 8,

        /// <summary>
        /// A device ordinal of a device on which a pointer was allocated or registered
        /// </summary>
        DeviceOrdinal = 9,

        /// <summary>
        /// 1 if this pointer maps to an allocation that is suitable for ::cudaIpcGetMemHandle, 0 otherwise
        /// </summary>
        IsLegacyCudaIPCCapable = 10,

        /// <summary>
        /// Starting address for this requested pointer
        /// </summary>
        RangeStartAddr = 11,

        /// <summary>
        /// Size of the address range for this requested pointer
        /// </summary>
        RangeSize = 12,

        /// <summary>
        /// 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise
        /// </summary>
        Mapped = 13,

        /// <summary>
        /// Bitmask of allowed ::CUmemAllocationHandleType for this allocation
        /// </summary>
        AllowedHandleTypes = 14,

        /// <summary>
        /// 1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API
        /// </summary>
        IsGPUDirectRDMACapable = 15,

        /// <summary>
        /// Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given
        /// </summary>
        AccessFlags = 16,

        /// <summary>
        /// Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise returns NULL.
        /// </summary>
        MempoolHandle = 17
    }

    /// <summary>
    /// CUDA devices corresponding to a D3D11, D3D10 or D3D9 device
    /// </summary>
    public enum CUd3dXDeviceList
    {
        /// <summary>
        /// The CUDA devices for all GPUs used by a D3D11 device.
        /// </summary>
        All = 0x01,
        /// <summary>
        /// The CUDA devices for the GPUs used by a D3D11 device in its currently rendering frame (in SLI).
        /// </summary>
        CurrentFrame = 0x02,
        /// <summary>
        /// The CUDA devices for the GPUs to be used by a D3D11 device in the next frame (in SLI).
        /// </summary>
        NextFrame = 0x03
    }

    /// <summary>
    /// CUDA devices corresponding to an OpenGL device.
    /// </summary>
    public enum CUGLDeviceList
    {
        /// <summary>
        /// The CUDA devices for all GPUs used by the current OpenGL context
        /// </summary>
        All = 0x01,
        /// <summary>
        /// The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame
        /// </summary>
        CurrentFrame = 0x02,
        /// <summary>
        /// The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame
        /// </summary>
        NextFrame = 0x03,
    }

    /// <summary>
    /// Shared memory configurations
    /// </summary>
    public enum CUsharedconfig
    {
        /// <summary>
        /// set default shared memory bank size 
        /// </summary>
        DefaultBankSize = 0x00,
        /// <summary>
        /// set shared memory bank width to four bytes
        /// </summary>
        FourByteBankSize = 0x01,
        /// <summary>
        /// set shared memory bank width to eight bytes
        /// </summary>
        EightByteBankSize = 0x02
    }

    /// <summary>
    /// CUipcMem_flags
    /// </summary>
    public enum CUipcMem_flags
    {
        /// <summary>
        /// Automatically enable peer access between remote devices as needed
        /// </summary>
        LazyEnablePeerAccess = 0x1
    }

    /// <summary>
    /// Resource view format
    /// </summary>
    public enum CUresourceViewFormat
    {
        /// <summary>
        /// No resource view format (use underlying resource format)
        /// </summary>
        None = 0x00,
        /// <summary>
        /// 1 channel unsigned 8-bit integers
        /// </summary>
        Uint1X8 = 0x01,
        /// <summary>
        /// 2 channel unsigned 8-bit integers
        /// </summary>
        Uint2X8 = 0x02,
        /// <summary>
        /// 4 channel unsigned 8-bit integers
        /// </summary>
        Uint4X8 = 0x03,
        /// <summary>
        /// 1 channel signed 8-bit integers
        /// </summary>
        Sint1X8 = 0x04,
        /// <summary>
        /// 2 channel signed 8-bit integers
        /// </summary>
        Sint2X8 = 0x05,
        /// <summary>
        /// 4 channel signed 8-bit integers
        /// </summary>
        Sint4X8 = 0x06,
        /// <summary>
        /// 1 channel unsigned 16-bit integers
        /// </summary>
        Uint1X16 = 0x07,
        /// <summary>
        /// 2 channel unsigned 16-bit integers
        /// </summary>
        Uint2X16 = 0x08,
        /// <summary>
        /// 4 channel unsigned 16-bit integers
        /// </summary>
        Uint4X16 = 0x09,
        /// <summary>
        /// 1 channel signed 16-bit integers
        /// </summary>
        Sint1X16 = 0x0a,
        /// <summary>
        /// 2 channel signed 16-bit integers
        /// </summary>
        Sint2X16 = 0x0b,
        /// <summary>
        /// 4 channel signed 16-bit integers
        /// </summary>
        Sint4X16 = 0x0c,
        /// <summary>
        /// 1 channel unsigned 32-bit integers
        /// </summary>
        Uint1X32 = 0x0d,
        /// <summary>
        /// 2 channel unsigned 32-bit integers
        /// </summary>
        Uint2X32 = 0x0e,
        /// <summary>
        /// 4 channel unsigned 32-bit integers 
        /// </summary>
        Uint4X32 = 0x0f,
        /// <summary>
        /// 1 channel signed 32-bit integers
        /// </summary>
        Sint1X32 = 0x10,
        /// <summary>
        /// 2 channel signed 32-bit integers
        /// </summary>
        Sint2X32 = 0x11,
        /// <summary>
        /// 4 channel signed 32-bit integers
        /// </summary>
        Sint4X32 = 0x12,
        /// <summary>
        /// 1 channel 16-bit floating point
        /// </summary>
        Float1X16 = 0x13,
        /// <summary>
        /// 2 channel 16-bit floating point
        /// </summary>
        Float2X16 = 0x14,
        /// <summary>
        /// 4 channel 16-bit floating point
        /// </summary>
        Float4X16 = 0x15,
        /// <summary>
        /// 1 channel 32-bit floating point
        /// </summary>
        Float1X32 = 0x16,
        /// <summary>
        /// 2 channel 32-bit floating point
        /// </summary>
        Float2X32 = 0x17,
        /// <summary>
        /// 4 channel 32-bit floating point
        /// </summary>
        Float4X32 = 0x18,
        /// <summary>
        /// Block compressed 1 
        /// </summary>
        UnsignedBC1 = 0x19,
        /// <summary>
        /// Block compressed 2
        /// </summary>
        UnsignedBC2 = 0x1a,
        /// <summary>
        /// Block compressed 3 
        /// </summary>
        UnsignedBC3 = 0x1b,
        /// <summary>
        /// Block compressed 4 unsigned
        /// </summary>
        UnsignedBC4 = 0x1c,
        /// <summary>
        /// Block compressed 4 signed 
        /// </summary>
        SignedBC4 = 0x1d,
        /// <summary>
        /// Block compressed 5 unsigned
        /// </summary>
        UnsignedBC5 = 0x1e,
        /// <summary>
        /// Block compressed 5 signed
        /// </summary>
        SignedBC5 = 0x1f,
        /// <summary>
        /// Block compressed 6 unsigned half-float
        /// </summary>
        UnsignedBC6H = 0x20,
        /// <summary>
        /// Block compressed 6 signed half-float
        /// </summary>
        SignedBC6H = 0x21,
        /// <summary>
        /// Block compressed 7 
        /// </summary>
        UnsignedBC7 = 0x22
    }

    /// <summary>
    /// Profiler Output Modes
    /// </summary>
    public enum CUoutputMode
    {
        /// <summary>
        /// Output mode Key-Value pair format.
        /// </summary>
        KeyValuePair = 0x00,
        /// <summary>
        /// Output mode Comma separated values format.
        /// </summary>
        CSV = 0x01
    }

    /// <summary>
    /// CUDA Mem Attach Flags
    /// </summary>
    public enum CUmemAttach_flags
    {
        /// <summary>
        /// Memory can be accessed by any stream on any device
        /// </summary>
        Global = 1,

        /// <summary>
        /// Memory cannot be accessed by any stream on any device
        /// </summary>
        Host = 2,

        /// <summary>
        /// Memory can only be accessed by a single stream on the associated device
        /// </summary>
        Single = 4
    }



    /// <summary>
    /// Occupancy calculator flag
    /// </summary>
    public enum CUoccupancy_flags
    {
        /// <summary>
        /// Default behavior
        /// </summary>
        Default = 0,

        /// <summary>
        /// Assume global caching is enabled and cannot be automatically turned off
        /// </summary>
        DisableCachingOverride = 1
    }

    //These library types do not really fit in here, but all libraries depend on managedCuda...
    /// <summary>
    /// cudaDataType
    /// </summary>
    public enum cudaDataType
    {
        /// <summary>
        /// 16 bit real 
        /// </summary>
        CUDA_R_16F = 2,

        /// <summary>
        /// 16 bit complex
        /// </summary>
        CUDA_C_16F = 6,

        /// <summary>
        /// 32 bit real
        /// </summary>
        CUDA_R_32F = 0,

        /// <summary>
        /// 32 bit complex
        /// </summary>
        CUDA_C_32F = 4,

        /// <summary>
        /// 64 bit real
        /// </summary>
        CUDA_R_64F = 1,

        /// <summary>
        /// 64 bit complex
        /// </summary>
        CUDA_C_64F = 5,

        /// <summary>
        /// 8 bit real as a signed integer 
        /// </summary>
        CUDA_R_8I = 3,

        /// <summary>
        /// 8 bit complex as a pair of signed integers
        /// </summary>
        CUDA_C_8I = 7,

        /// <summary>
        /// 8 bit real as a signed integer 
        /// </summary>
        CUDA_R_8U = 8,

        /// <summary>
        /// 8 bit complex as a pair of signed integers
        /// </summary>
        CUDA_C_8U = 9,

        /// <summary>
        /// real as a nv_bfloat16
        /// </summary>
        CUDA_R_16BF = 14,
        /// <summary>
        /// complex as a pair of nv_bfloat16 numbers
        /// </summary>
        CUDA_C_16BF = 15,
        /// <summary>
        /// real as a signed 4-bit int
        /// </summary>
        CUDA_R_4I = 16,
        /// <summary>
        /// complex as a pair of signed 4-bit int numbers
        /// </summary>
        CUDA_C_4I = 17,
        /// <summary>
        /// real as a unsigned 4-bit int
        /// </summary>
        CUDA_R_4U = 18,
        /// <summary>
        /// complex as a pair of unsigned 4-bit int numbers
        /// </summary>
        CUDA_C_4U = 19,
        /// <summary>
        /// real as a signed 16-bit int 
        /// </summary>
        CUDA_R_16I = 20,
        /// <summary>
        /// complex as a pair of signed 16-bit int numbers
        /// </summary>
        CUDA_C_16I = 21,
        /// <summary>
        /// real as a unsigned 16-bit int
        /// </summary>
        CUDA_R_16U = 22,
        /// <summary>
        /// complex as a pair of unsigned 16-bit int numbers
        /// </summary>
        CUDA_C_16U = 23,
        /// <summary>
        /// real as a signed 32-bit int
        /// </summary>
        CUDA_R_32I = 10,
        /// <summary>
        /// complex as a pair of signed 32-bit int numbers
        /// </summary>
        CUDA_C_32I = 11,
        /// <summary>
        /// real as a unsigned 32-bit int
        /// </summary>
        CUDA_R_32U = 12,
        /// <summary>
        /// complex as a pair of unsigned 32-bit int numbers
        /// </summary>
        CUDA_C_32U = 13,
        /// <summary>
        /// real as a signed 64-bit int
        /// </summary>
        CUDA_R_64I = 24,
        /// <summary>
        /// complex as a pair of signed 64-bit int numbers 
        /// </summary>
        CUDA_C_64I = 25,
        /// <summary>
        /// real as a unsigned 64-bit int
        /// </summary>
        CUDA_R_64U = 26,
        /// <summary>
        /// complex as a pair of unsigned 64-bit int numbers
        /// </summary>
        CUDA_C_64U = 27


    }


    /// <summary/>
    public enum libraryPropertyType
    {
        /// <summary/>
        MAJOR_VERSION,
        /// <summary/>
        MINOR_VERSION,
        /// <summary/>
        PATCH_LEVEL
    }

    /// <summary>
    /// Operations for ::cuStreamBatchMemOp
    /// </summary>
    public enum CUstreamBatchMemOpType
    {
        /// <summary>
        /// Represents a ::cuStreamWaitValue32 operation
        /// </summary>
        WaitValue32 = 1,
        /// <summary>
        /// Represents a ::cuStreamWriteValue32 operation
        /// </summary>
        WriteValue32 = 2,
        /// <summary>
        /// Represents a ::cuStreamWaitValue64 operation
        /// </summary>
        WaitValue64 = 4,
        /// <summary>
        /// Represents a ::cuStreamWriteValue64 operation
        /// </summary>
        WriteValue64 = 5,
        /// <summary>
        /// This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a standalone operation.
        /// </summary>
        FlushRemoteWrites = 3
    }

    /// <summary>
    /// 
    /// </summary>
    public enum CUmem_range_attribute
    {
        /// <summary>
        /// Whether the range will mostly be read and only occassionally be written to
        /// </summary>
        ReadMostly = 1,
        /// <summary>
        /// The preferred location of the range
        /// </summary>
        PreferredLocation = 2,
        /// <summary>
        /// Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device
        /// </summary>
        AccessedBy = 3,
        /// <summary>
        /// The last location to which the range was prefetched
        /// </summary>
        LastPrefetchLocation = 4
    }

    /// <summary>
    /// Shared memory carveout configurations
    /// </summary>
    public enum CUshared_carveout
    {
        /// <summary>
        /// no preference for shared memory or L1 (default)
        /// </summary>
        Default = -1,
        /// <summary>
        /// prefer maximum available shared memory, minimum L1 cache
        /// </summary>
        MaxShared = 100,
        /// <summary>
        /// prefer maximum available L1 cache, minimum shared memory
        /// </summary>
        MaxL1 = 0
    }

    /// <summary>
    /// Graph node types
    /// </summary>
    public enum CUgraphNodeType
    {
        /// <summary>
        /// GPU kernel node
        /// </summary>
        Kernel = 0,
        /// <summary>
        /// Memcpy node
        /// </summary>
        Memcpy = 1,
        /// <summary>
        /// Memset node
        /// </summary>
        Memset = 2,
        /// <summary>
        /// Host (executable) node
        /// </summary>
        Host = 3,
        /// <summary>
        /// Node which executes an embedded graph
        /// </summary>
        Graph = 4,
        /// <summary>
        /// Empty (no-op) node
        /// </summary>
        Empty = 5,
        /// <summary>
        /// External event wait node
        /// </summary>
        WaitEvent = 6,
        /// <summary>
        /// External event record node
        /// </summary>
        EventRecord = 7,
        /// <summary>
        /// External semaphore signal node
        /// </summary>
        ExtSemasSignal = 8,
        /// <summary>
        /// External semaphore wait node
        /// </summary>
        ExtSemasWait = 9,
        /// <summary>
        /// 
        /// </summary>
        CU_GRAPH_NODE_TYPE_COUNT
    }

    /// <summary>
    /// Possible stream capture statuses returned by ::cuStreamIsCapturing
    /// </summary>
    public enum CUstreamCaptureStatus
    {
        /// <summary>
        /// Stream is not capturing
        /// </summary>
        None = 0,
        /// <summary>
        /// Stream is actively capturing
        /// </summary>
        Active = 1,
        /// <summary>
        /// Stream is part of a capture sequence that has been invalidated, but not terminated
        /// </summary>
        Invalidated = 2
    }

    /// <summary>
    /// Possible modes for stream capture thread interactions. For more details see ::cuStreamBeginCapture and ::cuThreadExchangeStreamCaptureMode
    /// </summary>
    public enum CUstreamCaptureMode
    {
        /// <summary>
        /// 
        /// </summary>
        Global = 0,
        /// <summary>
        /// 
        /// </summary>
        Local = 1,
        /// <summary>
        /// 
        /// </summary>
        Relaxed = 2
    }

    /// <summary>
    /// External memory handle types
    /// </summary>
    public enum CUexternalMemoryHandleType
    {
        /// <summary>
        /// Handle is an opaque file descriptor
        /// </summary>
        OpaqueFD = 1,
        /// <summary>
        /// Handle is an opaque shared NT handle
        /// </summary>
        OpaqueWin32 = 2,
        /// <summary>
        /// Handle is an opaque, globally shared handle
        /// </summary>
        OpaqueWin32KMT = 3,
        /// <summary>
        /// Handle is a D3D12 heap object
        /// </summary>
        D3D12Heap = 4,
        /// <summary>
        /// Handle is a D3D12 committed resource
        /// </summary>
        D3D12Resource = 5,
        /// <summary>
        /// Handle is a shared NT handle to a D3D11 resource
        /// </summary>
        D3D11Resource = 6,
        /// <summary>
        /// Handle is a globally shared handle to a D3D11 resource
        /// </summary>
        D3D11ResourceKMT = 7,
        /// <summary>
        /// Handle is an NvSciBuf object
        /// </summary>
        NvSciBuf = 8
    }



    /// <summary>
    /// External semaphore handle types
    /// </summary>
    public enum CUexternalSemaphoreHandleType
    {
        /// <summary>
        /// Handle is an opaque file descriptor
        /// </summary>
        OpaqueFD = 1,
        /// <summary>
        /// Handle is an opaque shared NT handle
        /// </summary>
        OpaqueWin32 = 2,
        /// <summary>
        /// Handle is an opaque, globally shared handle
        /// </summary>
        OpaqueWin32KMT = 3,
        /// <summary>
        /// Handle is a shared NT handle referencing a D3D12 fence object
        /// </summary>
        D3D12DFence = 4,
        /// <summary>
        /// Handle is a shared NT handle referencing a D3D11 fence object
        /// </summary>
        D3D11Fence = 5,
        /// <summary>
        /// Opaque handle to NvSciSync Object
        /// </summary>
        NvSciSync = 6,
        /// <summary>
        /// Handle is a shared NT handle referencing a D3D11 keyed mutex object
        /// </summary>
        D3D11KeyedMutex = 7,
        /// <summary>
        /// Handle is a globally shared handle referencing a D3D11 keyed mutex object
        /// </summary>
        D3D11KeyedMutexKMT = 8,
        /// <summary>
        /// Handle is an opaque file descriptor referencing a timeline semaphore
        /// </summary>
        TimelineSemaphoreFD = 9,
        /// <summary>
        /// Handle is an opaque shared NT handle referencing a timeline semaphore
        /// </summary>
        TimelineSemaphoreWin32 = 10
    }


    /// <summary>
    /// Specifies the type of location
    /// </summary>
    public enum CUmemLocationType
    {
        /// <summary>
        /// 
        /// </summary>
        Invalid = 0x0,
        /// <summary>
        /// Location is a device location, thus id is a device ordinal
        /// </summary>
        Device = 0x1
    }

    /// <summary>
    /// Defines the allocation types available
    /// </summary>
    public enum CUmemAllocationType
    {
        /// <summary>
        /// 
        /// </summary>
        Invalid = 0x0,
        /// <summary>
        /// This allocation type is 'pinned', i.e. cannot migrate from its current
        /// location while the application is actively using it
        /// </summary>
        Pinned = 0x1
    }

    /// <summary>
    /// 
    /// </summary>
    public enum CUgraphExecUpdateResult
    {
        /// <summary>
        /// The update succeeded
        /// </summary>
        Success = 0x0,
        /// <summary>
        /// The update failed for an unexpected reason which is described in the return value of the function
        /// </summary>
        Error = 0x1,
        /// <summary>
        /// The update failed because the topology changed
        /// </summary>
        ErrorTopologyChanged = 0x2,
        /// <summary>
        /// The update failed because a node type changed
        /// </summary>
        ErrorNodeTypeChanged = 0x3,
        /// <summary>
        /// The update failed because the function of a kernel node changed
        /// </summary>
        ErrorFunctionChanged = 0x4,
        /// <summary>
        /// The update failed because the parameters changed in a way that is not supported
        /// </summary>
        ErrorParametersChanged = 0x5,
        /// <summary>
        /// The update failed because something about the node is not supported
        /// </summary>
        ErrorNotSupported = 0x6,
        /// <summary>
        /// The update failed because the function of a kernel node changed in an unsupported way
        /// </summary>
        ErrorUnsupportedFunctionChange = 0x7
    }

    /// <summary>
    /// Specifies performance hint with ::CUaccessPolicyWindow for hitProp and missProp members
    /// </summary>
    public enum CUaccessProperty
    {
        /// <summary>
        /// Normal cache persistence.
        /// </summary>
        Normal = 0,
        /// <summary>
        /// Streaming access is less likely to persit from cache.
        /// </summary>
        Streaming = 1,
        /// <summary>
        /// Persisting access is more likely to persist in cache.
        /// </summary>
        Persisting = 2
    }

    /// <summary>
    /// 
    /// </summary>
    public enum CUsynchronizationPolicy
    {
        /// <summary/>
        Auto = 1,
        /// <summary/>
        Spin = 2,
        /// <summary/>
        Yield = 3,
        /// <summary/>
        BlockingSync = 4
    }

    /// <summary>
    /// Graph kernel node Attributes 
    /// </summary>
    public enum CUkernelNodeAttrID
    {
        /// <summary>
        /// Identifier for ::CUkernelNodeAttrValue::accessPolicyWindow.
        /// </summary>
        AccessPolicyWindow = 1,
        /// <summary>
        /// Allows a kernel node to be cooperative (see ::cuLaunchCooperativeKernel).
        /// </summary>
        Cooperative = 2
    }

    /// <summary>
    /// 
    /// </summary>
    public enum CUstreamAttrID
    {
        /// <summary>
        /// Identifier for ::CUstreamAttrValue::accessPolicyWindow.
        /// </summary>
        AccessPolicyWindow = 1,
        /// <summary>
        /// ::CUsynchronizationPolicy for work queued up in this stream
        /// </summary>
        SynchronizationPolicy = 3
    }

    /// <summary>
    /// Specifies compression attribute for an allocation.
    /// </summary>
    public enum CUmemAllocationCompType : byte
    {
        /// <summary>
        /// Allocating non-compressible memory
        /// </summary>
        None = 0x0,
        /// <summary>
        /// Allocating  compressible memory
        /// </summary>
        Generic = 0x1
    }

    /// <summary>
    /// 
    /// </summary>
    public enum CUmemCreateUsage : ushort
    {
        /// <summary>
        /// 
        /// </summary>
        None = 0x0,
        /// <summary>
        /// This flag if set indicates that the memory will be used as a tile pool.
        /// </summary>
        TilePool = 0x1
    }

    /// <summary>
    /// Access flags that specify the level of access the current context's device has
    /// on the memory referenced.
    /// </summary>
    public enum CudaPointerAttributeAccessFlags
    {
        /// <summary>
        /// No access, meaning the device cannot access this memory at all, thus must be staged through accessible memory in order to complete certain operations
        /// </summary>
        None = 0x0,
        /// <summary>
        /// Read-only access, meaning writes to this memory are considered invalid accesses and thus return error in that case. 
        /// </summary>
        Read = 0x1,
        /// <summary>
        /// Read-write access, the device has full read-write access to the memory
        /// </summary>
        ReadWrite = 0x3
    }

    /// <summary>
    /// Sparse subresource types
    /// </summary>
    public enum CUarraySparseSubresourceType
    {
        /// <summary>
        /// 
        /// </summary>
        SparseLevel = 0,
        /// <summary>
        /// 
        /// </summary>
        MipTail = 1
    }

    /// <summary>
    /// Memory operation types
    /// </summary>
    public enum CUmemOperationType
    {
        /// <summary>
        /// 
        /// </summary>
        Map = 1,
        /// <summary>
        /// 
        /// </summary>
        Unmap = 2
    }

    /// <summary>
    /// Memory handle types
    /// </summary>
    public enum CUmemHandleType
    {
        /// <summary>
        /// 
        /// </summary>
        Generic = 0
    }

    /// <summary>
    /// 
    /// </summary>
    public enum CUmemPool_attribute
    {
        ReuseFollowEventDependencies = 1,
        ReuseAllowOpportunistic,
        ReuseAllowInternalDependencies,
        ReleaseThreshold,
        ReservedMemCurrent,
        ReservedMemHigh,
        UsedMemCurrent,
        UsedMemHigh
    }

    /// <summary>
    /// Flags for ::cuStreamUpdateCaptureDependencies
    /// </summary>
    public enum CUstreamUpdateCaptureDependencies_flags
    {
        /// <summary>
        /// Add new nodes to the dependency set
        /// </summary>
        Add = 0x0,
        /// <summary>
        /// Replace the dependency set with the new nodes
        /// </summary>
        Set = 0x1
    }

    /// <summary>
    /// Flags to specify search options. For more details see ::cuGetProcAddress
    /// </summary>
    public enum CUdriverProcAddress_flags
    {
        /// <summary>
        /// Default search mode for driver symbols.
        /// </summary>
        Default = 0,
        /// <summary>
        /// Search for legacy versions of driver symbols.
        /// </summary>
        LegacyStream = 1 << 0,
        /// <summary>
        /// Search for per-thread versions of driver symbols.
        /// </summary>
        PerThreadDefaultStream = 1 << 1
    }

    /// <summary>
    /// Platform native ordering for GPUDirect RDMA writes
    /// </summary>
    public enum CUGPUDirectRDMAWritesOrdering
    {
        /// <summary>
        /// The device does not natively support ordering of remote writes. ::cuFlushGPUDirectRDMAWrites() can be leveraged if supported.
        /// </summary>
        None = 0,
        /// <summary>
        /// Natively, the device can consistently consume remote writes, although other CUDA devices may not.
        /// </summary>
        Owner = 100,
        /// <summary>
        /// Any CUDA device in the system can consistently consume remote writes to this device.
        /// </summary>
        AllDevices = 200
    }

    /// <summary>
    /// The scopes for ::cuFlushGPUDirectRDMAWrites
    /// </summary>
    public enum CUflushGPUDirectRDMAWritesScope
    {
        /// <summary>
        /// Blocks until remote writes are visible to the CUDA device context owning the data.
        /// </summary>
        WritesToOwner = 100,
        /// <summary>
        /// Blocks until remote writes are visible to all CUDA device contexts.
        /// </summary>
        WritesToAllDevices = 200
    }

    /// <summary>
    /// The targets for ::cuFlushGPUDirectRDMAWrites
    /// </summary>
    public enum CUflushGPUDirectRDMAWritesTarget
    {
        /// <summary>
        /// Sets the target for ::cuFlushGPUDirectRDMAWrites() to the currently active CUDA device context.
        /// </summary>
        CurrentCtx = 0
    }
    #endregion
}
