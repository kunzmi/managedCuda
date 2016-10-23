/*
 * This code is based on code from the NVIDIA CUDA SDK. (Ported from C++ to C# using managedCUDA)
 * This software contains source code provided by NVIDIA Corporation.
 *
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using shrQATest;

namespace simpleStreams
{
	class Program
	{
		static string[] sEventSyncMethod = {"cudaEventDefault", "cudaEventBlockingSync", "cudaEventDisableTiming" };
		static string[] sDeviceSyncMethod = { "cudaDeviceScheduleAuto", "cudaDeviceScheduleSpin", "cudaDeviceScheduleYield", "INVALID", "cudaDeviceScheduleBlockingSync" };

		// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
		private struct sSMtoCores
		{
			public int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
			public int Cores;

			public sSMtoCores(int sm, int cores)
			{
				SM = sm;
				Cores = cores;
			}
		}

		// Beginning of GPU Architecture definitions
		private static int ConvertSMVer2Cores(int major, int minor)
		{
			sSMtoCores[] nGpuArchCoresPerSM = new[]
			{  
				new sSMtoCores(0x10,  8 ),
				new sSMtoCores( 0x11,  8 ),
				new sSMtoCores( 0x12,  8 ),
				new sSMtoCores( 0x13,  8 ),
				new sSMtoCores( 0x20, 32 ),
				new sSMtoCores( 0x21, 48 ),
				new sSMtoCores( 0x30, 192),
				new sSMtoCores( 0x35, 192),
				new sSMtoCores(   -1, -1 ) 
			};

			int index = 0;

			while (nGpuArchCoresPerSM[index].SM != -1)
			{
				if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
				{
					return nGpuArchCoresPerSM[index].Cores;
				}
				index++;
			}
			throw new CudaException("MapSMtoCores undefined SMversion " + major.ToString() + "." + minor.ToString() + "!");

		}

		static void Main(string[] args)
		{
			int cuda_device = 0;
			int nstreams = 4;               // number of streams for CUDA calls
			int nreps = 10;                 // number of times each experiment is repeated
			int n = 16 * 1024 * 1024;       // number of ints in the data set
			int nbytes = n * sizeof(int);   // number of data bytes
			dim3 threads, blocks;           // kernel launch configuration
			float elapsed_time, time_memcpy, time_kernel;   // timing variables
			float scale_factor = 1.0f;

			// allocate generic memory and pin it laster instead of using cudaHostAlloc()
			// Untested in C#, so stick to cudaHostAlloc().
			bool bPinGenericMemory = false; // we want this to be the default behavior
			CUCtxFlags device_sync_method = CUCtxFlags.BlockingSync; // by default we use BlockingSync

			int niterations;	// number of iterations for the loop inside the kernel
			
			ShrQATest.shrQAStart(args);

			Console.WriteLine("[ simpleStreams ]");

			foreach (var item in args)
			{
				if (item.Contains("help"))
				{
					printHelp();
					ShrQATest.shrQAFinishExit(args, ShrQATest.eQAstatus.QA_PASSED);
				}
			}

			bPinGenericMemory = false;
			foreach (var item in args)
			{
				if (item.Contains("use_generic_memory"))
				{
					bPinGenericMemory = true;
				}
			}

			for (int i = 0; i < args.Length; i++)
			{
				if (args[i].Contains("sync_method"))
				{
					int temp = -1;
					bool error = false;
					if (i < args.Length - 1)
					{
						error = int.TryParse(args[i + 1], out temp);
						switch (temp)
						{ 
							case 0:
								device_sync_method = CUCtxFlags.SchedAuto;
								break;
							case 1:
								device_sync_method = CUCtxFlags.SchedSpin;
								break;
							case 2:
								device_sync_method = CUCtxFlags.SchedYield;
								break;
							case 4:
								device_sync_method = CUCtxFlags.BlockingSync;
								break;
							default:
								error = true;
								break;
						}
					}
					if (!error)
					{
						Console.Write("Specifying device_sync_method = {0}, setting reps to 100 to demonstrate steady state\n", sDeviceSyncMethod[(int)device_sync_method]);
						nreps = 100;
					}
					else 
					{
						Console.Write("Invalid command line option sync_method=\"{0}\"\n", temp);
						ShrQATest.shrQAFinishExit(args, ShrQATest.eQAstatus.QA_FAILED);
					}
				}
			}

			int num_devices = CudaContext.GetDeviceCount();
			if(0==num_devices)
			{
				Console.Write("your system does not have a CUDA capable device, waiving test...\n");
				ShrQATest.shrQAFinishExit(args, ShrQATest.eQAstatus.QA_FAILED);
			}
			cuda_device = CudaContext.GetMaxGflopsDeviceId();

			CudaDeviceProperties deviceProp = CudaContext.GetDeviceInfo(cuda_device);
			if ((1 == deviceProp.ComputeCapability.Major) && (deviceProp.ComputeCapability.Minor < 1))
			{
				Console.Write("{0} does not have Compute Capability 1.1 or newer. Reducing workload.\n", deviceProp.DeviceName);
			}

			if (deviceProp.ComputeCapability.Major >= 2)
			{
				niterations = 100;
			}
			else
			{
				if (deviceProp.ComputeCapability.Minor > 1)
				{
					niterations = 5;
				}
				else
				{
					niterations = 1; // reduced workload for compute capability 1.0 and 1.1
				}
			}

			// Check if GPU can map host memory (Generic Method), if not then we override bPinGenericMemory to be false
			// In .net we cannot allocate easily generic aligned memory, so <bPinGenericMemory> is always false in our case...
			if (bPinGenericMemory)
			{
				Console.Write("Device: <{0}> canMapHostMemory: {1}\n", deviceProp.DeviceName, deviceProp.CanMapHostMemory ? "Yes" : "No");
				if (deviceProp.CanMapHostMemory == false)
				{
					Console.Write("Using cudaMallocHost, CUDA device does not support mapping of generic host memory\n");
					bPinGenericMemory = false;
				}
			}

			// Anything that is less than 32 Cores will have scaled down workload
			scale_factor = Math.Max((32.0f / (ConvertSMVer2Cores(deviceProp.ComputeCapability.Major, deviceProp.ComputeCapability.Minor) * (float)deviceProp.MultiProcessorCount)), 1.0f);
			n = (int)Math.Round((float)n / scale_factor);

			Console.Write("> CUDA Capable: SM {0}.{1} hardware\n", deviceProp.ComputeCapability.Major, deviceProp.ComputeCapability.Minor);
			Console.Write("> {0} Multiprocessor(s) x {1} (Cores/Multiprocessor) = {2} (Cores)\n",
					deviceProp.MultiProcessorCount,
					ConvertSMVer2Cores(deviceProp.ComputeCapability.Major, deviceProp.ComputeCapability.Minor),
					ConvertSMVer2Cores(deviceProp.ComputeCapability.Major, deviceProp.ComputeCapability.Minor) * deviceProp.MultiProcessorCount);

			Console.Write("> scale_factor = {0:0.0000}\n", 1.0f / scale_factor);
			Console.Write("> array_size   = {0}\n\n", n);

			// enable use of blocking sync, to reduce CPU usage
			Console.Write("> Using CPU/GPU Device Synchronization method ({0})\n", sDeviceSyncMethod[(int)device_sync_method]);

			CudaContext ctx;
			if (bPinGenericMemory)
				ctx = new CudaContext(cuda_device, device_sync_method | CUCtxFlags.MapHost);
			else
				ctx = new CudaContext(cuda_device, device_sync_method);
			
			//Load Kernel image from resources
			string resName;
			if (IntPtr.Size == 8)
				resName = "simpleStreams_x64.ptx";
			else
				resName = "simpleStreams.ptx";

			string resNamespace = "simpleStreams";
			string resource = resNamespace + "." + resName;
			Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resource);
			if (stream == null) throw new ArgumentException("Kernel not found in resources.");
			
			CudaKernel init_array = ctx.LoadKernelPTX(stream, "init_array");


			// allocate host memory
			int c = 5;											// value to which the array will be initialized
			int[] h_a = null;									// pointer to the array data in host memory
			CudaPageLockedHostMemory<int> hAligned_a = null;	// pointer to the array data in host memory (aligned to MEMORY_ALIGNMENT)
			//Note: In .net we have two seperated arrays: One is in managed memory (h_a), the other one in unmanaged memory (hAligned_a).
			//In C++ hAligned_a would point somewhere inside the h_a array.
			AllocateHostMemory(bPinGenericMemory, ref h_a, ref hAligned_a, nbytes);

			Console.Write("\nStarting Test\n");

			// allocate device memory
			CudaDeviceVariable<int> d_c = c; //using new implicit cast to allocate memory and asign value
			CudaDeviceVariable<int> d_a = new CudaDeviceVariable<int>(nbytes / sizeof(int));

			CudaStream[] streams = new CudaStream[nstreams];
			for (int i = 0; i < nstreams; i++)
			{
				streams[i] = new CudaStream();
			}

			// create CUDA event handles
			// use blocking sync
			CudaEvent start_event, stop_event;
			CUEventFlags eventflags = ((device_sync_method == CUCtxFlags.BlockingSync) ? CUEventFlags.BlockingSync : CUEventFlags.Default);

			start_event = new CudaEvent(eventflags);
			stop_event = new CudaEvent(eventflags);

			// time memcopy from device
			start_event.Record();     // record in stream-0, to ensure that all previous CUDA calls have completed
			hAligned_a.AsyncCopyToDevice(d_a, streams[0].Stream);
			stop_event.Record();
			stop_event.Synchronize();   // block until the event is actually recorded
			time_memcpy = CudaEvent.ElapsedTime(start_event, stop_event);
			Console.Write("memcopy:\t{0:0.00}\n", time_memcpy);
	
			// time kernel
			threads = new dim3(512, 1);
			blocks = new dim3(n / (int)threads.x, 1);
			start_event.Record();
			init_array.BlockDimensions = threads;
			init_array.GridDimensions = blocks;
			init_array.RunAsync(streams[0].Stream, d_a.DevicePointer, d_c.DevicePointer, niterations);
			stop_event.Record();
			stop_event.Synchronize();
			time_kernel = CudaEvent.ElapsedTime(start_event, stop_event);
			Console.Write("kernel:\t\t{0:0.00}\n", time_kernel);

			
			//////////////////////////////////////////////////////////////////////
			// time non-streamed execution for reference
			threads = new dim3(512, 1);
			blocks = new dim3(n / (int)threads.x, 1);
			start_event.Record();
			for(int k = 0; k < nreps; k++)
			{
				init_array.BlockDimensions = threads;
				init_array.GridDimensions = blocks;
				init_array.Run(d_a.DevicePointer, d_c.DevicePointer, niterations);
				hAligned_a.SynchronCopyToHost(d_a);
			}
			stop_event.Record();
			stop_event.Synchronize();
			elapsed_time = CudaEvent.ElapsedTime(start_event, stop_event);
			Console.Write("non-streamed:\t{0:0.00} ({1:00} expected)\n", elapsed_time / nreps, time_kernel + time_memcpy);

			//////////////////////////////////////////////////////////////////////
			// time execution with nstreams streams
			threads = new dim3(512, 1);
			blocks = new dim3(n / (int)(nstreams * threads.x), 1);
			byte[] memset = new byte[nbytes]; // set host memory bits to all 1s, for testing correctness
			for (int i = 0; i < nbytes; i++)
			{
				memset[i] = 255;
			}
			System.Runtime.InteropServices.Marshal.Copy(memset, 0, hAligned_a.PinnedHostPointer, nbytes);
			d_a.Memset(0); // set device memory to all 0s, for testing correctness
			
			start_event.Record();
			for(int k = 0; k < nreps; k++)
			{
				init_array.BlockDimensions = threads;
				init_array.GridDimensions = blocks;
				// asynchronously launch nstreams kernels, each operating on its own portion of data
				for(int i = 0; i < nstreams; i++)
					init_array.RunAsync(streams[i].Stream, d_a.DevicePointer + i * n / nstreams * sizeof(int), d_c.DevicePointer, niterations);

				// asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
				//   commence executing when all previous CUDA calls in stream x have completed
                for (int i = 0; i < nstreams; i++)
                    hAligned_a.AsyncCopyFromDevice(d_a, i * n / nstreams * sizeof(int), i * n / nstreams * sizeof(int), nbytes / nstreams, streams[i].Stream);
			}
			stop_event.Record();
			stop_event.Synchronize();
			elapsed_time = CudaEvent.ElapsedTime(start_event, stop_event);
			Console.Write("{0} streams:\t{1:0.00} ({2:0.00} expected with compute capability 1.1 or later)\n", nstreams, elapsed_time / nreps, time_kernel + time_memcpy / nstreams);

			// check whether the output is correct
			Console.Write("-------------------------------\n");
			//We can directly access data in hAligned_a using the [] operator, but copying
			//data first to h_a is faster.
			System.Runtime.InteropServices.Marshal.Copy(hAligned_a.PinnedHostPointer, h_a, 0, nbytes / sizeof(int));

			bool bResults = correct_data(h_a, n, c*nreps*niterations);

			// release resources
			for(int i = 0; i < nstreams; i++) {
				streams[i].Dispose();
			}
			start_event.Dispose();
			stop_event.Dispose();

			hAligned_a.Dispose();
			d_a.Dispose();
			d_c.Dispose();
			CudaContext.ProfilerStop();
			ctx.Dispose();
			
			ShrQATest.shrQAFinishExit(args, bResults ? ShrQATest.eQAstatus.QA_PASSED : ShrQATest.eQAstatus.QA_FAILED);
		}

		static void printHelp()
		{
			Console.Write("Usage: simpleStreams [options below]\n");
			Console.Write("\t--sync_method for CPU/GPU synchronization\n");
			Console.Write("\t             (0=Automatic Blocking Scheduling)\n");
			Console.Write("\t             (1=Spin Blocking Scheduling)\n");
			Console.Write("\t             (2=Yield Blocking Scheduling)\n");
			Console.Write("\t   <Default> (4=Blocking Sync Event Scheduling for low CPU utilization)\n");
			Console.Write("\t--use_generic_memory use generic page-aligned for system memory\n");
		}

		static void AllocateHostMemory(bool bPinGenericMemory, ref int[] pp_a, ref CudaPageLockedHostMemory<int> pp_Aligned_a, int nbytes)
		{
			Console.Write("> cudaMallocHost() allocating {0:0.00} Mbytes of system memory\n", (float)nbytes / 1048576.0f);
			// allocate host memory (pinned is required for achieve asynchronicity)
			if (pp_Aligned_a != null)
				pp_Aligned_a.Dispose();

			pp_Aligned_a = new CudaPageLockedHostMemory<int>(nbytes / sizeof(int));
			pp_a = new int[nbytes / sizeof(int)];
		}

		static bool correct_data(int[] a, int n, int c)
		{
			for(int i = 0; i < n; i++) {
				if(a[i] != c) {
					Console.Write("{0}: {1} {2}\n", i, a[i], c);
					return false;
				}
			}
			return true;
		}
	}
}
