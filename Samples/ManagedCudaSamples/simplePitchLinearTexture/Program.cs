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

namespace simplePitchLinearTexture
{
	class Program
	{
		const int NUM_REPS = 100;  // number of repetitions performed  
		const int TILE_DIM = 16;   // tile/block size

		static CudaContext ctx;

		static void Main(string[] args)
		{
			const int nx = 2048;
			const int ny = 2048;

			// shifts applied to x and y data
			const int x_shift = 5;
			const int y_shift = 7;

			ShrQATest.shrQAStart(args);
			
			if ((nx%TILE_DIM != 0)  || (ny%TILE_DIM != 0))
			{
				Console.Write("nx and ny must be multiples of TILE_DIM\n");
				ShrQATest.shrQAFinishExit(args, ShrQATest.eQAstatus.QA_WAIVED);
			}
			
			// execution configuration parameters
			dim3 grid = new dim3(nx/TILE_DIM, ny/TILE_DIM, 1);
			dim3 threads = new dim3(TILE_DIM, TILE_DIM, 1);

			// This will pick the best possible CUDA capable device
			int devID = findCudaDevice(args);

			
			//Load Kernel image from resources
			string resName;
			if (IntPtr.Size == 8)
				resName = "simplePitchLinearTexture_x64.ptx";
			else
				resName = "simplePitchLinearTexture.ptx";

			string resNamespace = "simplePitchLinearTexture";
			string resource = resNamespace + "." + resName;
			Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resource);
			if (stream == null) throw new ArgumentException("Kernel not found in resources.");
			byte[] kernels = new byte[stream.Length];

			int bytesToRead = (int)stream.Length;
			while (bytesToRead > 0)
			{
				bytesToRead -= stream.Read(kernels, (int)stream.Position, bytesToRead);
			}

			CudaKernel PLKernel = ctx.LoadKernelPTX(kernels, "shiftPitchLinear");
			CudaKernel ArrayKernel = ctx.LoadKernelPTX(kernels, "shiftArray");

			CudaStopWatch stopwatch = new CudaStopWatch();

			// ----------------------------------
			// Host allocation and initialization
			// ----------------------------------

			float[] h_idata = new float[nx * ny];
			float[] h_odata = new float[nx * ny];
			float[] gold = new float[nx * ny];

			for (int i = 0; i < nx * ny; ++i) h_idata[i] = (float)i;

			// ------------------------
			// Device memory allocation
			// ------------------------

			// Pitch linear input data
			CudaPitchedDeviceVariable<float> d_idataPL = new CudaPitchedDeviceVariable<float>(nx, ny);
			
			// Array input data
			CudaArray2D d_idataArray = new CudaArray2D(CUArrayFormat.Float, nx, ny, CudaArray2DNumChannels.One);

			// Pitch linear output data
			CudaPitchedDeviceVariable<float> d_odata = new CudaPitchedDeviceVariable<float>(nx, ny);

			// ------------------------
			// copy host data to device
			// ------------------------

			// Pitch linear
			d_idataPL.CopyToDevice(h_idata);

			// Array
			d_idataArray.CopyFromHostToThis<float>(h_idata);

			// ----------------------
			// Bind texture to memory
			// ----------------------

			// Pitch linear
			CudaTextureLinearPitched2D<float> texRefPL = new CudaTextureLinearPitched2D<float>(PLKernel, "texRefPL", CUAddressMode.Wrap, CUFilterMode.Point, CUTexRefSetFlags.NormalizedCoordinates, CUArrayFormat.Float, d_idataPL);
			CudaTextureArray2D texRefArray = new CudaTextureArray2D(ArrayKernel, "texRefArray", CUAddressMode.Wrap, CUFilterMode.Point, CUTexRefSetFlags.NormalizedCoordinates, d_idataArray);
			
			// ---------------------
			// reference calculation
			// ---------------------

			for (int j = 0; j < ny; j++)
			{
				int jshift = (j + y_shift) % ny;
				for (int i = 0; i < nx; i++)
				{
					int ishift = (i + x_shift) % nx;
					gold[j * nx + i] = h_idata[jshift * nx + ishift];
				}
			}

			// ----------------
			// shiftPitchLinear
			// ----------------

			ctx.ClearMemory(d_odata.DevicePointer, 0, d_odata.TotalSizeInBytes);
			PLKernel.BlockDimensions = threads;
			PLKernel.GridDimensions = grid;
			stopwatch.Start();
			for (int i=0; i < NUM_REPS; i++) 
			{
				PLKernel.Run(d_odata.DevicePointer, (int)(d_odata.Pitch/sizeof(float)), nx, ny, x_shift, y_shift);
			} 
			stopwatch.Stop();
			stopwatch.StopEvent.Synchronize();
			float timePL = stopwatch.GetElapsedTime();
			
			// check results
			d_odata.CopyToHost(h_odata);

			bool res = cutComparef(gold, h_odata);

			bool success = true;
			if (res == false) {
				Console.Write("*** shiftPitchLinear failed ***\n");
				success = false;
			}
	
			// ----------
			// shiftArray
			// ----------
			
			ctx.ClearMemory(d_odata.DevicePointer, 0, d_odata.TotalSizeInBytes);
			ArrayKernel.BlockDimensions = threads;
			ArrayKernel.GridDimensions = grid;
			stopwatch.Start();
			for (int i=0; i < NUM_REPS; i++) {
				ArrayKernel.Run(d_odata.DevicePointer, (int)(d_odata.Pitch/sizeof(float)), nx, ny, x_shift, y_shift);
				
			}
			
			stopwatch.Stop();
			stopwatch.StopEvent.Synchronize();
			float timeArray = stopwatch.GetElapsedTime();
			
			// check results
			d_odata.CopyToHost(h_odata);

			res = cutComparef(gold, h_odata);

			if (res == false) {
				Console.Write("*** shiftArray failed ***\n");
				success = false;
			}
	
			float bandwidthPL = 2.0f*1000.0f*nx*ny*sizeof(float)/(1e+9f)/(timePL/NUM_REPS);
			float bandwidthArray = 2.0f*1000.0f*nx*ny*sizeof(float)/(1e+9f)/(timeArray/NUM_REPS);
			Console.Write("\nBandwidth (GB/s) for pitch linear: {0}; for array: {1}\n", 
				bandwidthPL, bandwidthArray);

			float fetchRatePL = nx*ny/1e+6f/(timePL/(1000.0f*NUM_REPS));
			float fetchRateArray = nx*ny/1e+6f/(timeArray/(1000.0f*NUM_REPS));
			Console.Write("\nTexture fetch rate (Mpix/s) for pitch linear: {0}; for array: {1}\n\n", 
				fetchRatePL, fetchRateArray);


			// cleanup
			texRefPL.Dispose();
			texRefArray.Dispose();
			d_idataPL.Dispose();
			d_idataArray.Dispose();
			d_odata.Dispose();
			stopwatch.Dispose();
			ctx.Dispose();

			ShrQATest.shrQAFinishExit(args, (success == true) ? ShrQATest.eQAstatus.QA_PASSED : ShrQATest.eQAstatus.QA_FAILED);

		}


		// Initialization code to find the best CUDA Device
		static int findCudaDevice(string[] args)
		{
			int devID = 0;
			// If the command-line has a device number specified, use it
			bool found = false;
			foreach (var item in args)
			{
				if (item.Contains("device="))
				{
					found = true;
					if (!int.TryParse(item, out devID))
					{
						Console.WriteLine("Invalid command line parameters");
						Environment.Exit(-1);
					}
					if (devID < 0)
					{
						Console.WriteLine("Invalid command line parameters\n");
						Environment.Exit(-1);
					}
					else
					{
						devID = gpuDeviceInit(devID);
						if (devID < 0)
						{
							Console.WriteLine("exiting...\n");
							ShrQATest.shrQAFinishExit(args, ShrQATest.eQAstatus.QA_FAILED);
							Environment.Exit(-1);
						}
					}
				}
			}

			if (!found)
			{
				// Otherwise pick the device with highest Gflops/s
				devID = CudaContext.GetMaxGflopsDeviceId();
				ctx = new CudaContext(devID, CUCtxFlags.SchedAuto);
				Console.Write("> Using CUDA device [{0}]: {1}\n", devID, ctx.GetDeviceName());
			}
			return devID;
		}


		// General GPU Device CUDA Initialization
		static int gpuDeviceInit(int devID)
		{
			int deviceCount = CudaContext.GetDeviceCount();

			if (deviceCount == 0)
			{
				Console.Write("gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
				Environment.Exit(-1);
			}
			if (devID < 0)
				devID = 0;
			if (devID > deviceCount - 1)
			{
				Console.Write("\n");
				Console.Write(">> {0} CUDA capable GPU device(s) detected. <<\n", deviceCount);
				Console.Write(">> gpuDeviceInit (-device={0}) is not a valid GPU device. <<\n", devID);
				Console.Write("\n");
				return -devID;
			}


			if (CudaContext.GetDeviceComputeCapability(devID).Major < 1)
			{
				Console.Write("gpuDeviceInit(): GPU device does not support CUDA.\n");
				Environment.Exit(-1);
			}
			ctx = new CudaContext(devID);
			Console.Write("> gpuDeviceInit() CUDA device [{0}]: {1}\n", devID, ctx.GetDeviceName());
			return devID;
		}


		// Check Result
		static bool cutComparef(float[] reference, float[] data)
		{
			bool result = true;
			int len = reference.Length;
			float epsilon = 0;
			int error_count = 0;
			for (int i = 0; i < len; i++)
			{
				float diff = reference[i] - data[i];
				bool comp = (diff <= epsilon) && (diff >= -epsilon);
				result &= comp;

				if (!comp)
					error_count += 1;
			}

			return result;
		}
	}
}
