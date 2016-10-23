using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NVRTC;


namespace vectorAdd_nvrtc
{
    class Program
    {
        static void Main(string[] args)
        {
            string filename = "vectorAdd_kernel.cu"; //we assume the file is in the same folder...
            string fileToCompile = File.ReadAllText(filename);


            CudaRuntimeCompiler rtc = new CudaRuntimeCompiler(fileToCompile, "vectorAdd_kernel");

            rtc.Compile(args);

            string log = rtc.GetLogAsString();

            Console.WriteLine(log);

            byte[] ptx = rtc.GetPTX();

            rtc.Dispose();

            CudaContext ctx = new CudaContext(0);

            CudaKernel vectorAdd = ctx.LoadKernelPTX(ptx, "vectorAdd");
           

            // Print the vector length to be used, and compute its size
            int numElements = 50000;
            SizeT size = numElements * sizeof(float);
            Console.WriteLine("[Vector addition of {0} elements]", numElements);

            // Allocate the host input vector A
            float[] h_A = new float[numElements];
            // Allocate the host input vector B
            float[] h_B = new float[numElements];
            // Allocate the host output vector C
            float[] h_C = new float[numElements];

            Random rand = new Random(0);

            // Initialize the host input vectors
            for (int i = 0; i < numElements; ++i)
            {
                h_A[i] = (float)rand.NextDouble();
                h_B[i] = (float)rand.NextDouble();
            }

            Console.WriteLine("Allocate and copy input data from the host memory to the CUDA device\n");
            // Allocate the device input vector A and copy to device
            CudaDeviceVariable<float> d_A = h_A;

            // Allocate the device input vector B and copy to device
            CudaDeviceVariable<float> d_B = h_B;

            // Allocate the device output vector C
            CudaDeviceVariable<float> d_C = new CudaDeviceVariable<float>(numElements);

            // Launch the Vector Add CUDA Kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
            Console.WriteLine("CUDA kernel launch with {0} blocks of {1} threads\n", blocksPerGrid, threadsPerBlock);
            vectorAdd.BlockDimensions = new dim3(threadsPerBlock,1, 1);
            vectorAdd.GridDimensions = new dim3(blocksPerGrid, 1, 1);

            vectorAdd.Run(d_A.DevicePointer, d_B.DevicePointer, d_C.DevicePointer, numElements);

            // Copy the device result vector in device memory to the host result vector
            // in host memory.
            Console.WriteLine("Copy output data from the CUDA device to the host memory\n");
            d_C.CopyToHost(h_C);


            // Verify that the result vector is correct
            for (int i = 0; i < numElements; ++i)
            {
                if (Math.Abs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
                {
                    Console.WriteLine("Result verification failed at element {0}!\n", i);
                    return;
                }
            }

            Console.WriteLine("Test PASSED\n");

            // Free device global memory
            d_A.Dispose();
            d_B.Dispose();
            d_C.Dispose();

            ctx.Dispose();
            Console.WriteLine("Done\n");
        }
    }
}
