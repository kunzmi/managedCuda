using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace simpleOccupancy
{
    class Program
    {
        static CudaContext ctx;
        static CudaKernel kernel;
        static int manualBlockSize = 32;

        static void Main(string[] args)
        {
            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = "simpleOccupancy.simpleOccupancy.ptx";

            ctx = new CudaContext(0);
            string[] liste = assembly.GetManifestResourceNames();
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            {
                kernel = ctx.LoadKernelPTX(stream, "square");
            }


            Console.WriteLine("starting Simple Occupancy");
            Console.WriteLine();

            Console.WriteLine("[ Manual configuration with {0} threads per block ]", manualBlockSize);

            int status = test(false);
            if (status != 0)
            {
                Console.WriteLine("Test failed");
                return;
            }

            Console.WriteLine();

            Console.WriteLine("[ Automatic, occupancy-based configuration ]");
            status = test(true);
            if (status != 0)
            {
                Console.WriteLine("Test failed");
                return;
            }

            Console.WriteLine();
            Console.WriteLine("Test PASSED");
        }

        ////////////////////////////////////////////////////////////////////////////////
        // The test
        //
        // The test generates an array and squares it with a CUDA kernel, then
        // verifies the result.
        ////////////////////////////////////////////////////////////////////////////////
        static int test(bool automaticLaunchConfig, int count = 1000000)
        {
            int[] array;
            CudaDeviceVariable<int> dArray;

            array = new int[count];

            for (int i = 0; i < count; i += 1)
            {
                array[i] = i;
            }

            dArray = array;

            for (int i = 0; i < count; i += 1)
            {
                array[i] = 0;
            }

            launchConfig(dArray, count, automaticLaunchConfig);

            dArray.CopyToHost(array);
            dArray.Dispose();

            // Verify the return data
            //
            for (int i = 0; i < count; i += 1)
            {
                if (array[i] != i * i)
                {
                    Console.WriteLine("element {0} expected {1} actual {2}", i, i * i, array[i]);
                    return 1;
                }
            }

            return 0;
        }


        ////////////////////////////////////////////////////////////////////////////////
        // Occupancy-based launch configurator
        //
        // The launch configurator, cudaOccupancyMaxPotentialBlockSize and
        // cudaOccupancyMaxPotentialBlockSizeVariableSMem, suggests a block
        // size that achieves the best theoretical occupancy. It also returns
        // the minimum number of blocks needed to achieve the occupancy on the
        // whole device.
        //
        // This launch configurator is purely occupancy-based. It doesn't
        // translate directly to performance, but the suggestion should
        // nevertheless be a good starting point for further optimizations.
        //
        // This function configures the launch based on the "automatic"
        // argument, records the runtime, and reports occupancy and runtime.
        ////////////////////////////////////////////////////////////////////////////////
        static int launchConfig(CudaDeviceVariable<int> array, int arrayCount, bool automatic)
        {
            int blockSize = 0;
            int minGridSize = 0;
            int gridSize;
            SizeT dynamicSMemUsage = 0;
            

            float elapsedTime;

            double potentialOccupancy;

            CudaOccupancy.cudaOccDeviceState state = new CudaOccupancy.cudaOccDeviceState();
            state.cacheConfig = CudaOccupancy.cudaOccCacheConfig.PreferNone;

            if (automatic)
            {
                CudaOccupancy.cudaOccMaxPotentialOccupancyBlockSize(ref minGridSize, ref blockSize, new CudaOccupancy.cudaOccDeviceProp(0), new CudaOccupancy.cudaOccFuncAttributes(kernel), state, dynamicSMemUsage);

                Console.WriteLine("Suggested block size: {0}", blockSize);
                Console.WriteLine("Minimum grid size for maximum occupancy: {0}", minGridSize);
            }
            else
            {
                // This block size is too small. Given limited number of
                // active blocks per multiprocessor, the number of active
                // threads will be limited, and thus unable to achieve maximum
                // occupancy.
                //
                blockSize = manualBlockSize;
            }

            // Round up
            //
            gridSize = (arrayCount + blockSize - 1) / blockSize;

            // Launch and profile
            //
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            elapsedTime = kernel.Run(array.DevicePointer, arrayCount);

            // Calculate occupancy
            //
            potentialOccupancy = reportPotentialOccupancy(blockSize, dynamicSMemUsage);

            Console.WriteLine("Potential occupancy: {0}%", potentialOccupancy * 100);

            // Report elapsed time
            //
            Console.WriteLine("Elapsed time: {0}ms", elapsedTime * 100);

            return 0;
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Potential occupancy calculator
        //
        // The potential occupancy is calculated according to the kernel and
        // execution configuration the user desires. Occupancy is defined in
        // terms of active blocks per multiprocessor, and the user can convert
        // it to other metrics.
        //
        // This wrapper routine computes the occupancy of kernel, and reports
        // it in terms of active warps / maximum warps per SM.
        ////////////////////////////////////////////////////////////////////////////////
        static double reportPotentialOccupancy(int blockSize, SizeT dynamicSMem)
        {
            int device;

            int numBlocks;
            int activeWarps;
            int maxWarps;

            double occupancy;
            
            CudaOccupancy.cudaOccDeviceProp prop = new CudaOccupancy.cudaOccDeviceProp(0);

            CudaOccupancy.cudaOccResult result = new CudaOccupancy.cudaOccResult();
            CudaOccupancy.cudaOccFuncAttributes attributes = new CudaOccupancy.cudaOccFuncAttributes(kernel);
            CudaOccupancy.cudaOccDeviceState state = new CudaOccupancy.cudaOccDeviceState();
            state.cacheConfig = CudaOccupancy.cudaOccCacheConfig.PreferNone;

            CudaOccupancy.cudaOccMaxActiveBlocksPerMultiprocessor(result, prop, attributes, state, blockSize, dynamicSMem);

            

            numBlocks = result.ActiveBlocksPerMultiProcessor;

            activeWarps = numBlocks * blockSize / prop.warpSize;
            maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

            occupancy = (double)activeWarps / maxWarps;

            return occupancy;
        }

    }


}
