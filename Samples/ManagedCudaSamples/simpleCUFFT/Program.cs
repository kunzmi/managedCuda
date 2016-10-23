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
using ManagedCuda.CudaFFT;

namespace simpleCUFFT
{
    class Program
    {
        static void Main(string[] args)
        {
            int SIGNAL_SIZE = 50;
            int FILTER_KERNEL_SIZE = 11;

            Console.WriteLine("[simpleCUFFT] is starting...");

            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = "simpleCUFFT.simpleCUFFTKernel.ptx";

            CudaContext ctx = new CudaContext(0);
            CudaKernel ComplexPointwiseMulAndScale;
            string[] liste = assembly.GetManifestResourceNames();
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            {
                ComplexPointwiseMulAndScale = ctx.LoadKernelPTX(stream, "ComplexPointwiseMulAndScale");
            }
            
            // Allocate host memory for the signal
            cuFloatComplex[] h_signal = new cuFloatComplex[SIGNAL_SIZE]; //we use cuFloatComplex for complex multiplaction in reference host code...

            Random rand = new Random(0);
            // Initialize the memory for the signal
            for (int i = 0; i < SIGNAL_SIZE; ++i)
            {
                h_signal[i].real = (float)rand.NextDouble();
                h_signal[i].imag = 0;
            }

            // Allocate host memory for the filter
            cuFloatComplex[] h_filter_kernel = new cuFloatComplex[FILTER_KERNEL_SIZE];

            // Initialize the memory for the filter
            for (int i = 0; i < FILTER_KERNEL_SIZE; ++i)
            {
                h_filter_kernel[i].real = (float)rand.NextDouble();
                h_filter_kernel[i].imag = 0;
            }

            // Pad signal and filter kernel
            cuFloatComplex[] h_padded_signal = null;
            cuFloatComplex[] h_padded_filter_kernel = null;
            int new_size = PadData(h_signal, ref h_padded_signal, SIGNAL_SIZE,
                                   h_filter_kernel, ref h_padded_filter_kernel, FILTER_KERNEL_SIZE);
            int mem_size = (int)cuFloatComplex.SizeOf * new_size;


            // Allocate device memory for signal
            CudaDeviceVariable<cuFloatComplex> d_signal = new CudaDeviceVariable<cuFloatComplex>(new_size);
            // Copy host memory to device
            d_signal.CopyToDevice(h_padded_signal);

            // Allocate device memory for filter kernel
            CudaDeviceVariable<cuFloatComplex> d_filter_kernel = new CudaDeviceVariable<cuFloatComplex>(new_size);

            // Copy host memory to device
            d_filter_kernel.CopyToDevice(h_padded_filter_kernel);

            // CUFFT plan simple API
            CudaFFTPlan1D plan = new CudaFFTPlan1D(new_size, cufftType.C2C, 1);

            // Transform signal and kernel
            Console.WriteLine("Transforming signal cufftExecC2C");
            plan.Exec(d_signal.DevicePointer, TransformDirection.Forward);
            plan.Exec(d_filter_kernel.DevicePointer, TransformDirection.Forward);

            // Multiply the coefficients together and normalize the result
            Console.WriteLine("Launching ComplexPointwiseMulAndScale<<< >>>");
            ComplexPointwiseMulAndScale.BlockDimensions = 256;
            ComplexPointwiseMulAndScale.GridDimensions = 32;
            ComplexPointwiseMulAndScale.Run(d_signal.DevicePointer, d_filter_kernel.DevicePointer, new_size, 1.0f / new_size);

            // Transform signal back
            Console.WriteLine("Transforming signal back cufftExecC2C");
            plan.Exec(d_signal.DevicePointer, TransformDirection.Inverse);

            // Copy device memory to host
            cuFloatComplex[] h_convolved_signal = d_signal;

            // Allocate host memory for the convolution result
            cuFloatComplex[] h_convolved_signal_ref = new cuFloatComplex[SIGNAL_SIZE];

            // Convolve on the host
            Convolve(h_signal, SIGNAL_SIZE,
                     h_filter_kernel, FILTER_KERNEL_SIZE,
                     h_convolved_signal_ref);

            // check result
            bool bTestResult = sdkCompareL2fe(h_convolved_signal_ref, h_convolved_signal, 1e-5f);

            //Destroy CUFFT context
            plan.Dispose();

            // cleanup memory
            d_filter_kernel.Dispose();
            d_signal.Dispose();
            ctx.Dispose();

            if (bTestResult)
            {
                Console.WriteLine("Test Passed");
            }
            else
            {
                Console.WriteLine("Test Failed");
            }
        }


        // Pad data
        static int PadData(cuFloatComplex[] signal, ref cuFloatComplex[] padded_signal, int signal_size,
            cuFloatComplex[] filter_kernel, ref cuFloatComplex[] padded_filter_kernel, int filter_kernel_size)
        {
            int minRadius = filter_kernel_size / 2;
            int maxRadius = filter_kernel_size - minRadius;
            int new_size = signal_size + maxRadius;

            // Pad signal
            padded_signal = new cuFloatComplex[new_size];
            Array.Copy(signal, 0, padded_signal, 0, signal_size);

            // Pad filter
            padded_filter_kernel = new cuFloatComplex[new_size];
            Array.Copy(filter_kernel, minRadius, padded_filter_kernel, 0, maxRadius);
            Array.Copy(filter_kernel, 0, padded_filter_kernel, (new_size - minRadius), minRadius);
            
            return new_size;
        }

        // Computes convolution on the host
        static void Convolve(cuFloatComplex[] signal, int signal_size,
              cuFloatComplex[] filter_kernel, int filter_kernel_size,
              cuFloatComplex[] filtered_signal)
        {
            int minRadius = filter_kernel_size / 2;
                int maxRadius = filter_kernel_size - minRadius;

            // Loop over output element indices
            for (int i = 0; i<signal_size; ++i)
            {
                filtered_signal[i].real = filtered_signal[i].imag = 0;

                // Loop over convolution indices
                for (int j = -maxRadius + 1; j <= minRadius; ++j)
                {
                    int k = i + j;

                    if (k >= 0 && k<signal_size)
                    {
                        filtered_signal[i] = filtered_signal[i] + signal[k] * filter_kernel[minRadius - j];
                    }
                }
            }
        }

        static bool sdkCompareL2fe(cuFloatComplex[] h_convolved_signal_ref, cuFloatComplex[] h_convolved_signal, float eps)
        {
            float sumDiff = 0;
            for (int i = 0; i < h_convolved_signal_ref.Length; i++)
            {
                cuFloatComplex diff = h_convolved_signal_ref[i] - h_convolved_signal[i];
                sumDiff += diff.real * diff.real + diff.imag * diff.imag;
            }
            return (Math.Sqrt(sumDiff / h_convolved_signal_ref.Length / 2.0f) < eps);
        }
    }
}
