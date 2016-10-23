using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.CudaBlas;

namespace simpleCUBLAS
{
    class Program
    {

        /* Host implementation of a simple version of sgemm */
        static void simple_sgemm(int n, float alpha, float[] A, float[] B, float beta, float[] C)
        {
            int i;
            int j;
            int k;

            for (i = 0; i<n; ++i)
            {
                for (j = 0; j<n; ++j)
                {
                    float prod = 0;

                    for (k = 0; k<n; ++k)
                    {
                        prod += A[k * n + i] * B[j * n + k];
                    }

                    C[j * n + i] = alpha* prod + beta* C[j * n + i];
                }
            }
        }

        static void Main(string[] args)
        {
            int N = 275;
            
            float[] h_A;
            float[] h_B;
            float[] h_C;
            float[] h_C_ref;

            CudaDeviceVariable<float> d_A;
            CudaDeviceVariable<float> d_B;
            CudaDeviceVariable<float> d_C;
            float alpha = 1.0f;
            float beta = 0.0f;
            int n2 = N * N;
            int i;
            float error_norm;
            float ref_norm;
            float diff;
            CudaBlas handle;
            

            /* Initialize CUBLAS */
            Console.WriteLine("simpleCUBLAS test running.");

            handle = new CudaBlas();

            /* Allocate host memory for the matrices */
            h_A = new float[n2];
            h_B = new float[n2];
            //h_C = new float[n2];
            h_C_ref = new float[n2];

            Random rand = new Random(0);
            /* Fill the matrices with test data */
            for (i = 0; i < n2; i++)
            {
                h_A[i] = (float)rand.NextDouble();
                h_B[i] = (float)rand.NextDouble();
                //h_C[i] = (float)rand.NextDouble();
            }

            /* Allocate device memory for the matrices */
            d_A = new CudaDeviceVariable<float>(n2);
            d_B = new CudaDeviceVariable<float>(n2);
            d_C = new CudaDeviceVariable<float>(n2);


            /* Initialize the device matrices with the host matrices */
            d_A.CopyToDevice(h_A);
            d_B.CopyToDevice(h_B);
            //d_C.CopyToDevice(h_C);

            /* Performs operation using plain C code */
            simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);

            /* Performs operation using cublas */
            handle.Gemm(Operation.NonTranspose, Operation.NonTranspose, N, N, N, alpha, d_A, N, d_B, N, beta, d_C, N);
                        

            /* Allocate host memory for reading back the result from device memory */
            h_C = d_C;

            

            /* Check result against reference */
            error_norm = 0;
            ref_norm = 0;

            for (i = 0; i < n2; ++i)
            {
                diff = h_C_ref[i] - h_C[i];
                error_norm += diff * diff;
                ref_norm += h_C_ref[i] * h_C_ref[i];
            }

            error_norm = (float)Math.Sqrt((double)error_norm);
            ref_norm = (float)Math.Sqrt((double)ref_norm);

            if (Math.Abs(ref_norm) < 1e-7)
            {
                Console.WriteLine("!!!! reference norm is 0");
                return;
            }

            /* Memory clean up */
            d_A.Dispose();
            d_B.Dispose();
            d_C.Dispose();


            /* Shutdown */
            handle.Dispose();

            if (error_norm / ref_norm < 1e-6f)
            {
                Console.WriteLine("simpleCUBLAS test passed.");
                return;
            }
            else
            {
                Console.WriteLine("simpleCUBLAS test failed.");
                return;
            }
        }
    }
}
