using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVGraph;


namespace nvgraph_Pagerank
{
    class Program
    {

        /* PageRank
         *  Find PageRank for a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
         *  This is equivalent to an eigenvalue problem where we want the eigenvector corresponding to the maximum eigenvalue.
         *  By construction, the maximum eigenvalue is 1.
         *  The eigenvalue problem is solved with the power method.

        Initially :
        V = 6 
        E = 10

        Edges       W
        0 -> 1    0.50
        0 -> 2    0.50
        2 -> 0    0.33
        2 -> 1    0.33
        2 -> 4    0.33
        3 -> 4    0.50
        3 -> 5    0.50
        4 -> 3    0.50
        4 -> 5    0.50
        5 -> 3    1.00

        bookmark (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)^T note: 1.0 if i is a dangling node, 0.0 otherwise

        Source oriented representation (CSC):
        destination_offsets {0, 1, 3, 4, 6, 8, 10}
        source_indices {2, 0, 2, 0, 4, 5, 2, 3, 3, 4}
        W0 = {0.33, 0.50, 0.33, 0.50, 0.50, 1.00, 0.33, 0.50, 0.50, 1.00}

        ----------------------------------

        Operation : Pagerank with various damping factor 
        ----------------------------------

        Expected output for alpha= 0.9 (result stored in pr_2) : (0.037210, 0.053960, 0.041510, 0.37510, 0.206000, 0.28620)^T 
        From "Google's PageRank and Beyond: The Science of Search Engine Rankings" Amy N. Langville & Carl D. Meyer
        */

        static void Main(string[] args)
        {

            SizeT n = 6, nnz = 10, vertex_numsets = 3, edge_numsets = 1;
            float[] alpha1 = new float[] { 0.85f }, alpha2 = new float[] { 0.90f };

            int i;
            int[] destination_offsets_h, source_indices_h;
            float[] weights_h, bookmark_h, pr_1, pr_2;
            //void** vertex_dim;

            // nvgraph variables
            GraphContext handle;
            GraphDescriptor graph;
            nvgraphCSCTopology32I CSC_input;
            cudaDataType[] edge_dimT = new cudaDataType[] { cudaDataType.CUDA_R_32F };
            cudaDataType[] vertex_dimT;

            // use command-line specified CUDA device, otherwise use device with highest Gflops/s
            int cuda_device = 0;

            CudaDeviceProperties deviceProp = CudaContext.GetDeviceInfo(cuda_device);

            Console.WriteLine("> Detected Compute SM {0}.{1} hardware with {2} multi-processors",
                   deviceProp.ComputeCapability.Major, deviceProp.ComputeCapability.Minor, deviceProp.MultiProcessorCount);

            if (deviceProp.ComputeCapability.Major < 3)
            {
                Console.WriteLine("> nvGraph requires device SM 3.0+");
                Console.WriteLine("> Waiving.");
                return;
            }


            // Allocate host data
            destination_offsets_h = new int[n + 1];
            source_indices_h = new int[nnz];
            weights_h = new float[nnz];
            bookmark_h = new float[n];
            pr_1 = new float[n];
            pr_2 = new float[n];
            //vertex_dim = (void**)malloc(vertex_numsets * sizeof(void*));
            vertex_dimT = new cudaDataType[vertex_numsets];
            CSC_input = new nvgraphCSCTopology32I();
    
            // Initialize host data
            //vertex_dim[0] = (void*)bookmark_h; vertex_dim[1]= (void*)pr_1, vertex_dim[2]= (void*)pr_2;
            vertex_dimT[0] = cudaDataType.CUDA_R_32F; vertex_dimT[1] = cudaDataType.CUDA_R_32F; vertex_dimT[2]= cudaDataType.CUDA_R_32F;
    
            weights_h[0] = 0.333333f;
            weights_h[1] = 0.500000f;
            weights_h[2] = 0.333333f;
            weights_h[3] = 0.500000f;
            weights_h[4] = 0.500000f;
            weights_h[5] = 1.000000f;
            weights_h[6] = 0.333333f;
            weights_h[7] = 0.500000f;
            weights_h[8] = 0.500000f;
            weights_h[9] = 0.500000f;

            destination_offsets_h[0] = 0;
            destination_offsets_h[1] = 1;
            destination_offsets_h[2] = 3;
            destination_offsets_h[3] = 4;
            destination_offsets_h[4] = 6;
            destination_offsets_h[5] = 8;
            destination_offsets_h[6] = 10;

            source_indices_h[0] = 2;
            source_indices_h[1] = 0;
            source_indices_h[2] = 2;
            source_indices_h[3] = 0;
            source_indices_h[4] = 4;
            source_indices_h[5] = 5;
            source_indices_h[6] = 2;
            source_indices_h[7] = 3;
            source_indices_h[8] = 3;
            source_indices_h[9] = 4;

            bookmark_h[0] = 0.0f;
            bookmark_h[1] = 1.0f;
            bookmark_h[2] = 0.0f;
            bookmark_h[3] = 0.0f;
            bookmark_h[4] = 0.0f;
            bookmark_h[5] = 0.0f;

            // Starting nvgraph
            handle = new GraphContext();
            graph = handle.CreateGraphDecriptor();

            GCHandle destination_offsets_handle = GCHandle.Alloc(destination_offsets_h, GCHandleType.Pinned);
            GCHandle source_indices_handle = GCHandle.Alloc(source_indices_h, GCHandleType.Pinned);

            CSC_input.nvertices = n;
            CSC_input.nedges = nnz;
            CSC_input.destination_offsets = destination_offsets_handle.AddrOfPinnedObject();
            CSC_input.source_indices = source_indices_handle.AddrOfPinnedObject();

            // Set graph connectivity and properties (tranfers)
            graph.SetGraphStructure(CSC_input);
            graph.AllocateVertexData(vertex_dimT);
            graph.AllocateEdgeData(edge_dimT);

            graph.SetVertexData(bookmark_h, 0);
            graph.SetVertexData(pr_1, 1);
            graph.SetVertexData(pr_2, 2);

            graph.SetEdgeData(weights_h, 0);

            // First run with default values
            graph.Pagerank(0, alpha1, 0, 0, 1, 0.0f, 0);

            // Get and print result
            graph.GetVertexData(pr_1, 1);
            Console.WriteLine("pr_1, alpha = 0.85");
            for (i = 0; i<n; i++)
                Console.WriteLine(pr_1[i]);
            Console.WriteLine();

            // Second run with different damping factor and an initial guess
            for (i = 0; i<n; i++)  
                pr_2[i] =pr_1[i];

            graph.SetVertexData(pr_2, 2);
            graph.Pagerank(0, alpha2, 0, 1, 2, 0.0f, 0);

            // Get and print result
            graph.GetVertexData(pr_2, 2);
            Console.WriteLine("pr_2, alpha = 0.90");
            for (i = 0; i < n; i++)
                Console.WriteLine(pr_2[i]);
            Console.WriteLine();

            //Clean 
            graph.Dispose();
            handle.Dispose();
            

            Console.WriteLine("\nDone!");
        }
    }
}
