//	Copyright (c) 2016, Michael Kunz. All rights reserved.
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
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.NVGraph
{

	/// <summary>
	/// C# wrapper for nvgraph.h
	/// </summary>
	public static class NVGraphNativeMathods
	{
		internal const string NVGRAPH_API_DLL_NAME = "nvgraph64_80";

		[DllImport(NVGRAPH_API_DLL_NAME, EntryPoint = "nvgraphStatusGetString")]
		private static extern IntPtr nvgraphStatusGetStringInternal(nvgraphStatus status);

		/// <summary/>
		public static string nvgraphStatusGetString(nvgraphStatus status)
		{
			IntPtr ptr = nvgraphStatusGetStringInternal(status);
			string result;
			result = Marshal.PtrToStringAnsi(ptr);
			return result;
		}


		/// <summary>
		/// Open the library and create the handle
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphCreate(ref nvgraphContext handle);

		/// <summary>
		/// Close the library and destroy the handle
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphDestroy(nvgraphContext handle);

		/// <summary>
		/// Create an empty graph descriptor
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphCreateGraphDescr(nvgraphContext handle, ref nvgraphGraphDescr descrG);


		/// <summary>
		/// Destroy a graph descriptor
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphDestroyGraphDescr(nvgraphContext handle, nvgraphGraphDescr descrG);


		///// <summary>
		///// Set size, topology data in the graph descriptor
		///// </summary>
		//[DllImport(NVGRAPH_API_DLL_NAME)]
		//public static extern nvgraphStatus nvgraphSetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, IntPtr topologyData, nvgraphTopologyType TType);
		/// <summary>
		/// Set size, topology data in the graph descriptor
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphSetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, nvgraphTopologyBase topologyData, nvgraphTopologyType TType);
		///// <summary>
		///// Set size, topology data in the graph descriptor
		///// </summary>
		//[DllImport(NVGRAPH_API_DLL_NAME)]
		//public static extern nvgraphStatus nvgraphSetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, ref nvgraphCSCTopology32I topologyData, nvgraphTopologyType TType);
		///// <summary>
		///// Set size, topology data in the graph descriptor
		///// </summary>
		//[DllImport(NVGRAPH_API_DLL_NAME)]
		//public static extern nvgraphStatus nvgraphSetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, ref nvgraphCOOTopology32I topologyData, nvgraphTopologyType TType);


		///// <summary>
		///// Query size and topology information from the graph descriptor
		///// </summary>
		//[DllImport(NVGRAPH_API_DLL_NAME)]
		//public static extern nvgraphStatus nvgraphGetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, IntPtr topologyData, ref nvgraphTopologyType TType);
		/// <summary>
		/// Query size and topology information from the graph descriptor
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphGetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, nvgraphTopologyBase topologyData, ref nvgraphTopologyType TType);
		///// <summary>
		///// Query size and topology information from the graph descriptor
		///// </summary>
		//[DllImport(NVGRAPH_API_DLL_NAME)]
		//public static extern nvgraphStatus nvgraphGetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, ref nvgraphCSCTopology32I topologyData, ref nvgraphTopologyType TType);
		///// <summary>
		///// Query size and topology information from the graph descriptor
		///// </summary>
		//[DllImport(NVGRAPH_API_DLL_NAME)]
		//public static extern nvgraphStatus nvgraphGetGraphStructure(nvgraphContext handle, nvgraphGraphDescr descrG, ref nvgraphCOOTopology32I topologyData, ref nvgraphTopologyType TType);


		/// <summary>
		/// Allocate numsets vectors of size V reprensenting Vertex Data and attached them the graph. 
		/// settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphAllocateVertexData(nvgraphContext handle, nvgraphGraphDescr descrG, SizeT numsets, cudaDataType[] settypes);


		/// <summary>
		/// Allocate numsets vectors of size E reprensenting Edge Data and attached them the graph. 
		/// settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphAllocateEdgeData(nvgraphContext handle, nvgraphGraphDescr descrG, SizeT numsets, cudaDataType[] settypes);


		/// <summary>
		/// Update the vertex set #setnum with the data in *vertexData, sets have 0-based index
		/// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphSetVertexData(nvgraphContext handle, nvgraphGraphDescr descrG, IntPtr vertexData, SizeT setnum);
        /// <summary>
        /// Update the vertex set #setnum with the data in *vertexData, sets have 0-based index
        /// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure
        /// </summary>
        [DllImport(NVGRAPH_API_DLL_NAME)]
        public static extern nvgraphStatus nvgraphSetVertexData(nvgraphContext handle, nvgraphGraphDescr descrG, CUdeviceptr vertexData, SizeT setnum);


        /// <summary>
        /// Copy the edge set #setnum in *edgeData, sets have 0-based index
        /// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure
        /// </summary>
        [DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphGetVertexData(nvgraphContext handle, nvgraphGraphDescr descrG, IntPtr vertexData, SizeT setnum);
        /// <summary>
        /// Copy the edge set #setnum in *edgeData, sets have 0-based index
        /// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure
        /// </summary>
        [DllImport(NVGRAPH_API_DLL_NAME)]
        public static extern nvgraphStatus nvgraphGetVertexData(nvgraphContext handle, nvgraphGraphDescr descrG, CUdeviceptr vertexData, SizeT setnum);


        /// <summary>
        /// Convert the edge data to another topology
        /// </summary>
        [DllImport(NVGRAPH_API_DLL_NAME)]
        public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
                                        nvgraphTopologyType srcTType, nvgraphTopologyBase srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
                                        nvgraphTopologyType dstTType, nvgraphTopologyBase dstTopology, CUdeviceptr dstEdgeData);

        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCSRTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCSRTopology32I dstTopology, CUdeviceptr dstEdgeData);

        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCSCTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCSCTopology32I dstTopology, CUdeviceptr dstEdgeData);

        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCOOTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCOOTopology32I dstTopology, CUdeviceptr dstEdgeData);
        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCSRTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCSCTopology32I dstTopology, CUdeviceptr dstEdgeData);
        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCSRTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCOOTopology32I dstTopology, CUdeviceptr dstEdgeData);

        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCSCTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCSRTopology32I dstTopology, CUdeviceptr dstEdgeData);
        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCSCTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCOOTopology32I dstTopology, CUdeviceptr dstEdgeData);

        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCOOTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCSRTopology32I dstTopology, CUdeviceptr dstEdgeData);
        ///// <summary>
        ///// Convert the edge data to another topology
        ///// </summary>
        //[DllImport(NVGRAPH_API_DLL_NAME)]
        //public static extern nvgraphStatus nvgraphConvertTopology(nvgraphContext handle,
        //                                nvgraphTopologyType srcTType, ref nvgraphCOOTopology32I srcTopology, CUdeviceptr srcEdgeData, ref cudaDataType dataType,
        //                                nvgraphTopologyType dstTType, ref nvgraphCSCTopology32I dstTopology, CUdeviceptr dstEdgeData);

        /// <summary>
        /// Convert graph to another structure
        /// </summary>
        [DllImport(NVGRAPH_API_DLL_NAME)]
        public static extern nvgraphStatus nvgraphConvertGraph(nvgraphContext handle, nvgraphGraphDescr srcDescrG, nvgraphGraphDescr dstDescrG, nvgraphTopologyType dstTType);



        /// <summary>
        /// Update the edge set #setnum with the data in *edgeData, sets have 0-based index
        /// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure
        /// </summary>
        [DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphSetEdgeData(nvgraphContext handle, nvgraphGraphDescr descrG, IntPtr edgeData, SizeT setnum);
        /// <summary>
        /// Update the edge set #setnum with the data in *edgeData, sets have 0-based index
        /// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure
        /// </summary>
        [DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphSetEdgeData(nvgraphContext handle, nvgraphGraphDescr descrG, CUdeviceptr edgeData, SizeT setnum);

 
		/// <summary>
		/// Copy the edge set #setnum in *edgeData, sets have 0-based index
		/// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure */
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphGetEdgeData(nvgraphContext handle, nvgraphGraphDescr descrG, IntPtr edgeData, SizeT setnum);
		/// <summary>
		/// Copy the edge set #setnum in *edgeData, sets have 0-based index
		/// Conversions are not sopported so nvgraphTopologyType_t should match the graph structure */
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphGetEdgeData(nvgraphContext handle, nvgraphGraphDescr descrG, CUdeviceptr edgeData, SizeT setnum);


		/// <summary>
		/// create a new graph by extracting a subgraph given a list of vertices
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphExtractSubgraphByVertex(nvgraphContext handle, nvgraphGraphDescr descrG, nvgraphGraphDescr subdescrG, int[] subvertices, SizeT numvertices );


		/// <summary>
		/// create a new graph by extracting a subgraph given a list of edges
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphExtractSubgraphByEdge( nvgraphContext handle, nvgraphGraphDescr descrG, nvgraphGraphDescr subdescrG, int[] subedges , SizeT numedges);


		/// <summary>
		/// nvGRAPH Semi-ring sparse matrix vector multiplication
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphSrSpmv(nvgraphContext handle,
                                 nvgraphGraphDescr descrG,
                                 SizeT weight_index,
                                 IntPtr alpha,
                                 SizeT x_index,
                                 IntPtr beta,
                                 SizeT y_index,
                                 nvgraphSemiring SR);


		/// <summary>
		/// nvGRAPH Single Source Shortest Path (SSSP)
		/// Calculate the shortest path distance from a single vertex in the graph to all other vertices.
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphSssp(nvgraphContext handle,
                               nvgraphGraphDescr descrG,
                               SizeT weight_index,
                               ref int source_vert,
                               SizeT sssp_index);


		/// <summary>
		/// nvGRAPH WidestPath 
		/// Find widest path potential from source_index to every other vertices.
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphWidestPath(nvgraphContext handle,
                                  nvgraphGraphDescr descrG,
                                  SizeT weight_index,
                                  ref int source_vert,
                                  SizeT widest_path_index);


		/// <summary>
		/// nvGRAPH PageRank
		/// Find PageRank for each vertex of a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
		/// </summary>
		[DllImport(NVGRAPH_API_DLL_NAME)]
		public static extern nvgraphStatus nvgraphPagerank(nvgraphContext handle,
                                   nvgraphGraphDescr descrG,
                                   SizeT weight_index,
                                   IntPtr alpha,
                                   SizeT bookmark_index,
                                   int has_guess,
                                   SizeT pagerank_index,
                                   float tolerance,
                                   int max_iter );
		
	}
}
