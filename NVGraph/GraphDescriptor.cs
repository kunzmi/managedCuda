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
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NVGraph
{
	public class GraphDescriptor : IDisposable
	{
		bool disposed;
		nvgraphStatus res;
		nvgraphGraphDescr _descr;
		nvgraphContext _context;

		#region Constructor
		/// <summary>
		/// </summary>
		internal GraphDescriptor(nvgraphContext context)
		{
			_descr = new nvgraphGraphDescr();
			_context = context;

			res = NVGraphNativeMathods.nvgraphCreateGraphDescr(_context, ref _descr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphCreateGraphDescr", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~GraphDescriptor()
		{
			Dispose(false);
		}
		#endregion

		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>
		/// For IDisposable
		/// </summary>
		/// <param name="fDisposing"></param>
		protected virtual void Dispose(bool fDisposing)
		{
			if (fDisposing && !disposed)
			{
				res = NVGraphNativeMathods.nvgraphDestroyGraphDescr(_context, _descr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphDestroyGraphDescr", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA.NVGraph not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		public nvgraphGraphDescr GraphDescr
		{
			get { return _descr; }
		}

		//public void SetGraphStructure(nvgraphCSRTopology32I topologyData)
		//{
		//	res = NVGraphNativeMathods.nvgraphSetGraphStructure(_context, _descr, ref topologyData, nvgraphTopologyType.CSR32);
		//	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSetGraphStructure", res));
		//	if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		//}

		//public void SetGraphStructure(nvgraphCSCTopology32I topologyData)
		//{
		//	res = NVGraphNativeMathods.nvgraphSetGraphStructure(_context, _descr, ref topologyData, nvgraphTopologyType.CSC32);
		//	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSetGraphStructure", res));
		//	if (res != nvgraphStatus.Success) throw new NVGraphException(res);
  //      }

        public void SetGraphStructure(nvgraphTopologyBase topologyData)
        {
            nvgraphTopologyType type;
            if (topologyData is nvgraphCSRTopology32I)
            {
                type = nvgraphTopologyType.CSR32;
            }
            else
            {
                if (topologyData is nvgraphCSCTopology32I)
                {
                    type = nvgraphTopologyType.CSC32;
                }
                else
                {
                    type = nvgraphTopologyType.COO32;
                }
            }


            res = NVGraphNativeMathods.nvgraphSetGraphStructure(_context, _descr, topologyData, type);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSetGraphStructure", res));
            if (res != nvgraphStatus.Success) throw new NVGraphException(res);
        }

  //      public void GetGraphStructure(ref nvgraphCSRTopology32I topologyData)
		//{
  //          nvgraphTopologyType type = nvgraphTopologyType.CSR32;
  //          res = NVGraphNativeMathods.nvgraphGetGraphStructure(_context, _descr, ref topologyData, ref type);
		//	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphGetGraphStructure", res));
		//	if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		//}

		//public void GetGraphStructure(ref nvgraphCSCTopology32I topologyData)
  //      {
  //          nvgraphTopologyType type = nvgraphTopologyType.CSC32;
  //          res = NVGraphNativeMathods.nvgraphGetGraphStructure(_context, _descr, ref topologyData, ref type);
		//	Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphGetGraphStructure", res));
		//	if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		//}

		public void GetGraphStructure(nvgraphTopologyBase topologyData, ref nvgraphTopologyType type)
        {
            res = NVGraphNativeMathods.nvgraphGetGraphStructure(_context, _descr, topologyData, ref type);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphGetGraphStructure", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		}



		public void AllocateVertexData(cudaDataType[] settypes)
		{
			res = NVGraphNativeMathods.nvgraphAllocateVertexData(_context, _descr, settypes.Length, settypes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphAllocateVertexData", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		}

		public void AllocateEdgeData(cudaDataType[] settypes)
		{
			res = NVGraphNativeMathods.nvgraphAllocateEdgeData(_context, _descr, settypes.Length, settypes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphAllocateEdgeData", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		}

		public void SetVertexData(Array vertexData, SizeT setnum)
		{
			GCHandle handle = GCHandle.Alloc(vertexData, GCHandleType.Pinned);
			try
			{
				res = NVGraphNativeMathods.nvgraphSetVertexData(_context, _descr, handle.AddrOfPinnedObject(), setnum);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSetVertexData", res));
				if (res != nvgraphStatus.Success) throw new NVGraphException(res);
			}
			finally
			{
				handle.Free();
			}
        }

        public void SetVertexData<Type>(CudaDeviceVariable<Type> vertexData, SizeT setnum) where Type : struct
        {
            res = NVGraphNativeMathods.nvgraphSetVertexData(_context, _descr, vertexData.DevicePointer, setnum);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSetVertexData", res));
            if (res != nvgraphStatus.Success) throw new NVGraphException(res);            
        }

        public void GetVertexData(Array vertexData, SizeT setnum)
		{
			GCHandle handle = GCHandle.Alloc(vertexData, GCHandleType.Pinned);
			try
			{
				res = NVGraphNativeMathods.nvgraphGetVertexData(_context, _descr, handle.AddrOfPinnedObject(), setnum);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphGetVertexData", res));
				if (res != nvgraphStatus.Success) throw new NVGraphException(res);
			}
			finally
			{
				handle.Free();
			}
        }

        public void GetVertexData<Type>(CudaDeviceVariable<Type> vertexData, SizeT setnum) where Type : struct
        {
            res = NVGraphNativeMathods.nvgraphGetVertexData(_context, _descr, vertexData.DevicePointer, setnum);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphGetVertexData", res));
            if (res != nvgraphStatus.Success) throw new NVGraphException(res);
        }

        public void SetEdgeData(Array edgeData, SizeT setnum)
		{
			GCHandle handle = GCHandle.Alloc(edgeData, GCHandleType.Pinned);
			try
			{
				res = NVGraphNativeMathods.nvgraphSetEdgeData(_context, _descr, handle.AddrOfPinnedObject(), setnum);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSetEdgeData", res));
				if (res != nvgraphStatus.Success) throw new NVGraphException(res);
			}
			finally
			{
				handle.Free();
			}
        }

        public void SetEdgeData<Type>(CudaDeviceVariable<Type> edgeData, SizeT setnum) where Type : struct
        {
            res = NVGraphNativeMathods.nvgraphSetEdgeData(_context, _descr, edgeData.DevicePointer, setnum);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSetEdgeData", res));
            if (res != nvgraphStatus.Success) throw new NVGraphException(res);
        }

        public void GetEdgeData(Array edgeData, SizeT setnum)
		{
			GCHandle handle = GCHandle.Alloc(edgeData, GCHandleType.Pinned);
			try
			{
				res = NVGraphNativeMathods.nvgraphGetEdgeData(_context, _descr, handle.AddrOfPinnedObject(), setnum);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphGetEdgeData", res));
				if (res != nvgraphStatus.Success) throw new NVGraphException(res);
			}
			finally
			{
				handle.Free();
			}
        }

        public void GetEdgeData<Type>(CudaDeviceVariable<Type> edgeData, SizeT setnum) where Type : struct
        {
            res = NVGraphNativeMathods.nvgraphGetEdgeData(_context, _descr, edgeData.DevicePointer, setnum);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphGetEdgeData", res));
            if (res != nvgraphStatus.Success) throw new NVGraphException(res);
        }

        public GraphDescriptor ExtractSubgraphByVertex(int[] subvertices)
		{
			GraphDescriptor subdescr = new GraphDescriptor(_context);

			res = NVGraphNativeMathods.nvgraphExtractSubgraphByVertex(_context, _descr, subdescr.GraphDescr, subvertices, subvertices.Length);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphExtractSubgraphByVertex", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);

			return subdescr;
		}

		public GraphDescriptor ExtractSubgraphByEdge(int[] subedges)
		{
			GraphDescriptor subdescr = new GraphDescriptor(_context);

			res = NVGraphNativeMathods.nvgraphExtractSubgraphByEdge(_context, _descr, subdescr.GraphDescr, subedges, subedges.Length);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphExtractSubgraphByEdge", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);

			return subdescr;
		}

		public void SrSpmv(SizeT weight_index, Array alpha, SizeT x_index, Array beta, SizeT y_index, nvgraphSemiring SR)
		{
			GCHandle alphaHandle = GCHandle.Alloc(alpha, GCHandleType.Pinned);
			GCHandle betaHandle = GCHandle.Alloc(beta, GCHandleType.Pinned);
			try
			{
				res = NVGraphNativeMathods.nvgraphSrSpmv(_context, _descr, weight_index, alphaHandle.AddrOfPinnedObject(), x_index, betaHandle.AddrOfPinnedObject(), y_index, SR);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSrSpmv", res));
				if (res != nvgraphStatus.Success) throw new NVGraphException(res);
			}
			finally
			{
				alphaHandle.Free();
				betaHandle.Free();
			}
		}

		public void Sssp(SizeT weight_index, ref int source_vert, SizeT sssp_index)
		{
			res = NVGraphNativeMathods.nvgraphSssp(_context, _descr, weight_index, ref source_vert, sssp_index);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphSssp", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		}

		public void WidestPath(SizeT weight_index, ref int source_vert, SizeT widest_path_index)
		{
			res = NVGraphNativeMathods.nvgraphWidestPath(_context, _descr, weight_index, ref source_vert, widest_path_index);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphWidestPath", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);
		}

		public void Pagerank(SizeT weight_index, Array alpha, SizeT bookmark_index, int has_guess, SizeT pagerank_index, float tolerance, int max_iter)
		{
			GCHandle alphaHandle = GCHandle.Alloc(alpha, GCHandleType.Pinned);
			try
			{
				res = NVGraphNativeMathods.nvgraphPagerank(_context, _descr, weight_index, alphaHandle.AddrOfPinnedObject(), bookmark_index, has_guess, pagerank_index, tolerance, max_iter);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphPagerank", res));
				if (res != nvgraphStatus.Success) throw new NVGraphException(res);
			}
			finally
			{
				alphaHandle.Free();
			}

		}
	}
}
