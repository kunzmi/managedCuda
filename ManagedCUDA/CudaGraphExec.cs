//	Copyright (c) 2018, Michael Kunz. All rights reserved.
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
using ManagedCuda.BasicTypes;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Represents an executable Cuda graph.
    /// </summary>
    public class CudaGraphExec : IDisposable
    {
        bool disposed;
        CUResult res;
        CUgraphExec _graph;

        #region Constructor

        /// <summary>
        /// For clone graph method
        /// </summary>
        internal CudaGraphExec(CUgraphExec graph)
        {
            _graph = graph;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaGraphExec()
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
                res = DriverAPINativeMethods.GraphManagment.cuGraphExecDestroy(_graph);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods
        /// <summary>
        /// Launches an executable graph in a stream.<para/>
        /// Only one instance of GraphExec may be executing
        /// at a time. Each launch is ordered behind both any previous work in Stream
        /// and any previous launches of GraphExec.To execute a graph concurrently, it must be
        /// instantiated multiple times into multiple executable graphs.
        /// </summary>
        /// <param name="stream"></param>
        public void Launch(CudaStream stream)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphLaunch(_graph, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphLaunch", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Uploads an executable graph in a stream
        /// Uploads \p hGraphExec to the device in \p hStream without executing it.Uploads of
        /// the same \p hGraphExec will be serialized.Each upload is ordered behind both any
        /// previous work in \p hStream and any previous launches of \p hGraphExec.
        /// </summary>
        /// <param name="stream">Stream in which to upload the graph</param>
        /// <returns></returns>
        public void Upload(CudaStream stream)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphUpload(_graph, stream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphUpload", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }




        /// <summary>
        /// Sets the parameters for a kernel node in the given graphExec<para/>
        /// Sets the parameters of a kernel node in an executable graph \p hGraphExec.
        /// The node is identified by the corresponding node \p hNode in the
        /// non-executable graph, from which the executable graph was instantiated.<para/>
        /// \p hNode must not have been removed from the original graph.The \p func field
        /// of \p nodeParams cannot be modified and must match the original value.
        /// All other values can be modified.<para/>
        /// The modifications only affect future launches of \p hGraphExec. Already
        /// enqueued or running launches of \p hGraphExec are not affected by this call.
        /// \p hNode is also not modified by this call.
        /// </summary>
        /// <param name="hNode"></param>
        /// <param name="nodeParams"></param>
        public void SetParams(CUgraphNode hNode, ref CudaKernelNodeParams nodeParams)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecKernelNodeSetParams(_graph, hNode, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecKernelNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Sets the parameters for a memcpy node in the given graphExec.<para/>
        /// Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
        /// contained \p copyParams at instantiation.  hNode must remain in the graph which was 
        /// used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.<para/>
        /// The source and destination memory in \p copyParams must be allocated from the same 
        /// contexts as the original source and destination memory.  Both the instantiation-time 
        /// memory operands and the memory operands in \p copyParams must be 1-dimensional.
        /// Zero-length operations are not supported.<para/>
        /// The modifications only affect future launches of \p hGraphExec.  Already enqueued 
        /// or running launches of \p hGraphExec are not affected by this call.  hNode is also 
        /// not modified by this call.<para/>
        /// Returns CUDA_ERROR_INVALID_VALUE if the memory operands' mappings changed or
        /// either the original or new memory operands are multidimensional.
        /// </summary>
        public void SetParams(CUgraphNode hNode, ref CUDAMemCpy3D copyParams, CUcontext ctx)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecMemcpyNodeSetParams(_graph, hNode, ref copyParams, ctx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecMemcpyNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Sets the parameters for a memset node in the given graphExec.<para/>
        /// Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
        /// contained \p memsetParams at instantiation.  hNode must remain in the graph which was 
        /// used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.<para/>
        /// The destination memory in \p memsetParams must be allocated from the same 
        /// contexts as the original destination memory.  Both the instantiation-time 
        /// memory operand and the memory operand in \p memsetParams must be 1-dimensional.
        /// Zero-length operations are not supported.<para/>
        /// The modifications only affect future launches of \p hGraphExec.  Already enqueued 
        /// or running launches of \p hGraphExec are not affected by this call.  hNode is also 
        /// not modified by this call.<para/>
        /// Returns CUDA_ERROR_INVALID_VALUE if the memory operand's mappings changed or
        /// either the original or new memory operand are multidimensional.
        /// </summary>
        public void SetParams(CUgraphNode hNode, ref CudaMemsetNodeParams memsetParams, CUcontext ctx)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecMemsetNodeSetParams(_graph, hNode, ref memsetParams, ctx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecMemsetNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Sets the parameters for a host node in the given graphExec.<para/>
        /// Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
        /// contained \p nodeParams at instantiation.  hNode must remain in the graph which was 
        /// used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.<para/>
        /// The modifications only affect future launches of \p hGraphExec.  Already enqueued 
        /// or running launches of \p hGraphExec are not affected by this call.  hNode is also 
        /// not modified by this call.
        /// </summary>
        public void SetParams(CUgraphNode hNode, ref CudaHostNodeParams nodeParams)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecHostNodeSetParams(_graph, hNode, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecHostNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }





        /// <summary>
        /// Updates node parameters in the child graph node in the given graphExec.
        /// Updates the work represented by \p hNode in \p hGraphExec as though the nodes contained
        /// in \p hNode's graph had the parameters contained in \p childGraph's nodes at instantiation.
        /// \p hNode must remain in the graph which was used to instantiate \p hGraphExec.
        /// Changed edges to and from \p hNode are ignored.
        /// The modifications only affect future launches of \p hGraphExec.  Already enqueued 
        /// or running launches of \p hGraphExec are not affected by this call.  \p hNode is also
        /// not modified by this call.
        /// The topology of \p childGraph, as well as the node insertion order, must match that
        /// of the graph contained in \p hNode.  See::cuGraphExecUpdate() for a list of restrictions
        /// on what can be updated in an instantiated graph.The update is recursive, so child graph
        /// nodes contained within the top level child graph will also be updated.
        /// </summary>
        public void SetParams(CUgraphNode hNode, CudaGraph childGraph)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecChildGraphNodeSetParams(_graph, hNode, childGraph.Graph);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecChildGraphNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets the event for an event record node in the given graphExec
        /// Sets the event of an event record node in an executable graph \p hGraphExec.
        /// The node is identified by the corresponding node \p hNode in the
        /// non-executable graph, from which the executable graph was instantiated.
        /// The modifications only affect future launches of \p hGraphExec. Already
        /// enqueued or running launches of \p hGraphExec are not affected by this call.
        /// \p hNode is also not modified by this call.
        /// </summary>
        public void SetRecordEvent(CUgraphNode hNode, CudaEvent event_)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecEventRecordNodeSetEvent(_graph, hNode, event_.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecEventRecordNodeSetEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Sets the event for an event record node in the given graphExec
        /// Sets the event of an event record node in an executable graph \p hGraphExec.
        /// The node is identified by the corresponding node \p hNode in the
        /// non-executable graph, from which the executable graph was instantiated.
        /// The modifications only affect future launches of \p hGraphExec. Already
        /// enqueued or running launches of \p hGraphExec are not affected by this call.
        /// \p hNode is also not modified by this call.
        /// </summary>
        public void SetWaitEvent(CUgraphNode hNode, CudaEvent event_)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecEventWaitNodeSetEvent(_graph, hNode, event_.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecEventWaitNodeSetEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        #endregion

        #region Properties

        /// <summary>
        /// Returns the inner executable graph handle
        /// </summary>
        public CUgraphExec Graph
        {
            get { return _graph; }
        }
        #endregion
    }
}
