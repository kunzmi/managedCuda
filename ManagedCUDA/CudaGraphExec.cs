// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
// http://kunzmi.github.io/managedCuda
//
// This file is part of ManagedCuda.
//
// Commercial License Usage
//  Licensees holding valid commercial ManagedCuda licenses may use this
//  file in accordance with the commercial license agreement provided with
//  the Software or, alternatively, in accordance with the terms contained
//  in a written agreement between you and Artic Imaging SARL. For further
//  information contact us at managedcuda@articimaging.eu.
//  
// GNU General Public License Usage
//  Alternatively, this file may be used under the terms of the GNU General
//  Public License as published by the Free Software Foundation, either 
//  version 3 of the License, or (at your option) any later version.
//  
//  ManagedCuda is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <http://www.gnu.org/licenses/>.


using System;
using ManagedCuda.BasicTypes;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Represents an executable Cuda graph.
    /// </summary>
    public class CudaGraphExec : IDisposable
    {
        private bool disposed;
        private CUResult res;
        private CUgraphExec _graph;

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
        /// Sets the parameters for an external semaphore signal node in the given graphExec<para/>
        /// Sets the parameters of an external semaphore signal node in an executable graph \p hGraphExec.
        /// The node is identified by the corresponding node \p hNode in the
        /// non-executable graph, from which the executable graph was instantiated.<para/>
        /// hNode must not have been removed from the original graph.<para/>
        /// The modifications only affect future launches of \p hGraphExec. Already
        /// enqueued or running launches of \p hGraphExec are not affected by this call.
        /// hNode is also not modified by this call.<para/>
        /// Changing \p nodeParams->numExtSems is not supported.
        /// </summary>
        public void SetParams(CUgraphNode hNode, CudaExtSemSignalNodeParams nodeParams)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecExternalSemaphoresSignalNodeSetParams(_graph, hNode, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecExternalSemaphoresSignalNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets the parameters for an external semaphore wait node in the given graphExec<para/>
        /// Sets the parameters of an external semaphore wait node in an executable graph \p hGraphExec.<para/>
        /// The node is identified by the corresponding node \p hNode in the
        /// non-executable graph, from which the executable graph was instantiated.<para/>
        /// hNode must not have been removed from the original graph.<para/>
        /// The modifications only affect future launches of \p hGraphExec. Already
        /// enqueued or running launches of \p hGraphExec are not affected by this call.
        /// hNode is also not modified by this call.<para/>
        /// Changing \p nodeParams->numExtSems is not supported.
        /// </summary>
        public void SetParams(CUgraphNode hNode, CudaExtSemWaitNodeParams nodeParams)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecExternalSemaphoresWaitNodeSetParams(_graph, hNode, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecExternalSemaphoresWaitNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets the parameters for a batch mem op node in the given graphExec<para/>
        /// Sets the parameters of a batch mem op node in an executable graph \p hGraphExec.<para/>
        /// The node is identified by the corresponding node \p hNode in the 
        /// non-executable graph, from which the executable graph was instantiated.<para/>
        /// The following fields on operations may be modified on an executable graph:<para/>
        /// op.waitValue.address<para/>
        /// op.waitValue.value[64]<para/>
        /// op.waitValue.flags bits corresponding to wait type (i.e.CU_STREAM_WAIT_VALUE_FLUSH bit cannot be modified)<para/>
        /// op.writeValue.address<para/>
        /// op.writeValue.value[64]<para/>
        /// Other fields, such as the context, count or type of operations, and other types of operations such as membars, may not be modified.<para/>
        /// \p hNode must not have been removed from the original graph.<para/>
        /// The modifications only affect future launches of \p hGraphExec. Already
        /// enqueued or running launches of \p hGraphExec are not affected by this call.<para/>
        /// \p hNode is also not modified by this call.<para/>
        /// The paramArray inside \p nodeParams is copied and therefore it can be
        /// freed after the call returns.
        /// </summary>
        public void SetParams(CUgraphNode hNode, CudaBatchMemOpNodeParams nodeParams)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecBatchMemOpNodeSetParams(_graph, hNode, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecBatchMemOpNodeSetParams", res));
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


        /// <summary>
        /// Check whether an executable graph can be updated with a graph and perform the update if possible<para/>
        /// Updates the node parameters in the instantiated graph specified by \p hGraphExec with the node parameters in a topologically identical graph specified by \p hGraph.<para/>
        /// Limitations:<para/>
        /// - Kernel nodes:<para/>
        /// - The owning context of the function cannot change.<para/>
        /// - A node whose function originally did not use CUDA dynamic parallelism cannot be updated
        /// to a function which uses CDP.<para/>
        /// - A cooperative node cannot be updated to a non-cooperative node, and vice-versa.<para/>
        /// - If the graph was instantiated with CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, the
        /// priority attribute cannot change.Equality is checked on the originally requested
        /// priority values, before they are clamped to the device's supported range.<para/>
        /// - If \p hGraphExec was not instantiated for device launch, a node whose function originally did not use device-side cudaGraphLaunch() cannot be updated to a function which uses
        /// device-side cudaGraphLaunch() unless the node resides on the same context as nodes which contained such calls at instantiate-time.If no such calls were present at instantiation,
        /// these updates cannot be performed at all.<para/>
        /// - Memset and memcpy nodes:<para/>
        /// - The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.<para/>
        /// - The source/destination memory must be allocated from the same contexts as the original source/destination memory.<para/>
        /// - Only 1D memsets can be changed.<para/>
        /// - Additional memcpy node restrictions:<para/>
        /// - Changing either the source or destination memory type(i.e.CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_ARRAY, etc.) is not supported.<para/>
        /// - External semaphore wait nodes and record nodes:<para/>
        /// - Changing the number of semaphores is not supported.<para/>
        /// Note:  The API may add further restrictions in future releases.  The return code should always be checked.<para/>
        /// cuGraphExecUpdate sets the result member of \p resultInfo to CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED under the following conditions:
        /// - The count of nodes directly in \p hGraphExec and \p hGraph differ, in which case resultInfo->errorNode
        /// is set to NULL.<para/>
        /// - \p hGraph has more exit nodes than \p hGraph, in which case resultInfo->errorNode is set to one of the exit nodes in hGraph.
        /// - A node in \p hGraph has a different number of dependencies than the node from \p hGraphExec it is paired with,
        /// in which case resultInfo->errorNode is set to the node from \p hGraph.<para/>
        /// - A node in \p hGraph has a dependency that does not match with the corresponding dependency of the paired node
        /// from \p hGraphExec. resultInfo->errorNode will be set to the node from \p hGraph. resultInfo->errorFromNode
        /// will be set to the mismatched dependency. The dependencies are paired based on edge order and a dependency
        /// does not match when the nodes are already paired based on other edges examined in the graph.<para/>
        /// cuGraphExecUpdate sets the result member of \p resultInfo to: <para/>
        /// - CU_GRAPH_EXEC_UPDATE_ERROR if passed an invalid value.
        /// - CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED if the graph topology changed<para/>
        /// - CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED if the type of a node changed, in which case
        /// \p hErrorNode_out is set to the node from \p hGraph.<para/>
        /// - CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE if the function changed in an unsupported
        /// way(see note above), in which case \p hErrorNode_out is set to the node from \p hGraph<para/>
        /// - CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED if any parameters to a node changed in a way that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph.<para/>
        /// - CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED if any attributes of a node changed in a way that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph.<para/>
        /// - CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED if something about a node is unsupported, like the node's type or configuration, in which case \p hErrorNode_out is set to the node from \p hGraph<para/>
        /// If the update fails for a reason not listed above, the result member of \p resultInfo will be set
        /// to CU_GRAPH_EXEC_UPDATE_ERROR.If the update succeeds, the result member will be set to CU_GRAPH_EXEC_UPDATE_SUCCESS.<para/>
        /// cuGraphExecUpdate returns CUDA_SUCCESS when the updated was performed successfully.It returns 
        /// CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE if the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.
        /// </summary>
        /// <param name="hGraph">The graph containing the updated parameters</param>
        /// <param name="resultInfo">the error info structure</param>
        public void Update(CUgraph hGraph, ref CUgraphExecUpdateResultInfo resultInfo)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphExecUpdate(_graph, hGraph, ref resultInfo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecUpdate", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Enables or disables the specified node in the given graphExec<para/>
        /// Sets \p hNode to be either enabled or disabled.Disabled nodes are functionally equivalent 
        /// to empty nodes until they are reenabled.Existing node parameters are not affected by
        /// disabling/enabling the node.<para/>
        /// The node is identified by the corresponding node \p hNode in the non-executable
        /// graph, from which the executable graph was instantiated.<para/>
        /// \p hNode must not have been removed from the original graph.<para/>
        /// The modifications only affect future launches of \p hGraphExec. Already
        /// enqueued or running launches of \p hGraphExec are not affected by this call.
        /// \p hNode is also not modified by this call.<para/>
        /// \note Currently only kernel, memset and memcpy nodes are supported. 
        /// </summary>
        /// <param name="hNode">Node from the graph from which graphExec was instantiated</param>
        /// <param name="isEnabled">Node is enabled if != 0, otherwise the node is disabled</param>
        public void NodeSetEnabled(CUgraphNode hNode, bool isEnabled)
        {
            uint val = isEnabled ? (uint)1 : 0;
            res = DriverAPINativeMethods.GraphManagment.cuGraphNodeSetEnabled(_graph, hNode, val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeSetEnabled", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Query whether a node in the given graphExec is enabled<para/>
        /// Sets isEnabled to 1 if \p hNode is enabled, or 0 if \p hNode is disabled.<para/>
        /// The node is identified by the corresponding node \p hNode in the non-executable
        /// graph, from which the executable graph was instantiated.<para/>
        /// \p hNode must not have been removed from the original graph.<para/>
        /// \note Currently only kernel, memset and memcpy nodes are supported. 
        /// </summary>
        /// <param name="hNode">Node from the graph from which graphExec was instantiated</param>
        /// <returns>the enabled status of the node</returns>
        public bool NodeGetEnabled(CUgraphNode hNode)
        {
            int val = 0;
            res = DriverAPINativeMethods.GraphManagment.cuGraphNodeGetEnabled(_graph, hNode, ref val);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeSetEnabled", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return val != 0;
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

        /// <summary>
        /// Query the instantiation flags of an executable graph<para/>
        /// Returns the flags that were passed to instantiation for the given executable graph.
        /// ::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD will not be returned by this API as it does
        /// not affect the resulting executable graph.
        /// </summary>
        public CUgraphInstantiate_flags Flags
        {
            get
            {
                CUgraphInstantiate_flags flags = new CUgraphInstantiate_flags();
                res = DriverAPINativeMethods.GraphManagment.cuGraphExecGetFlags(_graph, ref flags);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExecGetFlags", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return flags;
            }
        }
        #endregion
    }
}
