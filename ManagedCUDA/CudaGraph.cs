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
    /// Represents a Cuda graph. On dispose() all graph nodes will be distroyed, too!
    /// </summary>
    public class CudaGraph : IDisposable
    {
        private bool disposed;
        private CUResult res;
        private CUgraph _graph;

        #region Constructor
        /// <summary>
        /// Creates a new CudaGraph
        /// </summary>
        public CudaGraph()
        {
            _graph = new CUgraph();

            res = DriverAPINativeMethods.GraphManagment.cuGraphCreate(ref _graph, 0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For clone graph method
        /// </summary>
        internal CudaGraph(CUgraph graph)
        {
            _graph = graph;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaGraph()
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
                SizeT numNodes = new SizeT();
                res = DriverAPINativeMethods.GraphManagment.cuGraphGetNodes(_graph, null, ref numNodes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetNodes", res));
                //if (res != CUResult.Success) throw new CudaException(res);

                if (numNodes > 0)
                {
                    CUgraphNode[] nodes = new CUgraphNode[numNodes];
                    res = DriverAPINativeMethods.GraphManagment.cuGraphGetNodes(_graph, nodes, ref numNodes);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetNodes", res));
                    //if (res != CUResult.Success) throw new CudaException(res);

                    for (SizeT i = 0; i < numNodes; i++)
                    {
                        res = DriverAPINativeMethods.GraphManagment.cuGraphDestroyNode(nodes[i]);
                        Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphDestroyNode", res));
                        //if (res != CUResult.Success) throw new CudaException(res);
                    }
                }

                res = DriverAPINativeMethods.GraphManagment.cuGraphDestroy(_graph);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods

        /// <summary>
        /// Creates an empty node and adds it to a graph<para/>
        /// Creates a new node which performs no operation, and adds it to to the graph with
        /// dependencies specified via dependencies.
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.
        /// <para/>
        /// An empty node performs no operation during execution, but can be used for
        /// transitive ordering. For example, a phased execution graph with 2 groups of n
        /// nodes with a barrier between them can be represented using an empty node and
        /// 2*n dependency edges, rather than no empty node and n^2 dependency edges.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddEmptyNode(CUgraphNode[] dependencies)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddEmptyNode(ref node, _graph, dependencies, numDependencies);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddEmptyNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a memset node and adds it to a graph<para/>
        /// Creates a new memset node and adds it to graph with 
        /// dependencies specified via dependencies.<para/>
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.<para/>
        /// The element size must be 1, 2, or 4 bytes.<para/>
        /// When the graph is launched, the node will perform the memset described by memsetParams.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="memsetParams">When the graph is launched, the node will perform the memset described by memsetParams.</param>
        /// <param name="ctx">Cuda context used for the operation</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddMemsetNode(CUgraphNode[] dependencies, CudaMemsetNodeParams memsetParams, CudaContext ctx)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddMemsetNode(ref node, _graph, dependencies, numDependencies, ref memsetParams, ctx.Context);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddMemsetNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a memset node and adds it to a graph<para/>
        /// Creates a new memset node and adds it to graph with 
        /// dependencies specified via dependencies.<para/>
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.<para/>
        /// The element size must be 1, 2, or 4 bytes.<para/>
        /// When the graph is launched, the node will perform the memset described by memsetParams.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="deviceVariable">When the graph is launched, the node will perform the memset on deviceVariable.</param>
        /// <param name="value">Value to set</param>
        /// <param name="ctx">Cuda context used for the operation</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddMemsetNode<T>(CUgraphNode[] dependencies, CudaDeviceVariable<T> deviceVariable, uint value, CudaContext ctx) where T : struct
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            CudaMemsetNodeParams memsetParams = CudaMemsetNodeParams.init<T>(deviceVariable, value);

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddMemsetNode(ref node, _graph, dependencies, numDependencies, ref memsetParams, ctx.Context);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddMemsetNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a memset node and adds it to a graph<para/>
        /// Creates a new memset node and adds it to graph with 
        /// dependencies specified via dependencies.<para/>
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.<para/>
        /// The element size must be 1, 2, or 4 bytes.<para/>
        /// When the graph is launched, the node will perform the memset described by memsetParams.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="deviceVariable">When the graph is launched, the node will perform the memset on deviceVariable.</param>
        /// <param name="value">Value to set</param>
        /// <param name="ctx">Cuda context used for the operation</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddMemsetNode<T>(CUgraphNode[] dependencies, CudaPitchedDeviceVariable<T> deviceVariable, uint value, CudaContext ctx) where T : struct
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            CudaMemsetNodeParams memsetParams = CudaMemsetNodeParams.init<T>(deviceVariable, value);

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddMemsetNode(ref node, _graph, dependencies, numDependencies, ref memsetParams, ctx.Context);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddMemsetNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a memcpy node and adds it to a graph<para/>
        /// Creates a new memcpy node and adds it to graph with
        /// dependencies specified via dependencies.<para/>
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.
        /// A handle to the new node will be returned.<para/>
        /// When the graph is launched, the node will perform the memcpy described by copyParams.
        /// See ::cuMemcpy3D() for a description of the structure and its restrictions.<para/>
        /// Memcpy nodes have some additional restrictions with regards to managed memory, if the
        /// system contains at least one device which has a zero value for the device attribute
        /// ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. If one or more of the operands refer
        /// to managed memory, then using the memory type ::CU_MEMORYTYPE_UNIFIED is disallowed
        /// for those operand(s). The managed memory will be treated as residing on either the
        /// host or the device, depending on which memory type is specified.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="copyParams">Parameters for the memory copy</param>
        /// <param name="ctx">Cuda context used for the operation</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddMemcpyNode(CUgraphNode[] dependencies, CUDAMemCpy3D copyParams, CudaContext ctx)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddMemcpyNode(ref node, _graph, dependencies, numDependencies, ref copyParams, ctx.Context);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddMemcpyNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a kernel execution node and adds it to a graph<para/>
        /// Creates a new kernel execution node and adds it to the graph with
        /// dependencies specified via dependencies and arguments specified in nodeParams.<para/>
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.<para/>
        /// A handle to the new node will be returned.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="nodeParams">Parameters for the GPU execution node</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddKernelNode(CUgraphNode[] dependencies, CudaKernelNodeParams nodeParams)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddKernelNode(ref node, _graph, dependencies, numDependencies, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddKernelNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a kernel execution node and adds it to a graph<para/>
        /// Creates a new kernel execution node and adds it to the graph with
        /// dependencies specified via dependencies and arguments specified in nodeParams.<para/>
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.<para/>
        /// A handle to the new node will be returned.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="kernel">Kernel to execute</param>
        /// <param name="parameters">Kernel parameters to pass. An Array of IntPtr each of them pointing to a parameters. Note that the parameters must be pinned by GC!</param>
        /// <param name="extras">Extra data</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddKernelNode(CUgraphNode[] dependencies, CudaKernel kernel, IntPtr parameters, IntPtr extras)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            CudaKernelNodeParams nodeParams = new CudaKernelNodeParams
            {
                blockDimX = kernel.BlockDimensions.x,
                blockDimY = kernel.BlockDimensions.y,
                blockDimZ = kernel.BlockDimensions.z,

                extra = extras,
                func = kernel.CUFunction,

                gridDimX = kernel.GridDimensions.x,
                gridDimY = kernel.GridDimensions.y,
                gridDimZ = kernel.GridDimensions.z,

                kernelParams = parameters,
                sharedMemBytes = kernel.DynamicSharedMemory
            };

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddKernelNode(ref node, _graph, dependencies, numDependencies, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddKernelNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a child graph node and adds it to a graph<para/>
        /// Creates a new node which executes an embedded graph, and adds it to this Graph with
        /// dependencies specified via dependencies.
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.
        /// A handle to the new node will be returned.<para/>
        /// The node executes an embedded child graph. The child graph is cloned in this call.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="childGraph"></param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddChildGraphNode(CUgraphNode[] dependencies, CudaGraph childGraph)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddChildGraphNode(ref node, _graph, dependencies, numDependencies, childGraph.Graph);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddChildGraphNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a host execution node and adds it to a graph<para/>
        /// Creates a new CPU execution node and adds it to the graph with
        /// dependencies specified via dependencies.
        /// It is possible for dependencies to be null, in which case the node will be placed
        /// at the root of the graph. Dependencies may not have any duplicate entries.
        /// A handle to the new node will be returned.<para/>
        /// When the graph is launched, the node will invoke the specified CPU function.
        /// </summary>
        /// <param name="dependencies">can be null</param>
        /// <param name="hostFunction">Host function to execute</param>
        /// <param name="userData">User data for host function. Note that the data object must be pinned by GC!</param>
        /// <returns>A handle to the new node will be returned.</returns>
        public CUgraphNode AddHostNode(CUgraphNode[] dependencies, CUhostFn hostFunction, IntPtr userData)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            CudaHostNodeParams nodeParams = new CudaHostNodeParams
            {
                fn = hostFunction,
                userData = userData
            };

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddHostNode(ref node, _graph, dependencies, numDependencies, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddHostNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates an event record node and adds it to a graph
        /// Creates a new event record node and adds it to \p hGraph with \p numDependencies
        /// dependencies specified via \p dependencies and arguments specified in \p params.
        /// It is possible for \p numDependencies to be 0, in which case the node will be placed
        /// at the root of the graph. \p dependencies may not have any duplicate entries.
        /// A handle to the new node will be returned in \p phGraphNode.
        /// Each launch of the graph will record \p event to capture execution of the
        /// node's dependencies.
        /// </summary>
        /// <param name="dependencies">Dependencies of the node</param>
        /// <param name="event_">Event for the node</param>
        /// <returns>Returns newly created node</returns>
        public CUgraphNode AddEventRecordNode(CUgraphNode[] dependencies, CudaEvent event_)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddEventRecordNode(ref node, _graph, dependencies, numDependencies, event_.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddEventRecordNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates an event wait node and adds it to a graph
        /// Creates a new event wait node and adds it to \p hGraph with \p numDependencies
        /// dependencies specified via \p dependencies and arguments specified in \p params.
        /// It is possible for \p numDependencies to be 0, in which case the node will be placed
        /// at the root of the graph. \p dependencies may not have any duplicate entries.
        /// A handle to the new node will be returned in \p phGraphNode.
        /// The graph node will wait for all work captured in \p event.  See ::cuEventRecord()
        /// for details on what is captured by an event. \p event may be from a different context
        /// or device than the launch stream.
        /// </summary>
        /// <param name="dependencies">Dependencies of the node</param>
        /// <param name="event_">Event for the node</param>
        /// <returns>Returns newly created node</returns>
        public CUgraphNode AddEventWaitNode(CUgraphNode[] dependencies, CudaEvent event_)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddEventWaitNode(ref node, _graph, dependencies, numDependencies, event_.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddEventWaitNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates an external semaphore signal node and adds it to a graph<para/>
        /// Creates a new external semaphore signal node and adds it to \p hGraph with \p
        /// numDependencies dependencies specified via \p dependencies and arguments specified
        /// in \p nodeParams.It is possible for \p numDependencies to be 0, in which case the
        /// node will be placed at the root of the graph. \p dependencies may not have any
        /// duplicate entries. A handle to the new node will be returned in \p phGraphNode.
        /// </summary>
        /// <param name="dependencies">Dependencies of the node</param>
        /// <param name="nodeParams">Parameters for the node</param>
        /// <returns>Returns newly created node</returns>
        public CUgraphNode AddExternalSemaphoresSignalNode(CUgraphNode[] dependencies, CudaExtSemSignalNodeParams nodeParams)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddExternalSemaphoresSignalNode(ref node, _graph, dependencies, numDependencies, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddExternalSemaphoresSignalNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates an external semaphore wait node and adds it to a graph<para/>
        /// Creates a new external semaphore wait node and adds it to \p hGraph with \p numDependencies
        /// dependencies specified via \p dependencies and arguments specified in \p nodeParams.
        /// It is possible for \p numDependencies to be 0, in which case the node will be placed
        /// at the root of the graph. \p dependencies may not have any duplicate entries. A handle
        /// to the new node will be returned in \p phGraphNode.
        /// </summary>
        /// <param name="dependencies">Dependencies of the node</param>
        /// <param name="nodeParams">Parameters for the node</param>
        /// <returns>Returns newly created node</returns>
        public CUgraphNode AddExternalSemaphoresWaitNode(CUgraphNode[] dependencies, CudaExtSemWaitNodeParams nodeParams)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddExternalSemaphoresWaitNode(ref node, _graph, dependencies, numDependencies, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddExternalSemaphoresWaitNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }


        /// <summary>
        /// Creates an allocation node and adds it to a graph<para/>
        /// Creates a new allocation node and adds it to \p hGraph with \p numDependencies
        /// dependencies specified via \p dependencies and arguments specified in \p nodeParams.
        /// It is possible for \p numDependencies to be 0, in which case the node will be placed
        /// at the root of the graph. \p dependencies may not have any duplicate entries. A handle
        /// to the new node will be returned in \p phGraphNode.
        /// </summary>
        /// <param name="dependencies">Dependencies of the node</param>
        /// <param name="nodeParams">Parameters for the node</param>
        /// <returns>Returns newly created node</returns>
        public CUgraphNode AddMemAllocNode(CUgraphNode[] dependencies, ref CudaMemAllocNodeParams nodeParams)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddMemAllocNode(ref node, _graph, dependencies, numDependencies, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddMemAllocNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a memory free node and adds it to a graph<para/>
        /// Creates a new memory free node and adds it to \p hGraph with \p numDependencies
        /// dependencies specified via \p dependencies and arguments specified in \p nodeParams.
        /// It is possible for \p numDependencies to be 0, in which case the node will be placed
        /// at the root of the graph. \p dependencies may not have any duplicate entries. A handle
        /// to the new node will be returned in \p phGraphNode.
        /// </summary>
        /// <param name="dependencies">Dependencies of the node</param>
        /// <param name="dptr">Parameters for the node</param>
        /// <returns>Returns newly created node</returns>
        public CUgraphNode AddMemFreeNode(CUgraphNode[] dependencies, CUdeviceptr dptr)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddMemFreeNode(ref node, _graph, dependencies, numDependencies, dptr);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddMemFreeNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }

        /// <summary>
        /// Creates a memory free node and adds it to a graph<para/>
        /// Creates a new memory free node and adds it to \p hGraph with \p numDependencies
        /// dependencies specified via \p dependencies and arguments specified in \p nodeParams.
        /// It is possible for \p numDependencies to be 0, in which case the node will be placed
        /// at the root of the graph. \p dependencies may not have any duplicate entries. A handle
        /// to the new node will be returned in \p phGraphNode.
        /// </summary>
        /// <param name="dependencies">Dependencies of the node</param>
        /// <param name="nodeParams">Parameters for the node</param>
        /// <returns>Returns newly created node</returns>
        public CUgraphNode AddBatchMemOpNode(CUgraphNode[] dependencies, ref CudaBatchMemOpNodeParams nodeParams)
        {
            CUgraphNode node = new CUgraphNode();
            SizeT numDependencies = 0;
            if (dependencies != null)
                numDependencies = dependencies.Length;

            res = DriverAPINativeMethods.GraphManagment.cuGraphAddBatchMemOpNode(ref node, _graph, dependencies, numDependencies, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddBatchMemOpNode", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return node;
        }



        /// <summary>
        /// Clones a graph<para/>
        /// This function creates a copy of the original Graph.
        /// All parameters are copied into the cloned graph. The original graph may be modified
        /// after this call without affecting the clone.<para/>
        /// Child graph nodes in the original graph are recursively copied into the clone.
        /// </summary>
        public CudaGraph Clone()
        {
            CUgraph clone = new CUgraph();

            res = DriverAPINativeMethods.GraphManagment.cuGraphClone(ref clone, _graph);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphClone", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return new CudaGraph(clone);
        }

        /// <summary>
        /// Finds a cloned version of a node<para/>
        /// This function returns the node corresponding to originalNode
        /// in the original graph.<para/>
        /// This cloned graph must have been cloned from the original Graph via its Clone() method.
        /// OriginalNode must have been in that graph at the time of the call to
        /// Clone(), and the corresponding cloned node in this graph must not have
        /// been removed. The cloned node is then returned.
        /// </summary>
        /// <param name="originalNode"></param>
        public CUgraphNode NodeFindInClone(CUgraphNode originalNode)
        {
            CUgraphNode clone = new CUgraphNode();

            res = DriverAPINativeMethods.GraphManagment.cuGraphNodeFindInClone(ref clone, originalNode, _graph);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeFindInClone", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return clone;
        }

        /// <summary>
        /// Returns a graph's nodes
        /// </summary>
        /// <returns></returns>
        public CUgraphNode[] GetNodes()
        {
            SizeT numNodes = new SizeT();
            res = DriverAPINativeMethods.GraphManagment.cuGraphGetNodes(_graph, null, ref numNodes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetNodes", res));
            if (res != CUResult.Success) throw new CudaException(res);

            if (numNodes > 0)
            {
                CUgraphNode[] nodes = new CUgraphNode[numNodes];
                res = DriverAPINativeMethods.GraphManagment.cuGraphGetNodes(_graph, nodes, ref numNodes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetNodes", res));
                if (res != CUResult.Success) throw new CudaException(res);

                return nodes;
            }

            return null;
        }

        /// <summary>
        /// Returns a graph's root nodes
        /// </summary>
        /// <returns></returns>
        public CUgraphNode[] GetRootNodes()
        {
            SizeT numNodes = new SizeT();
            res = DriverAPINativeMethods.GraphManagment.cuGraphGetRootNodes(_graph, null, ref numNodes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetRootNodes", res));
            if (res != CUResult.Success) throw new CudaException(res);

            if (numNodes > 0)
            {
                CUgraphNode[] nodes = new CUgraphNode[numNodes];
                res = DriverAPINativeMethods.GraphManagment.cuGraphGetNodes(_graph, nodes, ref numNodes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetRootNodes", res));
                if (res != CUResult.Success) throw new CudaException(res);

                return nodes;
            }

            return null;
        }

        /// <summary>
        /// Returns a graph's dependency edges
        /// </summary>
        /// <param name="from"></param>
        /// <param name="to"></param>
        public void GetEdges(out CUgraphNode[] from, out CUgraphNode[] to)
        {
            from = null;
            to = null;
            SizeT numNodes = new SizeT();
            res = DriverAPINativeMethods.GraphManagment.cuGraphGetEdges(_graph, null, null, ref numNodes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetEdges", res));
            if (res != CUResult.Success) throw new CudaException(res);

            if (numNodes > 0)
            {
                from = new CUgraphNode[numNodes];
                to = new CUgraphNode[numNodes];
                res = DriverAPINativeMethods.GraphManagment.cuGraphGetEdges(_graph, from, to, ref numNodes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphGetNodes", res));
                if (res != CUResult.Success) throw new CudaException(res);
            }
        }

        /// <summary>
        /// Adds dependency edges to a graph
        /// Elements in from and to at corresponding indices define a dependency.<para/>
        /// Each node in from and to must belong to this Graph.<para/>
        /// Specifying an existing dependency will return an error.
        /// </summary>
        /// <param name="from"></param>
        /// <param name="to"></param>
        public void AddDependencies(CUgraphNode[] from, CUgraphNode[] to)
        {
            if (from.Length != to.Length)
            {
                throw new ArgumentException("from and to must have equal size!");
            }

            SizeT numNodes = from.Length;
            res = DriverAPINativeMethods.GraphManagment.cuGraphAddDependencies(_graph, from, to, numNodes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphAddDependencies", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Removes dependency edges to a graph
        /// Elements in from and to at corresponding indices define a dependency.<para/>
        /// Each node in from and to must belong to this Graph.<para/>
        /// Specifying an existing dependency will return an error.
        /// </summary>
        /// <param name="from"></param>
        /// <param name="to"></param>
        public void RemoveDependencies(CUgraphNode[] from, CUgraphNode[] to)
        {
            if (from.Length != to.Length)
            {
                throw new ArgumentException("from and to must have equal size!");
            }

            SizeT numNodes = from.Length;
            res = DriverAPINativeMethods.GraphManagment.cuGraphRemoveDependencies(_graph, from, to, numNodes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphRemoveDependencies", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates an executable graph from a graph<para/>
        /// Instantiates this Graph as an executable graph. The graph is validated for any
        /// structural constraints or intra-node constraints which were not previously
        /// validated. If instantiation is successful, a handle to the instantiated graph
        /// is returned.
        /// </summary>
        public CudaGraphExec Instantiate(ref CudaGraphInstantiateParams parameters)
        {
            CUgraphExec graphExec = new CUgraphExec();
            res = DriverAPINativeMethods.GraphManagment.cuGraphInstantiateWithParams(ref graphExec, _graph, ref parameters);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphInstantiateWithParams", res));
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
            return new CudaGraphExec(graphExec);
        }

        /// <summary>
        /// Creates an executable graph from a graph<para/>
        /// Instantiates \p hGraph as an executable graph. The graph is validated for any
        /// structural constraints or intra-node constraints which were not previously
        /// validated.If instantiation is successful, a handle to the instantiated graph
        /// is returned in \p phGraphExec.
        /// </summary>
        public CudaGraphExec Instantiate(CUgraphInstantiate_flags flags)
        {
            CUgraphExec graphExec = new CUgraphExec();
            res = DriverAPINativeMethods.GraphManagment.cuGraphInstantiate(ref graphExec, _graph, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphInstantiate", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return new CudaGraphExec(graphExec);
        }

        /// <summary>
        /// Write a DOT file describing graph structure<para/>
        /// Using the provided \p hGraph, write to \p path a DOT formatted description of the graph.
        /// By default this includes the graph topology, node types, node id, kernel names and memcpy direction.
        /// \p flags can be specified to write more detailed information about each node type such as
        /// parameter values, kernel attributes, node and function handles.
        /// </summary>
        /// <param name="path">The path to write the DOT file to</param>
        /// <param name="flags">Flags from CUgraphDebugDot_flags for specifying which additional node information to write</param>
        public void DebugDotPrint(string path, CUgraphDebugDot_flags flags)
        {
            res = DriverAPINativeMethods.GraphManagment.cuGraphDebugDotPrint(_graph, path, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphDebugDotPrint", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion

        #region Properties

        /// <summary>
        /// Returns the inner graph handle
        /// </summary>
        public CUgraph Graph
        {
            get { return _graph; }
        }
        #endregion
    }
}
