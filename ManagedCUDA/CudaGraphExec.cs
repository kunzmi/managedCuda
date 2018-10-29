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
