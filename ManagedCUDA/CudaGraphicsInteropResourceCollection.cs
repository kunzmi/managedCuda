//	Copyright (c) 2012, Michael Kunz. All rights reserved.
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
using System.Collections;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Groupes several wrapped CUgraphicsResources together, so that the map() call to the CUDA API can be efficiently on all
    /// resources together.
    /// </summary>
    public class CudaGraphicsInteropResourceCollection : ICollection<ICudaGraphicsInteropResource>, IDisposable
    {
        List<ICudaGraphicsInteropResource> _resources;
        List<CUgraphicsResource> _CUResources;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new CudaGraphicsInteropResourceCollection
        /// </summary>
        public CudaGraphicsInteropResourceCollection()
        {
            _resources = new List<ICudaGraphicsInteropResource>();
            _CUResources = new List<CUgraphicsResource>();
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaGraphicsInteropResourceCollection()
        {
            Dispose(false);
        }
        #endregion

        #region ICollection
        /// <summary>
        /// Returns the number of resources in the collection
        /// </summary>
        public int Count
        {
            get { return _resources.Count; }
        }

        bool ICollection<ICudaGraphicsInteropResource>.IsReadOnly
        {
            get { return false; }
        }

        /// <summary>
        /// Adds a new resource to the collection
        /// </summary>
        /// <param name="item"></param>
        public void Add(ICudaGraphicsInteropResource item)
        {
            if (!_resources.Contains(item))
            _resources.Add(item);
            _CUResources.Add(item.GetCUgraphicsResource());
        }

        /// <summary>
        /// Removes all resources in the collection, an disposes every element.
        /// </summary>
        public void Clear()
        {
            foreach (var elem in _resources)
                elem.Dispose();
            _resources.Clear();
            _CUResources.Clear();
        }

        /// <summary>
        /// Returns true, if the given resource is part of the collection
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Contains(ICudaGraphicsInteropResource item)
        {
            return _resources.Contains(item);
        }

        /// <summary>
        /// Throws NotImplementedException.
        /// </summary>
        /// <param name="array"></param>
        /// <param name="arrayIndex"></param>
        void ICollection<ICudaGraphicsInteropResource>.CopyTo(ICudaGraphicsInteropResource[] array, int arrayIndex)
        {
            throw new NotImplementedException("CopyTo is not implemented!");
        }

        /// <summary>
        /// Removes a resource from the collection. The resource is not disposed.
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Remove(ICudaGraphicsInteropResource item)
        {
            return _resources.Remove(item) && _CUResources.Remove(item.GetCUgraphicsResource()); ;
        }

        IEnumerator<ICudaGraphicsInteropResource> IEnumerable<ICudaGraphicsInteropResource>.GetEnumerator()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            return _resources.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            return _resources.GetEnumerator();
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
                foreach (var elem in _resources)
                    elem.Dispose();
                disposed = true;
            }
        }
        #endregion

        #region Properties
        /// <summary>
        /// Returns the ICudaGraphicsInteropResource at index index.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public ICudaGraphicsInteropResource this[int index]
        {
            get { return _resources[index]; }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Maps all graphics resources for access by CUDA.<para/>
        /// The resources may be accessed by CUDA until they are unmapped. The graphics API from which the resource
        /// was registered should not access any resources while they are mapped by CUDA. If an application does
        /// so, the results are undefined.<para/>
        /// This function provides the synchronization guarantee that any graphics calls issued before <see cref="MapAllResources()"/>
        /// will complete before any subsequent CUDA work issued in <c>stream</c> begins.<para/>
        /// If any of the resources is presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> exception is thrown.
        /// </summary>
        public void MapAllResources()
        {
            MapAllResources(new CUstream());
        }

        /// <summary>
        /// Maps all graphics resources for access by CUDA.<para/>
        /// The resources may be accessed by CUDA until they are unmapped. The graphics API from which the resource
        /// was registered should not access any resources while they are mapped by CUDA. If an application does
        /// so, the results are undefined.<para/>
        /// This function provides the synchronization guarantee that any graphics calls issued before <see cref="MapAllResources()"/>
        /// will complete before any subsequent CUDA work issued in <c>stream</c> begins.<para/>
        /// If any of the resources is presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> exception is thrown.
        /// </summary>
        /// <param name="stream"></param>
        public void MapAllResources(CUstream stream)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            CUResult res;
            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsMapResources((uint) _CUResources.Count, _CUResources.ToArray(), stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsMapResources", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            foreach (var elem in _resources)
                elem.SetIsMapped();
        }

        /// <summary>
        /// Maps all graphics resources for access by CUDA.<para/>
        /// The resources may be accessed by CUDA until they are unmapped. The graphics API from which the resource
        /// was registered should not access any resources while they are mapped by CUDA. If an application does
        /// so, the results are undefined.<para/>
        /// This function provides the synchronization guarantee that any graphics calls issued before <see cref="MapAllResources()"/>
        /// will complete before any subsequent CUDA work issued in <c>stream</c> begins.<para/>
        /// If any of the resources is presently mapped for access by CUDA then <see cref="CUResult.ErrorAlreadyMapped"/> exception is thrown.
        /// </summary>
        public void UnmapAllResources()
        {
            UnmapAllResources(new CUstream());
        }

        /// <summary>
        /// Unmaps all graphics resources.<para/>
        /// Once unmapped, the resources may not be accessed by CUDA until they are mapped again.<para/>
        /// This function provides the synchronization guarantee that any CUDA work issued in <c>stream</c> before <see cref="UnmapAllResources()"/>
        /// will complete before any subsequently issued graphics work begins.<para/>
        /// If any of the resources are not presently mapped for access by CUDA then <see cref="CUResult.ErrorNotMapped"/> exception is thrown.
        /// </summary>
        /// <param name="stream"></param>
        public void UnmapAllResources(CUstream stream)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
            CUResult res;
            res = DriverAPINativeMethods.GraphicsInterop.cuGraphicsUnmapResources((uint)_CUResources.Count, _CUResources.ToArray(), stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphicsUnmapResources", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            foreach (var elem in _resources)
                elem.SetIsUnmapped();
        }
        #endregion

    }
}
