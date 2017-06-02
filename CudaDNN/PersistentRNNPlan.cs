//	Copyright (c) 2017, Michael Kunz. All rights reserved.
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
using System.Text;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaDNN
{
    /// <summary>
    /// PersistentRNNPlan is a pointer to an opaque structure holding a plan to
    /// execute a dynamic persistent RNN. cudnnCreatePersistentRNNPlan() is used to
    /// create and initialize one instance.
    /// </summary>
    public class PersistentRNNPlan : IDisposable
    {
        private cudnnPersistentRNNPlan _desc;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        public PersistentRNNPlan(RNNDescriptor rnnDesc, int minibatch, cudnnDataType dataType)
        {
            _desc = new cudnnPersistentRNNPlan();
            res = CudaDNNNativeMethods.cudnnCreatePersistentRNNPlan(rnnDesc.Desc, minibatch, dataType, ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreatePersistentRNNPlan", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~PersistentRNNPlan()
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
                //Ignore if failing
                res = CudaDNNNativeMethods.cudnnDestroyPersistentRNNPlan(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyPersistentRNNPlan", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnPersistentRNNPlan Plan
        {
            get { return _desc; }
        }

    }
}
