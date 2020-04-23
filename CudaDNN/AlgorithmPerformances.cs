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
using System.Text;
using System.Linq;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaDNN
{
    /// <summary>
    /// 
    /// </summary>
    public class AlgorithmPerformances : IDisposable
    {

        private cudnnAlgorithmPerformance[] _perfs;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        public AlgorithmPerformances(int count)
        {
            _perfs = new cudnnAlgorithmPerformance[count];
            res = CudaDNNNativeMethods.cudnnCreateAlgorithmPerformance(_perfs, count);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateAlgorithmPerformance", res));
            if (res != cudnnStatus.Success)
                throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~AlgorithmPerformances()
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
                res = CudaDNNNativeMethods.cudnnDestroyAlgorithmPerformance(_perfs, _perfs.Length);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyAlgorithmPerformance", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handles.
        /// </summary>
        public cudnnAlgorithmPerformance[] Perfs
        {
            get { return _perfs; }
        }

        public int Count
        {
            get { return _perfs.Length; }
        }



        public void SetAlgorithmPerformance(int index, AlgorithmDescriptor algoDesc, cudnnStatus status, float time, SizeT memory)
        {
            res = CudaDNNNativeMethods.cudnnSetAlgorithmPerformance(_perfs[index], algoDesc.Desc, status, time, memory);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetAlgorithmPerformance", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        public void GetAlgorithmPerformance(int index, ref AlgorithmDescriptor algoDesc, ref cudnnStatus status, ref float time, ref SizeT memory)
        {
            cudnnAlgorithmDescriptor descTemp = new cudnnAlgorithmDescriptor();
            res = CudaDNNNativeMethods.cudnnGetAlgorithmPerformance(_perfs[index], ref descTemp, ref status, ref time, ref memory);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetAlgorithmPerformance", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            algoDesc = new AlgorithmDescriptor(descTemp);
        }


    }
}
