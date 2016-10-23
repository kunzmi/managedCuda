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
	public class GraphContext : IDisposable
	{
		bool disposed;
		nvgraphStatus res;
		nvgraphContext _context;

        #region Constructor
        /// <summary>
        /// </summary>
        public GraphContext()
		{
			_context = new nvgraphContext();

			res = NVGraphNativeMathods.nvgraphCreate(ref _context);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphCreate", res));
			if (res != nvgraphStatus.Success) throw new NVGraphException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
		~GraphContext()
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
				res = NVGraphNativeMathods.nvgraphDestroy(_context);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvgraphDestroy", res));
                disposed = true;
            }
			if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA.NVGraph not-disposed warning: {0}", this.GetType()));
        }
        #endregion

		public GraphDescriptor CreateGraphDecriptor()
		{
			GraphDescriptor descr = new GraphDescriptor(_context);
			return descr;
		}

        public nvgraphContext Context
        {
            get { return _context; }
        }
	}
}
