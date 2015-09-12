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
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NPP
{
	/// <summary>
	/// Graphcut of a flow network (32bit signed integer edge capacities). 4 neighborhood labeling.
	/// </summary>
	[Obsolete("Graphcut will be deprecated in a future release.")]
	public class GraphCut4 : IDisposable
	{
		NppiGraphcutState _state;
		NppStatus status;
		NppiSize _size;
		CudaDeviceVariable<byte> _buffer;
		bool disposed;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="size">Graph size</param>
		public GraphCut4(NppiSize size)
		{
			_size = size;
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutGetSize(_size, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutGetSize", status));
			NPPException.CheckNppStatus(status, this);

			_buffer = new CudaDeviceVariable<byte>(bufferSize);

			_state = new NppiGraphcutState();
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutInitAlloc(_size, ref _state, _buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutInitAlloc", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~GraphCut4()
		{
			Dispose(false);
		}
		#endregion
		
		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public virtual void Dispose()
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
				status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutFree(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutFree", status));
				
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("NPP not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Graphcut of a flow network (32bit signed integer edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 4-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(NPPImage_32sC1 Terminals, NPPImage_32sC1 LeftTransposed, NPPImage_32sC1 RightTransposed, 
			NPPImage_32sC1 Top, NPPImage_32sC1 Bottom, NPPImage_8uC1 Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut_32s8u(Terminals.DevicePointer, LeftTransposed.DevicePointer, 
				RightTransposed.DevicePointer, Top.DevicePointer, Bottom.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, 
				Label.DevicePointer, Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut_32s8u", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Graphcut of a flow network (32bit signed integer edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 4-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(CudaPitchedDeviceVariable<int> Terminals, CudaPitchedDeviceVariable<int> LeftTransposed, CudaPitchedDeviceVariable<int> RightTransposed,
			CudaPitchedDeviceVariable<int> Top, CudaPitchedDeviceVariable<int> Bottom, CudaPitchedDeviceVariable<byte> Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut_32s8u(Terminals.DevicePointer, LeftTransposed.DevicePointer, 
				RightTransposed.DevicePointer, Top.DevicePointer, Bottom.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, 
				Label.DevicePointer, Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut_32s8u", status));
			NPPException.CheckNppStatus(status, this);
		}
	}

	/// <summary>
	/// Graphcut of a flow network (32bit signed integer edge capacities). 8 neighborhood labeling.
	/// </summary>
	[Obsolete("Graphcut will be deprecated in a future release.")]
	public class GraphCut8 : IDisposable
	{
		NppiGraphcutState _state;
		NppStatus status;
		NppiSize _size;
		CudaDeviceVariable<byte> _buffer;
		bool disposed;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="size">Graph size</param>
		public GraphCut8(NppiSize size)
		{
			_size = size;
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8GetSize(_size, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8GetSize", status));
			NPPException.CheckNppStatus(status, this);

			_buffer = new CudaDeviceVariable<byte>(bufferSize);

			_state = new NppiGraphcutState();
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8InitAlloc(_size, ref _state, _buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8InitAlloc", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~GraphCut8()
		{
			Dispose(false);
		}
		#endregion

		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public virtual void Dispose()
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
				status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutFree(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutFree", status));

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("NPP not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Graphcut of a flow network (32bit signed integer edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 8-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="TopLeft">Pointer to top left edge capacities (topleft(*,0) </param>
		/// <param name="TopRight">Pointer to top right edge capacities (topright(*,0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="BottomLeft">Pointer to bottom left edge capacities </param>
		/// <param name="BottomRight">Pointer to bottom right edge capacities </param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(NPPImage_32sC1 Terminals, NPPImage_32sC1 LeftTransposed, NPPImage_32sC1 RightTransposed,
			NPPImage_32sC1 Top, NPPImage_32sC1 TopLeft, NPPImage_32sC1 TopRight, NPPImage_32sC1 Bottom, NPPImage_32sC1 BottomLeft,
			NPPImage_32sC1 BottomRight, NPPImage_8uC1 Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8_32s8u(Terminals.DevicePointer, LeftTransposed.DevicePointer, 
				RightTransposed.DevicePointer, Top.DevicePointer, TopLeft.DevicePointer, TopRight.DevicePointer, Bottom.DevicePointer, 
				BottomLeft.DevicePointer, BottomRight.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, Label.DevicePointer, 
				Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8_32s8u", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Graphcut of a flow network (32bit signed integer edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 8-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="TopLeft">Pointer to top left edge capacities (topleft(*,0) </param>
		/// <param name="TopRight">Pointer to top right edge capacities (topright(*,0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="BottomLeft">Pointer to bottom left edge capacities </param>
		/// <param name="BottomRight">Pointer to bottom right edge capacities </param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(CudaPitchedDeviceVariable<int> Terminals, CudaPitchedDeviceVariable<int> LeftTransposed, CudaPitchedDeviceVariable<int> RightTransposed,
			CudaPitchedDeviceVariable<int> Top, CudaPitchedDeviceVariable<int> TopLeft, CudaPitchedDeviceVariable<int> TopRight, CudaPitchedDeviceVariable<int> Bottom, CudaPitchedDeviceVariable<int> BottomLeft,
			CudaPitchedDeviceVariable<int> BottomRight, CudaPitchedDeviceVariable<byte> Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8_32s8u(Terminals.DevicePointer, LeftTransposed.DevicePointer,
				RightTransposed.DevicePointer, Top.DevicePointer, TopLeft.DevicePointer, TopRight.DevicePointer, Bottom.DevicePointer,
				BottomLeft.DevicePointer, BottomRight.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, Label.DevicePointer,
				Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8_32s8u", status));
			NPPException.CheckNppStatus(status, this);
		}
	}

	/// <summary>
	/// Graphcut of a flow network (32bit floating point edge capacities). 4 neighborhood labeling.
	/// </summary>
	[Obsolete("Graphcut will be deprecated in a future release.")]
	public class GraphCut4f : IDisposable
	{
		NppiGraphcutState _state;
		NppStatus status;
		NppiSize _size;
		CudaDeviceVariable<byte> _buffer;
		bool disposed;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="size">Graph size</param>
		public GraphCut4f(NppiSize size)
		{
			_size = size;
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutGetSize(_size, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutGetSize", status));
			NPPException.CheckNppStatus(status, this);

			_buffer = new CudaDeviceVariable<byte>(bufferSize);

			_state = new NppiGraphcutState();
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutInitAlloc(_size, ref _state, _buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutInitAlloc", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~GraphCut4f()
		{
			Dispose(false);
		}
		#endregion
		
		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public virtual void Dispose()
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
				status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutFree(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutFree", status));
				
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("NPP not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Graphcut of a flow network (32bit floating point edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 4-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(NPPImage_32fC1 Terminals, NPPImage_32fC1 LeftTransposed, NPPImage_32fC1 RightTransposed, 
			NPPImage_32fC1 Top, NPPImage_32fC1 Bottom, NPPImage_8uC1 Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut_32f8u(Terminals.DevicePointer, LeftTransposed.DevicePointer, 
				RightTransposed.DevicePointer, Top.DevicePointer, Bottom.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, 
				Label.DevicePointer, Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut_32f8u", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Graphcut of a flow network (32bit floating point edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 4-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(CudaPitchedDeviceVariable<float> Terminals, CudaPitchedDeviceVariable<float> LeftTransposed, CudaPitchedDeviceVariable<float> RightTransposed,
			CudaPitchedDeviceVariable<float> Top, CudaPitchedDeviceVariable<float> Bottom, CudaPitchedDeviceVariable<byte> Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut_32f8u(Terminals.DevicePointer, LeftTransposed.DevicePointer, 
				RightTransposed.DevicePointer, Top.DevicePointer, Bottom.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, 
				Label.DevicePointer, Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut_32f8u", status));
			NPPException.CheckNppStatus(status, this);
		}
	}

	/// <summary>
	/// Graphcut of a flow network (32bit floating point edge capacities). 8 neighborhood labeling.
	/// </summary>
	[Obsolete("Graphcut will be deprecated in a future release.")]
	public class GraphCut8f : IDisposable
	{
		NppiGraphcutState _state;
		NppStatus status;
		NppiSize _size;
		CudaDeviceVariable<byte> _buffer;
		bool disposed;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="size">Graph size</param>
		public GraphCut8f(NppiSize size)
		{
			_size = size;
			int bufferSize = 0;
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8GetSize(_size, ref bufferSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8GetSize", status));
			NPPException.CheckNppStatus(status, this);

			_buffer = new CudaDeviceVariable<byte>(bufferSize);

			_state = new NppiGraphcutState();
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8InitAlloc(_size, ref _state, _buffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8InitAlloc", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~GraphCut8f()
		{
			Dispose(false);
		}
		#endregion

		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public virtual void Dispose()
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
				status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcutFree(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcutFree", status));

				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("NPP not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Graphcut of a flow network (32bit floating point edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 8-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="TopLeft">Pointer to top left edge capacities (topleft(*,0) </param>
		/// <param name="TopRight">Pointer to top right edge capacities (topright(*,0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="BottomLeft">Pointer to bottom left edge capacities </param>
		/// <param name="BottomRight">Pointer to bottom right edge capacities </param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(NPPImage_32fC1 Terminals, NPPImage_32fC1 LeftTransposed, NPPImage_32fC1 RightTransposed,
			NPPImage_32fC1 Top, NPPImage_32fC1 TopLeft, NPPImage_32fC1 TopRight, NPPImage_32fC1 Bottom, NPPImage_32fC1 BottomLeft,
			NPPImage_32fC1 BottomRight, NPPImage_8uC1 Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8_32f8u(Terminals.DevicePointer, LeftTransposed.DevicePointer, 
				RightTransposed.DevicePointer, Top.DevicePointer, TopLeft.DevicePointer, TopRight.DevicePointer, Bottom.DevicePointer, 
				BottomLeft.DevicePointer, BottomRight.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, Label.DevicePointer, 
				Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8_32f8u", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Graphcut of a flow network (32bit floating point edge capacities). The
		/// function computes the minimal cut (graphcut) of a 2D regular 8-connected
		/// graph. <para/>
		/// The inputs are the capacities of the horizontal (in transposed form),
		/// vertical and terminal (source and sink) edges. The capacities to source and
		/// sink 
		/// are stored as capacity differences in the terminals array 
		/// ( terminals(x) = source(x) - sink(x) ). The implementation assumes that the
		/// edge capacities 
		/// for boundary edges that would connect to nodes outside the specified domain
		/// are set to 0 (for example left(0,*) == 0). If this is not fulfilled the
		/// computed labeling may be wrong!<para/>
		/// The computed binary labeling is encoded as unsigned 8bit values (0 and >0).
		/// </summary>
		/// <param name="Terminals">Pointer to differences of terminal edge capacities</param>
		/// <param name="LeftTransposed">Pointer to transposed left edge capacities</param>
		/// <param name="RightTransposed">Pointer to transposed right edge capacities</param>
		/// <param name="Top">Pointer to top edge capacities (top(*,0) must be 0)</param>
		/// <param name="TopLeft">Pointer to top left edge capacities (topleft(*,0) </param>
		/// <param name="TopRight">Pointer to top right edge capacities (topright(*,0)</param>
		/// <param name="Bottom">Pointer to bottom edge capacities (bottom(*,height-1)</param>
		/// <param name="BottomLeft">Pointer to bottom left edge capacities </param>
		/// <param name="BottomRight">Pointer to bottom right edge capacities </param>
		/// <param name="Label">Pointer to destination label image </param>
		/// <returns></returns>
		public void GraphCut(CudaPitchedDeviceVariable<float> Terminals, CudaPitchedDeviceVariable<float> LeftTransposed, CudaPitchedDeviceVariable<float> RightTransposed,
			CudaPitchedDeviceVariable<float> Top, CudaPitchedDeviceVariable<float> TopLeft, CudaPitchedDeviceVariable<float> TopRight, CudaPitchedDeviceVariable<float> Bottom, CudaPitchedDeviceVariable<float> BottomLeft,
			CudaPitchedDeviceVariable<float> BottomRight, CudaPitchedDeviceVariable<byte> Label)
		{
			status = NPPNativeMethods.NPPi.ImageLabeling.nppiGraphcut8_32f8u(Terminals.DevicePointer, LeftTransposed.DevicePointer,
				RightTransposed.DevicePointer, Top.DevicePointer, TopLeft.DevicePointer, TopRight.DevicePointer, Bottom.DevicePointer,
				BottomLeft.DevicePointer, BottomRight.DevicePointer, Terminals.Pitch, LeftTransposed.Pitch, _size, Label.DevicePointer,
				Label.Pitch, _state);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiGraphcut8_32f8u", status));
			NPPException.CheckNppStatus(status, this);
		}
	}

}
