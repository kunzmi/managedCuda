//	Copyright (c) 2015, Michael Kunz. All rights reserved.
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
    /// An opaque structure holding the
    /// description of a generic n-D dataset.
    /// </summary>
    public class SpatialTransformerDescriptor : IDisposable
    {
        private cudnnSpatialTransformerDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;
        private cudnnHandle _handle;

        #region Contructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="context"></param>
        public SpatialTransformerDescriptor(CudaDNNContext context)
        {
            _handle = context.Handle;
            _desc = new cudnnSpatialTransformerDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateSpatialTransformerDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateSpatialTransformerDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~SpatialTransformerDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroySpatialTransformerDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroySpatialTransformerDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnSpatialTransformerDescriptor Desc
        {
            get { return _desc; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="samplerType">Enumerant to specify the sampler type.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Dimension of the transformed tensor.</param>
        /// <param name="dimA">Array of dimension nbDims containing the size of the transformed tensor for every dimension.</param>
        public void SetSpatialTransformerNdDescriptor(
                                        cudnnSamplerType samplerType,
                                        cudnnDataType dataType,
                                        int nbDims,
                                        int[] dimA)
        {
            res = CudaDNNNativeMethods.cudnnSetSpatialTransformerNdDescriptor(_desc, samplerType, dataType, nbDims, dimA);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetSpatialTransformerNdDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
        /// </summary>
        /// <param name="theta">Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, where n is the number of images specified in stDesc.</param>
        /// <param name="grid">A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n, h, w is specified in stDesc. In the 4th dimension, the first coordinate is x, and the second coordinate is y.</param>
        public void SpatialTfGridGeneratorForward(
                                         CudaDeviceVariable<float> theta,
                                         CudaDeviceVariable<float> grid)
        {
            res = CudaDNNNativeMethods.cudnnSpatialTfGridGeneratorForward(_handle, _desc, theta.DevicePointer, grid.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSpatialTfGridGeneratorForward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function computes the gradient of a grid generation operation.
        /// </summary>
        /// <param name="dgrid">Data pointer to GPU memory contains the input differential data.</param>
        /// <param name="dtheta">Data pointer to GPU memory contains the output differential data.</param>
        /// <returns></returns>
        public void SpatialTfGridGeneratorBackward(
                                         CudaDeviceVariable<float> dgrid,
                                         CudaDeviceVariable<float> dtheta)
        {
            res = CudaDNNNativeMethods.cudnnSpatialTfGridGeneratorBackward(_handle, _desc, dtheta.DevicePointer, dgrid.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSpatialTfGridGeneratorBackward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function performs a sampler operation and generates the output tensor using the grid given by the grid generator.
        /// </summary>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="grid">A grid of coordinates generated by cudnnSpatialTfGridGeneratorForward.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        public void SpatialTfSamplerForward(
                                         float alpha,
                                         TensorDescriptor xDesc,
                                         CudaDeviceVariable<float> x,
                                         CudaDeviceVariable<float> grid,
                                         float beta,
                                         TensorDescriptor yDesc,
                                         CudaDeviceVariable<float> y)
        {
            res = CudaDNNNativeMethods.cudnnSpatialTfSamplerForward(_handle, _desc, ref alpha, xDesc.Desc, x.DevicePointer, grid.DevicePointer,
                ref beta, yDesc.Desc, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSpatialTfSamplerForward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);

        }

        /// <summary>
        /// This function performs a sampler operation and generates the output tensor using the grid given by the grid generator.
        /// </summary>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="grid">A grid of coordinates generated by cudnnSpatialTfGridGeneratorForward.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        public void SpatialTfSamplerForward(
                                            double alpha,
                                            TensorDescriptor xDesc,
                                            CudaDeviceVariable<double> x,
                                            CudaDeviceVariable<double> grid,
                                            double beta,
                                            TensorDescriptor yDesc,
                                            CudaDeviceVariable<double> y)
        {
            res = CudaDNNNativeMethods.cudnnSpatialTfSamplerForward(_handle, _desc, ref alpha, xDesc.Desc, x.DevicePointer, grid.DevicePointer,
                ref beta, yDesc.Desc, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSpatialTfSamplerForward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function computes the gradient of a sampling operation.
        /// </summary>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor dxDesc.</param>
        /// <param name="alphaDgrid">Pointer to scaling factor (in host memory) used to blend the gradient outputs dgrid with prior value in the destination pointer as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor dyDesc.</param>
        /// <param name="grid">A grid of coordinates generated by cudnnSpatialTfGridGeneratorForward.</param>
        /// <param name="betaDgrid">Pointer to scaling factor (in host memory) used to blend the gradient outputs dgrid with prior value in the destination pointer as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dgrid">Data pointer to GPU memory contains the output differential data.</param>
        public void cudnnSpatialTfSamplerBackward(
                                        float alpha,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<float> x,
                                        float beta,
                                        TensorDescriptor dxDesc,
                                        CudaDeviceVariable<float> dx,
                                        CudaDeviceVariable<float> alphaDgrid,
                                        TensorDescriptor dyDesc,
                                        CudaDeviceVariable<float> dy,
                                        CudaDeviceVariable<float> grid,
                                        float betaDgrid,
                                        CudaDeviceVariable<float> dgrid)
        {
            res = CudaDNNNativeMethods.cudnnSpatialTfSamplerBackward(_handle, _desc, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer,
                alphaDgrid.DevicePointer, dyDesc.Desc, dy.DevicePointer, grid.DevicePointer, ref betaDgrid, dgrid.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSpatialTfSamplerBackward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function computes the gradient of a sampling operation.
        /// </summary>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor dxDesc.</param>
        /// <param name="alphaDgrid">Pointer to scaling factor (in host memory) used to blend the gradient outputs dgrid with prior value in the destination pointer as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor dyDesc.</param>
        /// <param name="grid">A grid of coordinates generated by cudnnSpatialTfGridGeneratorForward.</param>
        /// <param name="betaDgrid">Pointer to scaling factor (in host memory) used to blend the gradient outputs dgrid with prior value in the destination pointer as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dgrid">Data pointer to GPU memory contains the output differential data.</param>
        public void cudnnSpatialTfSamplerBackward(
                                        double alpha,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<double> x,
                                        double beta,
                                        TensorDescriptor dxDesc,
                                        CudaDeviceVariable<double> dx,
                                        CudaDeviceVariable<double> alphaDgrid,
                                        TensorDescriptor dyDesc,
                                        CudaDeviceVariable<double> dy,
                                        CudaDeviceVariable<double> grid,
                                        double betaDgrid,
                                        CudaDeviceVariable<double> dgrid)
        {
            res = CudaDNNNativeMethods.cudnnSpatialTfSamplerBackward(_handle, _desc, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer,
                alphaDgrid.DevicePointer, dyDesc.Desc, dy.DevicePointer, grid.DevicePointer, ref betaDgrid, dgrid.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSpatialTfSamplerBackward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }




    }
}
