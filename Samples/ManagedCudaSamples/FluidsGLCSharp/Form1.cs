/*
 * This software contains source code provided by NVIDIA Corporation.
 * FluidsGLCSharp is a C# port of fluidsGL from the CUDA SDK using 
 * the ManagedCUDA and OpenTK libraries.
 */

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.CudaFFT;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using cData = ManagedCuda.VectorTypes.float2;

namespace FluidsGLCSharp
{
    public partial class Form1 : Form
    {
        OpenTK.GLControl m_renderControl;
        const int DIM = 512;    // Square size of solver domain

        const int DS = (DIM*DIM);           // Total domain size
        const int CPADW = (DIM/2+1);        // Padded width for real->complex in-place FFT
        const int RPADW = (2*(DIM/2+1));    // Padded width for real->complex in-place FFT
        const int PDS = (DIM*CPADW);        // Padded total domain size

        const float DT = 0.09f;             // Delta T for interative solver
        const float VIS = 0.0025f;          // Viscosity constant
        const float FORCE = (5.8f*DIM);     // Force scale factor 
        const int FR = 4;                   // Force update radius

        const int TILEX = 64; // Tile width
        const int TILEY = 64; // Tile height
        const int TIDSX = 64; // Tids in X
        const int TIDSY = 4;  // Tids in Y

        // CUFFT plan handle
        CudaFFTPlan2D planr2c;
        CudaFFTPlan2D planc2r;

        CudaDeviceVariable<cData> vxfield = null;
        CudaDeviceVariable<cData> vyfield = null;

        cData[] hvfield = null;
        CudaPitchedDeviceVariable<cData> dvfield = null;
        int wWidth = Math.Max(512, DIM);
        int wHeight = Math.Max(512, DIM);

        bool clicked = false;
        int fpsCount = 0;
        int fpsLimit = 1;
        CudaStopWatch stopwatch;

        // Particle data
        uint vbo = 0;                 // OpenGL vertex buffer object
        CudaGraphicsInteropResourceCollection cuda_vbo_resource; // handles OpenGL-CUDA exchange
        CudaTextureArray2D texref;  //Using ManagedCUDA we must explicitly define the texture reference (as with the CUDA driver API)
         
        cData[] particles = null; // particle positions in host memory
        int lastx = 0, lasty = 0;

        // Texture pitch
        SizeT tPitch = 0; // we store the pitch here for more consistency to the original SDK code

        CudaContext ctx;
        CudaKernel addForces_k;
        CudaKernel advectVelocity_k; //uses the texture
        CudaKernel diffuseProject_k;
        CudaKernel updateVelocity_k;
        CudaKernel advectParticles_k;

        bool isRunning = false;
        bool isInit = false;


        public Form1()
        {
            Console.WriteLine("[" + System.AppDomain.CurrentDomain.FriendlyName.Replace(".exe","") + "] - [OpenGL/CUDA simulation] starting...");
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //init openGL and CUDA
            initGLAndCuda();
            Show();
            Application.DoEvents();

            isRunning = true;

            //Render loop
            while (isRunning)
            {
                display();
                // Yield the CPU
                System.Threading.Thread.Sleep(1);
                Application.DoEvents();
            }
        }

        private void initGLAndCuda()
        {
            //Create render target control
            m_renderControl = new OpenTK.GLControl(GraphicsMode.Default, 1, 0, GraphicsContextFlags.Default);
            m_renderControl.Dock = DockStyle.Fill;
            m_renderControl.BackColor = Color.White;
            m_renderControl.BorderStyle = BorderStyle.FixedSingle;
            m_renderControl.KeyDown += new KeyEventHandler(m_renderControl_KeyDown);
            m_renderControl.MouseMove += new MouseEventHandler(m_renderControl_MouseMove);
            m_renderControl.MouseDown += new MouseEventHandler(m_renderControl_MouseDown);
            m_renderControl.SizeChanged += new EventHandler(m_renderControl_SizeChanged);
            
            panel1.Controls.Add(m_renderControl);
            Console.WriteLine("   OpenGL device is Available");
            
            int deviceID = CudaContext.GetMaxGflopsDeviceId();

            ctx = CudaContext.CreateOpenGLContext(deviceID, CUCtxFlags.BlockingSync);
            string console = string.Format("CUDA device [{0}] has {1} Multi-Processors", ctx.GetDeviceName(), ctx.GetDeviceInfo().MultiProcessorCount);
            Console.WriteLine(console);

            CUmodule module = ctx.LoadModulePTX("kernel.ptx");

            addForces_k = new CudaKernel("addForces_k", module, ctx);
            advectVelocity_k = new CudaKernel("advectVelocity_k", module, ctx);
            diffuseProject_k = new CudaKernel("diffuseProject_k", module, ctx);
            updateVelocity_k = new CudaKernel("updateVelocity_k", module, ctx);
			advectParticles_k = new CudaKernel("advectParticles_OGL", module, ctx);
            
            hvfield = new cData[DS];
            dvfield = new CudaPitchedDeviceVariable<cData>(DIM, DIM);
            tPitch = dvfield.Pitch;

            dvfield.CopyToDevice(hvfield);

            vxfield = new CudaDeviceVariable<cData>(DS);
            vyfield = new CudaDeviceVariable<cData>(DS);

            // Create particle array
            particles = new cData[DS];
            initParticles(particles, DIM, DIM);

            // TODO: update kernels to use the new unpadded memory layout for perf
            // rather than the old FFTW-compatible layout
            planr2c = new CudaFFTPlan2D(DIM, DIM, cufftType.R2C, Compatibility.FFTWPadding);
            planc2r = new CudaFFTPlan2D(DIM, DIM, cufftType.C2R, Compatibility.FFTWPadding);

            GL.GenBuffers(1, out vbo);
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.BufferData<cData>(BufferTarget.ArrayBuffer, new IntPtr(cData.SizeOf * DS), particles, BufferUsageHint.DynamicDraw);
            int bsize;
            GL.GetBufferParameter(BufferTarget.ArrayBuffer, BufferParameterName.BufferSize, out bsize);

            if (bsize != DS * cData.SizeOf)
                throw new Exception("Sizes don't match.");

            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);

            cuda_vbo_resource = new CudaGraphicsInteropResourceCollection();
            cuda_vbo_resource.Add(new CudaOpenGLBufferInteropResource(vbo, CUGraphicsRegisterFlags.None));

            texref = new CudaTextureArray2D(advectVelocity_k, "texref", CUAddressMode.Wrap, CUFilterMode.Linear, 0, CUArrayFormat.Float, DIM, DIM, CudaArray2DNumChannels.Two);

            stopwatch = new CudaStopWatch(CUEventFlags.Default);

            reshape();
            isInit = true;
            display();            
        }

        void simulateFluids()
        {
           // simulate fluid
           advectVelocity(dvfield, vxfield, vyfield, DIM, RPADW, DIM, DT, tPitch);
		   diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS, tPitch);
		   updateVelocity(dvfield, vxfield, vyfield, DIM, RPADW, DIM, tPitch);
		   advectParticles(vbo, dvfield, DIM, DIM, DT, tPitch);
        }

        private void reshape()
        {
            GL.Viewport(0, 0, m_renderControl.Width, m_renderControl.Height);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();
            GL.Ortho(0, 1, 1, 0, 0, 1);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
        }
        
        private void display()
        {
            stopwatch.Start();
            simulateFluids();

            // render points from vertex buffer
            GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Color4(0, 1, 0, 0.5f);
            GL.PointSize(1);
            GL.Enable(EnableCap.PointSmooth);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.Disable(EnableCap.DepthTest);
            GL.Disable(EnableCap.CullFace);
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.VertexPointer(2, VertexPointerType.Float, 0, 0);
            GL.DrawArrays(BeginMode.Points, 0, DS);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
            GL.DisableClientState(ArrayCap.VertexArray);
            GL.DisableClientState(ArrayCap.TextureCoordArray);
            GL.Disable(EnableCap.Texture2D);

            // Finish timing before swap buffers to avoid refresh sync
            stopwatch.Stop();
            m_renderControl.SwapBuffers();

            fpsCount++;
            if (fpsCount == fpsLimit)
            {
				float ifps = 1.0f / (stopwatch.GetElapsedTime() / 1000.0f);
				string fps = string.Format(System.Globalization.CultureInfo.InvariantCulture, "Cuda/GL Stable Fluids ({0} x {1}): {2} fps", DIM, DIM, ifps);
				this.Text = fps;
				fpsCount = 0;
				fpsLimit = (int)Math.Max(ifps, 1.0f);
            }
        }

        private void initParticles(cData[] p, int dx, int dy)
        {
            Random rand = new Random();

            int i, j;
            for (i = 0; i < dy; i++)
            {
                for (j = 0; j < dx; j++)
                {
                    p[i * dx + j].x = (j + 0.5f + ((float)rand.NextDouble() - 0.5f)) / dx;
                    p[i * dx + j].y = (i + 0.5f + ((float)rand.NextDouble() - 0.5f)) / dy;
                }
            }
        }

        #region Cuda kernel call methods
        void addForces(CudaPitchedDeviceVariable<float2> v, int dx, int dy, int spx, int spy, float fx, float fy, int r, SizeT tPitch)
        {
            dim3 tids = new dim3((uint)(2 * r + 1), (uint)(2 * r + 1), 1);

            addForces_k.GridDimensions = new dim3(1);
            addForces_k.BlockDimensions = tids;
            addForces_k.Run(v.DevicePointer, dx, dy, spx, spy, fx, fy, r, tPitch);
        }

        void updateTexture(CudaPitchedDeviceVariable<cData> data, SizeT wib, SizeT h, SizeT pitch)
        {
            texref.Array.CopyFromDeviceToThis<float2>(data);
        }

        void advectVelocity(CudaPitchedDeviceVariable<cData> v, CudaDeviceVariable<cData> vx, CudaDeviceVariable<cData> vy, int dx, int pdx, int dy, float dt, SizeT tPitch)
        {
            dim3 grid = new dim3((uint)((dx / TILEX) + (!(dx % TILEX != 0) ? 0 : 1)), (uint)((dy / TILEY) + (!(dy % TILEY != 0) ? 0 : 1)), 1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            updateTexture(v, DIM * float2.SizeOf, DIM, tPitch);

            advectVelocity_k.GridDimensions = grid;
            advectVelocity_k.BlockDimensions = tids;
            advectVelocity_k.Run(v.DevicePointer, vx.DevicePointer, vy.DevicePointer, dx, pdx, dy, dt, TILEY / TIDSY);
        }

        void diffuseProject(CudaDeviceVariable<cData> vx, CudaDeviceVariable<cData> vy, int dx, int dy, float dt, float visc, SizeT tPitch)
        {
            // Forward FFT
            planr2c.Exec(vx.DevicePointer);
            planr2c.Exec(vy.DevicePointer);

            dim3 grid = new dim3((uint)((dx / TILEX) + (!(dx % TILEX != 0) ? 0 : 1)), (uint)((dy / TILEY) + (!(dy % TILEY != 0) ? 0 : 1)), 1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            diffuseProject_k.GridDimensions = grid;
            diffuseProject_k.BlockDimensions = tids;
            diffuseProject_k.Run(vx.DevicePointer, vy.DevicePointer, dx, dy, dt, visc, TILEY / TIDSY);

            // Inverse FFT
            planc2r.Exec(vx.DevicePointer);
            planc2r.Exec(vy.DevicePointer);
        }

        void updateVelocity(CudaPitchedDeviceVariable<cData> v, CudaDeviceVariable<cData> vx, CudaDeviceVariable<cData> vy, int dx, int pdx, int dy, SizeT tPitch)
        {
            dim3 grid = new dim3((uint)((dx / TILEX) + (!(dx % TILEX != 0) ? 0 : 1)), (uint)((dy / TILEY) + (!(dy % TILEY != 0) ? 0 : 1)), 1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            updateVelocity_k.GridDimensions = grid;
            updateVelocity_k.BlockDimensions = tids;
            updateVelocity_k.Run(v.DevicePointer, vx.DevicePointer, vy.DevicePointer, dx, pdx, dy, TILEY / TIDSY, tPitch);
        }

        void advectParticles(uint vbo, CudaPitchedDeviceVariable<cData> v, int dx, int dy, float dt, SizeT tPitch)
        {
            dim3 grid = new dim3((uint)((dx / TILEX) + (!(dx % TILEX != 0) ? 0 : 1)), (uint)((dy / TILEY) + (!(dy % TILEY != 0) ? 0 : 1)), 1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            cuda_vbo_resource.MapAllResources();            
            CUdeviceptr p = cuda_vbo_resource[0].GetMappedPointer();
            advectParticles_k.GridDimensions = grid;
            advectParticles_k.BlockDimensions = tids;
            advectParticles_k.Run(p, v.DevicePointer, dx, dy, dt, TILEY / TIDSY, tPitch);
			cuda_vbo_resource.UnmapAllResources();
        }
        #endregion

        #region Events
        //Clean up before closing
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            isRunning = false;
            isInit = false;
            cuda_vbo_resource.Dispose();
            texref.Dispose();
            dvfield.Dispose();
            vxfield.Dispose();
            vyfield.Dispose();

            planc2r.Dispose();
            planr2c.Dispose();

            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
            GL.DeleteBuffers(1, ref vbo);

            stopwatch.Dispose();
            ctx.Dispose();
        }

        //"click" in the original code
        void m_renderControl_MouseDown(object sender, MouseEventArgs e)
        {
            lastx = e.X; lasty = e.Y;
        }

        void m_renderControl_SizeChanged(object sender, EventArgs e)
        {
            if (isInit)
                reshape();
        }

        //"motion" in the original
        void m_renderControl_MouseMove(object sender, MouseEventArgs e)
        {
            clicked = e.Button == System.Windows.Forms.MouseButtons.Left;

            // Convert motion coordinates to domain
            float fx = (lastx / (float)m_renderControl.Width);
            float fy = (lasty / (float)m_renderControl.Height);
            int nx = (int)(fx * DIM);
            int ny = (int)(fy * DIM);
            
            if (clicked && nx < DIM - FR && nx > FR - 1 && ny < DIM - FR && ny > FR - 1)
            {
                int ddx = e.X - lastx;
                int ddy = e.Y - lasty;
                fx = ddx / (float)m_renderControl.Width;
                fy = ddy / (float)m_renderControl.Height;
                int spy = ny - FR;
                int spx = nx - FR;
                addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR, tPitch);
                lastx = e.X; lasty = e.Y;
            } 
        }

        void m_renderControl_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Escape)
            {
                Close();
            }

            if (e.KeyCode == Keys.R)
            {
                Array.Clear(hvfield, 0, DS);
                dvfield.CopyToDevice(hvfield);
                initParticles(particles, DIM, DIM);

                cuda_vbo_resource.Clear();

                GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
                GL.BufferData<cData>(BufferTarget.ArrayBuffer, new IntPtr(cData.SizeOf * DS), particles, BufferUsageHint.DynamicDraw);
                GL.BindBuffer(BufferTarget.ArrayBuffer, 0);

                cuda_vbo_resource.Add(new CudaOpenGLBufferInteropResource(vbo, CUGraphicsRegisterFlags.None));
            }
        }
        #endregion
    }
}
