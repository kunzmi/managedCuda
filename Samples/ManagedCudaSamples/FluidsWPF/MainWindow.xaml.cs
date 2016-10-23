/*
 * This software contains source code provided by NVIDIA Corporation.
 * FluidsWPF is a C# port of fluidsD3D9 from the CUDA SDK using 
 * the ManagedCUDA and SlimDX libraries.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Interop;
using System.Windows.Threading;
using System.Threading;
using System.Runtime.InteropServices;
using SlimDX;
using SlimDX.Direct3D9;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.CudaFFT;

namespace FluidsWPF
{
	/// <summary>
	/// Interaktionslogik für MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		const int DIM = 512;                // Square size of solver domain
		const int DS = DIM * DIM;           // Total domain size
		const int CPADW = (DIM/2+1);        // Padded width for real->complex in-place FFT
		const int RPADW = (2*(DIM/2+1));    // Padded width for real->complex in-place FFT
		const int PDS = (DIM * CPADW);      // Padded total domain size

		const float DT = 0.09f;             // Delta T for interative solver
		const float VIS = 0.0025f;          // Viscosity constant
		const float FORCE = (5.8f*DIM);     // Force scale factor 
		const int FR = 4;                   // Force update radius

		const int TILEX = 64;               // Tile width
		const int TILEY = 64;               // Tile height
		const int TIDSX = 64;               // Tids in X
		const int TIDSY = 4;                // Tids in Y

        //We need to define the vertex struct again in order to use it with C#
		[StructLayout(LayoutKind.Sequential)]
		struct vertex
		{
			public float x;
			public float y;
			public float z;
			public uint c;
		}

        int g_iAdapter;
        bool bDeviceFound;
		DeviceEx device;
		Direct3DEx d3d;
		VertexBuffer g_pVB;
        Texture g_pTexture;

		CudaContext ctx;
		CudaKernel addForces_k;
		CudaKernel advectVelocity_k; //uses the texture
		CudaKernel diffuseProject_k;
		CudaKernel updateVelocity_k;
		CudaKernel advectParticles_k;
        CudaStopWatch stopwatch;

		float2[] g_hvfield;
		CudaPitchedDeviceVariable<float2> g_dvfield;
        int wWidth = Math.Max(512, DIM);
        int wHeight = Math.Max(512, DIM);

        bool clicked = false;
        int fpsCount = 0;
        int fpsLimit = 1;

        // Particle data in host and device memory
        CudaDeviceVariable<vertex> g_mparticles;
		float2[] g_particles;

        int lastx = 0, lasty = 0;

        //Texture pitch
        SizeT g_tPitch = 0;
        
		CudaTextureArray2D texref;
        CudaGraphicsInteropResourceCollection graphicsres;

        // CUFFT plan handle
        CudaFFTPlan2D g_planr2c;
        CudaFFTPlan2D g_planc2r;
		CudaDeviceVariable<float2> g_vxfield;
		CudaDeviceVariable<float2> g_vyfield;

        DispatcherTimer frameTimer;
        double realElapsed;
        double realTotalElapsed;
        int realLastTick;
        int realCurrentTick;
        int realFrameCount;
        double realFrameCountTime;
        double realFrameRate;

		public MainWindow()
        {
			InitializeComponent();
            
            //CudaGraphicsInteropResourceCollection is just a collection class and does not call
            //any CUDA API function in it's constructor. We can instantiate here before CUDA API init.
            graphicsres = new CudaGraphicsInteropResourceCollection();

            frameTimer = new DispatcherTimer();
            frameTimer.Tick += new EventHandler(frameTimer_Tick);
            frameTimer.Interval = TimeSpan.FromMilliseconds(1);
		}

        private void frameTimer_Tick(object sender, EventArgs e)
        {
            realCurrentTick = Environment.TickCount;
            realElapsed = (double)(realCurrentTick - realLastTick) / 1000.0;
            realTotalElapsed += realElapsed;
            realLastTick = realCurrentTick;
            realFrameCount++;

            d3dimage.Lock();
            display();
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, d3dimage.PixelWidth, d3dimage.PixelHeight));
            d3dimage.Unlock();

            realFrameCountTime += realElapsed;

            if (realFrameCountTime >= 1.0)
            {
                realFrameRate = realFrameCount;
                realFrameCount = 0;
                realFPS.Text = "Real FPS: " + (realFrameRate / realFrameCountTime).ToString("0.00");
                realFrameCountTime = 0;
            }

        }

        #region Init
        private int findGraphicsGPU(out string devName)
        {
            int nGraphicsGPU = 0;
            int deviceCount = 0;
            bool bFoundGraphics = false;
            string firstGraphicsName = string.Empty, temp;
            devName = string.Empty;

            deviceCount = CudaContext.GetDeviceCount();

            // This function call returns 0 if there are no CUDA capable devices.
            if (deviceCount == 0)
            {
                infoLog.AppendText("There are no device(s) supporting CUDA\n");
                return 0;
            }
            else
            {
                infoLog.AppendText("> Found " + deviceCount + " CUDA Capable Device(s)\n");
            }

            for (int dev = 0; dev < deviceCount; dev++)
            {
                temp = CudaContext.GetDeviceName(dev);
                bool bGraphics = !temp.Contains("Tesla");
                StringBuilder sb = new StringBuilder();
                sb.Append("> ");
                if (bGraphics)
                    sb.Append("Graphics");
                else
                    sb.Append("Compute");

                sb.Append("\t\tGPU ").Append(dev).Append(": ").Append(CudaContext.GetDeviceName(dev)).Append("\n");
                infoLog.AppendText(sb.ToString());

		        if (bGraphics) 
                {
			        if (!bFoundGraphics) 
                    {
				        firstGraphicsName = temp;
			        }
        			nGraphicsGPU++;
		        }
            }

            if (nGraphicsGPU != 0)
            {
                devName = firstGraphicsName;
            }
            else
            {
                devName = "this hardware";
            }
            return nGraphicsGPU;
        }

        private bool InitializeD3D()
        {
            HwndSource hwnd = new HwndSource(0, 0, 0, 0, 0, "null", IntPtr.Zero);
            // Create the D3D object.
			d3d = new Direct3DEx();


			PresentParameters pp = new PresentParameters();
            pp.BackBufferWidth = 512;
            pp.BackBufferHeight = 512;
			pp.BackBufferFormat = Format.Unknown;
            pp.BackBufferCount = 0;
            pp.Multisample = MultisampleType.None;
            pp.MultisampleQuality = 0;
			pp.SwapEffect = SwapEffect.Discard;
            pp.DeviceWindowHandle = (IntPtr)0;
			pp.Windowed = true;
            pp.EnableAutoDepthStencil = false;
            pp.AutoDepthStencilFormat = Format.Unknown;
            pp.PresentationInterval = PresentInterval.Default;

            bDeviceFound = false;
            CUdevice[] cudaDevices = null;
            for (g_iAdapter = 0; g_iAdapter < d3d.AdapterCount; g_iAdapter++)
            {
                device = new DeviceEx(d3d, d3d.Adapters[g_iAdapter].Adapter, DeviceType.Hardware, hwnd.Handle, CreateFlags.HardwareVertexProcessing | CreateFlags.Multithreaded, pp);
                try 
	            {	        
		            cudaDevices = CudaContext.GetDirectXDevices(device.ComPointer, CUd3dXDeviceList.All, CudaContext.DirectXVersion.D3D9);
                    bDeviceFound = cudaDevices.Length > 0;
                    infoLog.AppendText("> Display Device #" + d3d.Adapters[g_iAdapter].Adapter
                        + ": \"" + d3d.Adapters[g_iAdapter].Details.Description + "\" supports Direct3D9 and CUDA.\n");
                    break;
	            }
	            catch (CudaException)
	            {
                    //No Cuda device found for this Direct3D9 device
                    infoLog.AppendText("> Display Device #" + d3d.Adapters[g_iAdapter].Adapter
                        + ": \"" + d3d.Adapters[g_iAdapter].Details.Description + "\" supports Direct3D9 but not CUDA.\n");
	            }
            }

            // we check to make sure we have found a cuda-compatible D3D device to work on  
            if (!bDeviceFound)
            {
                infoLog.AppendText("No CUDA-compatible Direct3D9 device available");
                if (device != null)
                    device.Dispose();
                return false;
            }

			ctx = new CudaContext(cudaDevices[0], device.ComPointer, CUCtxFlags.BlockingSync, CudaContext.DirectXVersion.D3D9);
            deviceName.Text = "Device name: " + ctx.GetDeviceName();

			// Set projection matrix
			SlimDX.Matrix matProj = SlimDX.Matrix.OrthoOffCenterLH(0, 1, 1, 0, 0, 1);
			device.SetTransform(TransformState.Projection, matProj);
           
			// Turn off D3D lighting, since we are providing our own vertex colors
			device.SetRenderState(RenderState.Lighting, false);

            //Load kernels
            CUmodule module = ctx.LoadModulePTX("kernel.ptx");

            addForces_k = new CudaKernel("addForces_k", module, ctx);
            advectVelocity_k = new CudaKernel("advectVelocity_k", module, ctx);
            diffuseProject_k = new CudaKernel("diffuseProject_k", module, ctx);
            updateVelocity_k = new CudaKernel("updateVelocity_k", module, ctx);
            advectParticles_k = new CudaKernel("advectParticles_k", module, ctx);

            d3dimage.Lock();
            Surface surf = device.GetBackBuffer(0, 0);
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, surf.ComPointer);
            d3dimage.Unlock();
            surf.Dispose();

            //Setup the "real" frame rate counter.
            //The cuda counter only measures cuda runtime, not the overhead to actually
            //show the result via DirectX and WPF.
            realLastTick = Environment.TickCount;
            return true;
		}

        private void InitializeCUFFT()
        {
            g_hvfield = new float2[DS];

            g_dvfield = new CudaPitchedDeviceVariable<float2>(DIM, DIM);
            g_tPitch = g_dvfield.Pitch; //Store pitch in g_tPitch to keep consistency to the C++ code
            g_dvfield.CopyToDevice(g_hvfield);

            // Temporary complex velocity field data   
            g_vxfield = new CudaDeviceVariable<float2>(PDS);
            g_vyfield = new CudaDeviceVariable<float2>(PDS);

            texref = new CudaTextureArray2D(advectVelocity_k, "texref", CUAddressMode.Wrap, CUFilterMode.Linear, 0, CUArrayFormat.Float, DIM, DIM, CudaArray2DNumChannels.Two);

            g_particles = new float2[DS];
            initParticles(g_particles, DIM, DIM);

            g_planr2c = new CudaFFTPlan2D(DIM, DIM, cufftType.R2C, Compatibility.FFTWPadding);
            g_planc2r = new CudaFFTPlan2D(DIM, DIM, cufftType.C2R, Compatibility.FFTWPadding);
        }

        private void initParticles(float2[] p, int dx, int dy)
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

        private void InitializeVB()
		{
			g_pVB = new VertexBuffer(device, DS * Marshal.SizeOf(typeof(vertex)), Usage.None, VertexFormat.Position | VertexFormat.Diffuse, Pool.Default);

            updateVB();

            graphicsres.Add(new CudaDirectXInteropResource(g_pVB.ComPointer, CUGraphicsRegisterFlags.None, CudaContext.DirectXVersion.D3D9));

		}

        private void InitializePointTexture()
        {
            int width = 64;
            int height = width;

            g_pTexture = new Texture(device, width, height, 0, Usage.AutoGenerateMipMap | Usage.Dynamic, Format.A8R8G8B8, Pool.Default);

            DataRectangle rect = g_pTexture.LockRectangle(0, LockFlags.None);
            
            rect.Data.Position = 0;
                
            for (int y = - height / 2; y < height / 2; ++y) 
            {
                float yf = y + 0.5f;
                int counter = 0;
                for (int x = - width / 2; x < width / 2; ++x) 
                {
                    float xf = x + 0.5f;
                    float radius = (float)width / 32;
                    float dist = (float)Math.Sqrt(xf * xf + yf * yf) / radius;
                    float n = 0.1f;
                    float value;
                    if (dist < 1)
                        value = 1 - 0.5f * (float)Math.Pow(dist, n);
                    else if (dist < 2)
                        value = 0.5f * (float)Math.Pow(2 - dist, n);
                    else
                        value = 0;
                    value *= 75;
                    rect.Data.Write<byte>((byte)value);
                    rect.Data.Write<byte>((byte)value);
                    rect.Data.Write<byte>((byte)value);
                    rect.Data.Write<byte>((byte)value);
                    counter += 4;
                }
                rect.Data.Position += rect.Pitch - counter;
            }
            g_pTexture.UnlockRectangle(0);
            
            device.SetSamplerState(0, SamplerState.MinFilter, TextureFilter.Linear);
            device.SetSamplerState(0, SamplerState.MagFilter, TextureFilter.Linear);


        }
        #endregion

        private void updateVB() 
		{
			DataStream data = g_pVB.Lock(0, DS * Marshal.SizeOf(typeof(vertex)), LockFlags.None);
			data.Position = 0;
			
			for (int i = 0; i < DS; i++) {
				vertex v = new vertex();
				v.x = g_particles[i].x;
				v.y = g_particles[i].y;
				v.z = 0.0f;
				v.c = 0xff00ff00;

				data.Write<vertex>(v);
			}
			g_pVB.Unlock();
            data.Dispose();
		}

        private void display()
        {
            stopwatch.Start();

            advectVelocity(g_dvfield, g_vxfield, g_vyfield, DIM, RPADW, DIM, DT, g_tPitch);

            {
                // Forward FFT
                g_planr2c.Exec(g_vxfield.DevicePointer);
                g_planr2c.Exec(g_vyfield.DevicePointer);

                diffuseProject(g_vxfield, g_vyfield, CPADW, DIM, DT, VIS, g_tPitch);

                // Inverse FFT
                g_planc2r.Exec(g_vxfield.DevicePointer);
                g_planc2r.Exec(g_vyfield.DevicePointer);
            }
            updateVelocity(g_dvfield, g_vxfield, g_vyfield, DIM, RPADW, DIM, g_tPitch);
           
            // Map D3D9 vertex buffer to CUDA
            {
                graphicsres.MapAllResources();

                if (g_mparticles != null)
                    g_mparticles.Dispose();

                g_mparticles = graphicsres[0].GetMappedPointer<vertex>();
                advectParticles(g_mparticles, g_dvfield, DIM, DIM, DT, g_tPitch);
                graphicsres.UnmapAllResources();
            }

            device.Clear(ClearFlags.Target, new Color4(0.0f, 0, 0), 0.0f, 0);
            device.SetRenderState(RenderState.ZWriteEnable, false);
            device.SetRenderState(RenderState.AlphaBlendEnable, true);
            device.SetRenderState(RenderState.SourceBlend, Blend.One);
            device.SetRenderState(RenderState.DestinationBlend, Blend.One);
            device.SetRenderState(RenderState.PointSpriteEnable, true);
            float size = 16.0f;
            device.SetRenderState(RenderState.PointSize, size);
            device.SetTexture(0, g_pTexture);

            if (device.BeginScene().IsSuccess)
            {             
                Result res;
                //Draw particles
                res = device.SetStreamSource(0, g_pVB, 0, Marshal.SizeOf(typeof(vertex)));
                device.VertexFormat = VertexFormat.Position | VertexFormat.Diffuse;
                res = device.DrawPrimitives(PrimitiveType.PointList, 0, DS);
                device.EndScene();
            }
            stopwatch.Stop();

            device.Present();
            fpsCount++;

            if (fpsCount == fpsLimit)
            {
                float elaps = stopwatch.GetElapsedTime();
                float ifps = 1.0f / (elaps / 1000.0f);
                string fps = string.Format(System.Globalization.CultureInfo.InvariantCulture, 
                    "CUDA/D3D9 Stable Fluids ({0} x {1}): {2} fps", DIM, DIM, ifps);
                myWindow.Title = fps;
                fpsCount = 0;
                fpsLimit = (int)Math.Max(ifps, 1.0f);
            }
        }

        private void updateTexture(CudaPitchedDeviceVariable<float2> data, SizeT wib, SizeT h, SizeT pitch)
        {
            texref.Array.CopyFromDeviceToThis<float2>(data);
        }

        #region Kernel launch methods
        private void addForces(CudaPitchedDeviceVariable<float2> v, int dx, int dy, int spx, int spy, float fx, float fy, int r, SizeT tPitch) 
        { 
            dim3 tids = new dim3((uint)(2*r+1), (uint)(2*r+1), 1);
    
            addForces_k.GridDimensions = new dim3(1);
            addForces_k.BlockDimensions = tids;
            addForces_k.Run(v.DevicePointer, dx, dy, spx, spy, fx, fy, r, tPitch);
        }

        private void advectVelocity(CudaPitchedDeviceVariable<float2> v, CudaDeviceVariable<float2> vx, CudaDeviceVariable<float2> vy, int dx, int pdx, int dy, float dt, SizeT tPitch) 
        { 
            dim3 grid = new dim3((uint)((dx/TILEX)+(!(dx%TILEX!=0)?0:1)), (uint)((dy/TILEY)+(!(dy%TILEY!=0)?0:1)), 1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            updateTexture(v, DIM*float2.SizeOf, DIM, tPitch);

            advectVelocity_k.GridDimensions = grid;
            advectVelocity_k.BlockDimensions = tids;
            advectVelocity_k.Run(v.DevicePointer, vx.DevicePointer, vy.DevicePointer, dx, pdx, dy, dt, TILEY/TIDSY);
        }

        private void diffuseProject(CudaDeviceVariable<float2> vx, CudaDeviceVariable<float2> vy, int dx, int dy, float dt, float visc, SizeT tPitch)
        {
            dim3 grid = new dim3((uint)((dx / TILEX) + (!(dx % TILEX != 0) ? 0 : 1)), (uint)((dy / TILEY) + (!(dy % TILEY != 0) ? 0 : 1)), 1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            diffuseProject_k.GridDimensions = grid;
            diffuseProject_k.BlockDimensions = tids;
            diffuseProject_k.Run(vx.DevicePointer, vy.DevicePointer, dx, dy, dt, visc, TILEY / TIDSY);   
        }

        private void updateVelocity(CudaPitchedDeviceVariable<float2> v, CudaDeviceVariable<float2> vx, CudaDeviceVariable<float2> vy, int dx, int pdx, int dy, SizeT tPitch) 
        { 
            dim3 grid = new dim3((uint)((dx/TILEX)+(!(dx%TILEX!=0)?0:1)), (uint)((dy/TILEY)+(!(dy%TILEY!=0)?0:1)),1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            updateVelocity_k.GridDimensions = grid;
            updateVelocity_k.BlockDimensions = tids;
            updateVelocity_k.Run(v.DevicePointer, vx.DevicePointer, vy.DevicePointer, dx, pdx, dy, TILEY/TIDSY, tPitch);
        }

        private void advectParticles(CudaDeviceVariable<vertex> p, CudaPitchedDeviceVariable<float2> v, int dx, int dy, float dt, SizeT tPitch) 
        {
            dim3 grid = new dim3((uint)((dx/TILEX)+(!(dx%TILEX!=0)?0:1)), (uint)((dy/TILEY)+(!(dy%TILEY!=0)?0:1)),1);

            dim3 tids = new dim3(TIDSX, TIDSY, 1);

            advectParticles_k.GridDimensions = grid;
            advectParticles_k.BlockDimensions = tids;
            advectParticles_k.Run(p.DevicePointer, v.DevicePointer, dx, dy, dt, TILEY/TIDSY, tPitch);
        }
        #endregion

        #region Events
        private void myWindow_Loaded(object sender, RoutedEventArgs e)
        {
            infoLog.SetValue(Paragraph.LineHeightProperty, 1.0);
            string devName;
            if (findGraphicsGPU(out devName) == 0)
            {
                infoLog.AppendText(devName + " is not supported on " + System.AppDomain.CurrentDomain.FriendlyName + ".");
                return;
            }

            if (!InitializeD3D())
                return;
            InitializeCUFFT();
            InitializeVB();
            InitializePointTexture();
            stopwatch = new CudaStopWatch(CUEventFlags.BlockingSync);

            frameTimer.Start();

            //Make sure we start with a content size of 512x512 but can still resize the window after start up
            ClearValue(SizeToContentProperty);
            infoLog.ClearValue(WidthProperty);
            grid.ClearValue(WidthProperty); 
            grid.ClearValue(HeightProperty);
        }

        private void myWindow_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                Close();
            }

            if (e.Key == Key.S) //stop
            {
                frameTimer.Tick -= new EventHandler(frameTimer_Tick);
                frameTimer.Stop();
            }

            if (e.Key == Key.P) //play
            {
                frameTimer.Tick += new EventHandler(frameTimer_Tick);
                frameTimer.Start();
            }

            if (e.Key == Key.R)
            {
                Array.Clear(g_hvfield, 0, DS);
                g_dvfield.CopyToDevice(g_hvfield);
                initParticles(g_particles, DIM, DIM);
                updateVB();
            }
        }

        private void myWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            //Stop render loop before closing
            if (frameTimer != null)
            {
                frameTimer.Tick -= new EventHandler(frameTimer_Tick);
                frameTimer.Stop();
            }
            
            //Cleanup
            if (graphicsres != null) graphicsres.Dispose();
            if (g_mparticles != null) g_mparticles.Dispose();
            if (stopwatch != null) stopwatch.Dispose();

            if (texref != null) texref.Dispose();
            if (g_dvfield != null) g_dvfield.Dispose();
            if (g_vxfield != null) g_vxfield.Dispose();
            if (g_vyfield != null) g_vyfield.Dispose();

            if (g_planc2r != null) g_planc2r.Dispose();
            if (g_planr2c != null) g_planr2c.Dispose();

            if (g_pVB != null) g_pVB.Dispose();
            if (g_pTexture != null) g_pTexture.Dispose();

            if (device != null) device.Dispose();
            if (d3d != null) d3d.Dispose();

            if (ctx != null) ctx.Dispose();

        }
        
        private void output_MouseMove(object sender, MouseEventArgs e)
        {
            clicked = e.LeftButton == MouseButtonState.Pressed;

            Point p = e.GetPosition(output);
            int x = (int)p.X;
            int y = (int)p.Y;

            // Convert motion coordinates to domain
            float fx = (x / (float)output.ActualWidth/*wWidth*/);
            float fy = (y / (float)output.ActualHeight/*wHeight*/);
            int nx = (int)(fx * DIM);
            int ny = (int)(fy * DIM);

            if (clicked && nx < DIM - FR && nx > FR - 1 && ny < DIM - FR && ny > FR - 1)
            {
                int ddx = x - lastx;
                int ddy = y - lasty;

                fx = ddx / (float)output.ActualWidth/*wWidth*/;
                fy = ddy / (float)output.ActualHeight/*wHeight*/;
                int spy = ny - FR;
                int spx = nx - FR;
                addForces(g_dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR, g_tPitch);
                lastx = x; lasty = y;
            } 
        }
        #endregion
	}
}
