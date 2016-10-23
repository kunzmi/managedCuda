/*
 * This code is based on code from the NVIDIA CUDA SDK. (Ported from C++ to C# using managedCUDA)
 * This software contains source code provided by NVIDIA Corporation.
 *
 */

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Windows.Forms;
using ManagedCuda;
using ManagedCuda.NPP;

namespace NPP_HistogramEqualization
{
	public partial class Form1 : Form
	{
		bool _nppOK = false;
		int _colorChannels = 0;
		NPPImage_8uC1 src_c1;
		NPPImage_8uC1 dest_c1;
		NPPImage_8uC3 src_c3;
		NPPImage_8uC3 dest_c3;
		NPPImage_8uC4 src_c4;
		NPPImage_8uC4 dest_c4;
		
		public Form1()
		{
			InitializeComponent();
		}

		private void Form1_Load(object sender, EventArgs e)
		{
			NppLibraryVersion libVer = NPPNativeMethods.NPPCore.nppGetLibVersion();

			GpuComputeCapability computeCap = NPPNativeMethods.NPPCore.nppGetGpuComputeCapability();

			string output = string.Format("NPP Library Version {0}\n", libVer.ToString());
			txt_info.AppendText(output);

			_nppOK = computeCap != GpuComputeCapability.CudaNotCapable;

			if (_nppOK)
			{
				output = string.Format("{0} using GPU <{1}> with {2} SM(s) with", Assembly.GetExecutingAssembly().GetName().Name,
					NPPNativeMethods.NPPCore.nppGetGpuName(), NPPNativeMethods.NPPCore.nppGetGpuNumSMs());
				txt_info.AppendText(output);
				if (computeCap > 0)
				{
					output = " Compute capability " + ((int)computeCap / 100).ToString() + "." + ((int)computeCap % 100).ToString() + "\n";
					//output = string.Format(" {0}\n", sComputeCap[(int)computeCap]);
					txt_info.AppendText(output);
				}
				else
				{
					txt_info.AppendText(" Unknown Compute Capabilities\n");
				}
			}
			else
			{
				output = string.Format(" {0}\n", "No CUDA Capable Device Found");
				txt_info.AppendText(output);
			}

			//The NPP Library distinguishes warnings and errors. Errors are wrapped to standard exceptions.
			//Warnings are handled using a singleton NPPWarningHandler instance where you can find the OnNPPWarning event:
			NPPWarningHandler.GetInstance().OnNPPWarning += new NPPWarningHandler.NPPWarningEventHandler(nppwarnings_OnNPPWarning);
		}

		void nppwarnings_OnNPPWarning(object sender, NPPWarningHandler.NPPWarningEventArgs e)
		{
			txt_info.AppendText("NPP Warning: " + e.Message + "\n");
		}

		private void btn_open_Click(object sender, EventArgs e)
		{
			if (!_nppOK) return; 

			CleanUp();
			
			OpenFileDialog ofd = new OpenFileDialog();
			ofd.Filter = "Images|*.jpg;*.bmp;*.png;*.tif";
			if (ofd.ShowDialog() != System.Windows.Forms.DialogResult.OK) return;

			Bitmap src = new Bitmap(ofd.FileName);

			switch (src.PixelFormat)
			{
				case PixelFormat.Format24bppRgb:
					_colorChannels = 3;
					break;
				case PixelFormat.Format32bppArgb:
					_colorChannels = 4;
					break;
				case PixelFormat.Format32bppRgb:
					_colorChannels = 4;
					break;
				case PixelFormat.Format8bppIndexed:
					_colorChannels = 1;
					break;
				default:
					_colorChannels = 0;
					txt_info.AppendText(ofd.FileName + " has an unsupported pixel format.\n");
					break;
			}

			try
			{
				switch (_colorChannels)
				{
					case 1:
						//Allocate memory on device for one channel images...
						src_c1 = new NPPImage_8uC1(src.Width, src.Height);
						dest_c1 = new NPPImage_8uC1(src.Width, src.Height);
						src_c1.CopyToDevice(src);
						txt_info.AppendText("Info: Loaded image '" + ofd.FileName + "' succesfully (Size: " + src.Width.ToString() + " x " + src.Height.ToString() + ", color channels: " + _colorChannels.ToString() + ")\n");
						break;
					case 3:
						//As of version 5, NPP has new histogram and LUT functions for three channel images, no more need to convert first to 4 channels.
						//Allocate memory on device for four channel images...
						src_c3 = new NPPImage_8uC3(src.Width, src.Height);
						dest_c3 = new NPPImage_8uC3(src.Width, src.Height);

                        //Fill 3 channel image in device memory
						src_c3.CopyToDevice(src);

						txt_info.AppendText("Info: Loaded image '" + ofd.FileName + "' succesfully (Size: " + src.Width.ToString() + " x " + src.Height.ToString() + ", color channels: " + _colorChannels.ToString() + ")\n");
						break;
					case 4:
						//Allocate memory on device for four channel images...
						src_c4 = new NPPImage_8uC4(src.Width, src.Height);
						dest_c4 = new NPPImage_8uC4(src.Width, src.Height);
						src_c4.CopyToDevice(src);
						txt_info.AppendText("Info: Loaded image '" + ofd.FileName + "' succesfully (Size: " + src.Width.ToString() + " x " + src.Height.ToString() + ", color channels: " + _colorChannels.ToString() + ")\n");
						break;
				}
			}
			catch (Exception ex)
			{
				if (ex is NPPException)
				{
					txt_info.AppendText("NPPException: " + ex.Message + "\n");
					CleanUp();
				}
				else if (ex is CudaException)
				{
					txt_info.AppendText("CudaException: " + ex.Message + "\n");
					CleanUp();
				}
				else throw;
			}
			//Show original image
			pictureBox_src.Image = src;
		}

		private void CleanUp()
		{
			pictureBox_src.Image = null;
			pictureBox_dest.Image = null;

			hist_rb_src.Image = null;
			hist_g_src.Image = null;
			hist_b_src.Image = null;
			hist_rb_dest.Image = null;
			hist_g_dest.Image = null;
			hist_b_dest.Image = null;

			lbl_max.Text = "0";

			_colorChannels = 0;
			if (src_c1 != null)
				src_c1.Dispose();
			if (src_c3 != null)
				src_c3.Dispose();
			if (src_c4 != null)
				src_c4.Dispose();
			if (dest_c1 != null)
				dest_c1.Dispose();
			if (dest_c4 != null)
				dest_c4.Dispose();
			if (dest_c3 != null)
				dest_c3.Dispose();
		}

		//Set palette for gray scale images
		private void SetPalette(Bitmap bmp)
		{
			ColorPalette pal = bmp.Palette;
			if (pal == null) return;
			if (pal.Entries.Length < 256) return;
			for (int i = 0; i < 256; i++)
			{
				pal.Entries[i] = Color.FromArgb(i, i, i);
			}
			bmp.Palette = pal;
		}

		//Save result image to file
		private void btn_Save_Click(object sender, EventArgs e)
		{
			if (pictureBox_dest.Image == null) return;

			SaveFileDialog sfd = new SaveFileDialog();
			sfd.Filter = "Bitmap-Image|*.bmp|TIFF-Image|*.tif|PNG-Image|*.png|JPEG-Image|*.jpg";
			if (sfd.ShowDialog() != System.Windows.Forms.DialogResult.OK) return;

			string ending = sfd.FileName.Substring(sfd.FileName.Length - 3);
			ImageFormat format;
			switch (ending)
			{
				case "bmp": format = ImageFormat.Bmp; break;
				case "tif": format = ImageFormat.Tiff; break;
				case "jpg": format = ImageFormat.Jpeg; break;
				default: format = ImageFormat.Png; break;
			}

			pictureBox_dest.Image.Save(sfd.FileName, format);
		}

		//Compute histogram and apply LUT to image
		private void btn_calc_Click(object sender, EventArgs e)
		{
			if (_colorChannels < 1 || !_nppOK) return;
			
			try
			{
				int binCount = 255;
				int levelCount = binCount + 1;

				int[] levels;
				int[] bins;
				int[] lut = new int[levelCount];
				int totalSum = 0;
				float mutiplier = 0;
				int runningSum = 0;
				Bitmap res;

				switch (_colorChannels)
				{
					case 1:
						//The NPP library sets up a CUDA context, we can directly use it without access to it
						CudaDeviceVariable<int> bins_d = new CudaDeviceVariable<int>(binCount);
						levels = src_c1.EvenLevels(levelCount, 0, levelCount);
						//Even levels in Cuda 5.5 seems to be broken: set it manually
						for (int i = 0; i < levelCount; i++)
						{
							levels[i] = i;
						}

						//Compute histogram from source image
						src_c1.HistogramEven(bins_d, 0, binCount+1);
						//Copy data from device to host:
						bins = bins_d;

						//draw histogram image
						hist_rb_src.Image = GetHistogramImage(bins, 0);

						//compute histogram equalization
						for (int i = 0; i < binCount; i++)
						{
							totalSum += bins[i];
						}
						Debug.Assert(totalSum == src_c1.Width * src_c1.Height);

						if (totalSum == 0) totalSum = 1;

						mutiplier = 1.0f / (float)totalSum * 255.0f;

						for (int i = 0; i < binCount; i++)
						{
							lut[i] = (int)(runningSum * mutiplier + 0.5f);
							runningSum += bins[i];
						}

						lut[binCount] = 255;

						//Aplly this lut to src image and get result in dest image
						src_c1.LUT(dest_c1, lut, levels);

						//Create new bitmap in host memory for result image
						res = new Bitmap(src_c1.Width, src_c1.Height, PixelFormat.Format8bppIndexed);
						SetPalette(res);

						//Copy result from device to host
						dest_c1.CopyToHost(res);

						pictureBox_dest.Image = res;

						//Compute new histogram and show it
						dest_c1.HistogramEven(bins_d, 0, binCount);
						hist_g_src.Image = GetHistogramImage(bins_d, 0);
						//Free temp memory
						bins_d.Dispose();
						break;
					case 3:
						//The NPP library sets up a CUDA context, we can directly use it without access to it
						CudaDeviceVariable<int>[] bins_ds = new CudaDeviceVariable<int>[3];
						bins_ds[0] = new CudaDeviceVariable<int>(binCount);
						bins_ds[1] = new CudaDeviceVariable<int>(binCount);
						bins_ds[2] = new CudaDeviceVariable<int>(binCount);
						levels = src_c3.EvenLevels(levelCount, 0, levelCount);
						//Even levels in Cuda 5.5 seems to be broken: set it manually
						for (int i = 0; i < levelCount; i++)
						{
							levels[i] = i;
						}
						int[] ll = new int[] { 0, 0, 0 };
						int[] up = new int[] { binCount+1, binCount+1, binCount+1 };

						//Compute histogram from source image
						src_c3.HistogramEven(bins_ds, ll, up);
						
						int[][] bins3 = new int[3][];
						int[][] luts = new int[3][];
						for (int c = 0; c < 3; c++)
						{
							//Copy data from device to host:
							bins3[c] = bins_ds[c];
							luts[c] = new int[levelCount];
						}

						//draw histogram images
						hist_rb_src.Image = GetHistogramImage(bins3[2], bins3[1], bins3[0], 1);
						hist_g_src.Image = GetHistogramImage(bins3[1], bins3[0], bins3[2], 2);
						hist_b_src.Image = GetHistogramImage(bins3[0], bins3[1], bins3[2], 3);

						//compute histogram equalization
						for (int c = 0; c < 3; c++)
						{
							totalSum = 0;
							runningSum = 0;
							for (int i = 0; i < binCount; i++)
							{
								totalSum += bins3[c][i];
							}
							Debug.Assert(totalSum == src_c3.Width * src_c3.Height);

							if (totalSum == 0) totalSum = 1;

							mutiplier = 1.0f / (float)totalSum * 255.0f;
						
							for (int i = 0; i < binCount; i++)
							{
								luts[c][i] = (int)(runningSum * mutiplier + 0.5f);
								runningSum += bins3[c][i];
							}
							luts[c][binCount] = 255;
						}
						//Aplly this lut to src image and get result in dest image
						src_c3.Lut(dest_c3, luts[0], levels, luts[1], levels, luts[2], levels);

						res = new Bitmap(src_c3.Width, src_c3.Height, PixelFormat.Format24bppRgb);

						//Copy result from device to host
						dest_c3.CopyToHost(res);

						pictureBox_dest.Image = res;

						//Compute new histogram and show it
						dest_c3.HistogramEven(bins_ds, ll, up);
						bins3[0] = bins_ds[0];
						bins3[1] = bins_ds[1];
						bins3[2] = bins_ds[2];
						hist_rb_dest.Image = GetHistogramImage(bins3[2], bins3[1], bins3[0], 1);//r
						hist_g_dest.Image = GetHistogramImage(bins3[1], bins3[0], bins3[2], 2);//g
						hist_b_dest.Image = GetHistogramImage(bins3[0], bins3[1], bins3[2], 3);//b

						//Free temp memory
						bins_ds[0].Dispose();
						bins_ds[1].Dispose();
						bins_ds[2].Dispose();
						break;
					case 4:
						//The NPP library sets up a CUDA context, we can directly use it without access to it
						CudaDeviceVariable<int>[] bins_ds4 = new CudaDeviceVariable<int>[4];
						bins_ds4[0] = new CudaDeviceVariable<int>(binCount);
						bins_ds4[1] = new CudaDeviceVariable<int>(binCount);
						bins_ds4[2] = new CudaDeviceVariable<int>(binCount);
						bins_ds4[3] = new CudaDeviceVariable<int>(binCount);
						levels = src_c4.EvenLevels(levelCount, 0, levelCount);
						//Even levels in Cuda 5.5 seems to be broken: set it manually
						for (int i = 0; i < levelCount; i++)
						{
							levels[i] = i;
						}
						int[] ll4 = new int[] { 0, 0, 0, 0 };
						int[] up4 = new int[] { binCount+1, binCount+1, binCount+1, binCount+1 };

						//Compute histogram from source image
						src_c4.HistogramEven(bins_ds4, ll4, up4);

						int[][] bins4 = new int[4][];
						int[][] luts4 = new int[4][];
						for (int c = 0; c < 4; c++)
						{
							//Copy data from device to host:
							bins4[c] = bins_ds4[c];
							luts4[c] = new int[levelCount];
						}

						//draw histogram images
						hist_rb_src.Image = GetHistogramImage(bins4[2], bins4[1], bins4[0], 1);
						hist_g_src.Image = GetHistogramImage(bins4[1], bins4[0], bins4[2], 2);
						hist_b_src.Image = GetHistogramImage(bins4[0], bins4[1], bins4[2], 3);

						//compute histogram equalization
						for (int c = 0; c < 3; c++)
						{
							totalSum = 0;
							runningSum = 0;
							for (int i = 0; i < binCount; i++)
							{
								totalSum += bins4[c][i];
							}
							Debug.Assert(totalSum == src_c4.Width * src_c4.Height);

							if (totalSum == 0) totalSum = 1;

							mutiplier = 1.0f / (float)totalSum * 255.0f;

							for (int i = 0; i < binCount; i++)
							{
								luts4[c][i] = (int)(runningSum * mutiplier + 0.5f);
								runningSum += bins4[c][i];
							}
							luts4[c][binCount] = 255;
						}

						//Aplly this lut to src image and get result in dest image
						src_c4.LutA(dest_c4, luts4[0], levels, luts4[1], levels, luts4[2], levels);

						//Set alpha channel to 255
						dest_c4.Set(255, 3);
						res = new Bitmap(src_c4.Width, src_c4.Height, PixelFormat.Format32bppArgb);

						//Copy result from device to host
						dest_c4.CopyToHost(res);

						pictureBox_dest.Image = res;

						//Compute new histogram and show it
						dest_c4.HistogramEven(bins_ds4, ll4, up4);
						bins4[0] = bins_ds4[0];
						bins4[1] = bins_ds4[1];
						bins4[2] = bins_ds4[2];
						hist_rb_dest.Image = GetHistogramImage(bins4[2], bins4[1], bins4[0], 1);//r
						hist_g_dest.Image = GetHistogramImage(bins4[1], bins4[0], bins4[2], 2);//g
						hist_b_dest.Image = GetHistogramImage(bins4[0], bins4[1], bins4[2], 3);//b

						//Free temp memory
						bins_ds4[0].Dispose();
						bins_ds4[1].Dispose();
						bins_ds4[2].Dispose();
						bins_ds4[3].Dispose();
						break;
				}
			}
			catch (Exception ex)
			{
				if (ex is NPPException)
				{
					txt_info.AppendText("NPPException: " + ex.Message + "\n");
					CleanUp();
				}
				else if (ex is CudaException)
				{
					txt_info.AppendText("CudaException: " + ex.Message + "\n");
					CleanUp();
				}
				else throw;
			}
		}

		//Draw histogram (one channel)
		private Bitmap GetHistogramImage(int[] histi, int channel)
		{
			Bitmap hist = new Bitmap(255, 255, PixelFormat.Format24bppRgb);
			Graphics gr = Graphics.FromImage(hist);

			gr.Clear(this.BackColor);
			int maxi = 0;
			for (int i = 0; i < 255; i++)
			{
				maxi = Math.Max(maxi, histi[i]);
			}
			float maxf = (float)maxi;
			Color c = Color.DarkGray;
			switch (channel)
			{
				case 0: c = Color.DarkGray; break;
				case 1: c = Color.Red; break;
				case 2: c = Color.Green; break;
				case 3: c = Color.Blue; break;
			}

			Pen pen = new Pen(c);

			for (int i = 0; i < 255; i++)
			{
				gr.DrawLine(pen, i, 255, i, 254 - (int)(histi[i] * 255.0f / maxf));
			}
			pen.Dispose();
			gr.Dispose(); 
			return hist;
		}

		//Draw histogram (three channels)
		private Bitmap GetHistogramImage(int[] histi1, int[] histi2, int[] histi3, int channel)
		{
			Bitmap hist = new Bitmap(255, 255, PixelFormat.Format24bppRgb);
			Graphics gr = Graphics.FromImage(hist);

			gr.Clear(this.BackColor);
			int maxi = 0;
			for (int i = 0; i < 255; i++)
			{
				maxi = Math.Max(maxi, histi1[i]);
				maxi = Math.Max(maxi, histi2[i]);
				maxi = Math.Max(maxi, histi3[i]);
			}
			float maxf = (float)maxi;
			lbl_max.Text = maxi.ToString();
			Color c = Color.DarkGray;
			switch (channel)
			{
				case 0: c = Color.DarkGray; break;
				case 1: c = Color.Red; break;
				case 2: c = Color.Green; break;
				case 3: c = Color.Blue; break;
			}

			Pen pen = new Pen(c);

			for (int i = 0; i < 255; i++)
			{
				gr.DrawLine(pen, i, 255, i, 254 - (int)(histi1[i] * 255.0f / maxf));
			}
			pen.Dispose();
			gr.Dispose();
			return hist;
		}

		//Cleanup when closing
		private void Form1_FormClosing(object sender, FormClosingEventArgs e)
		{
			CleanUp();
		}
	}
}
