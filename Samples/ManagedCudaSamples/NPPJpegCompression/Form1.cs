using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ManagedCuda.NPP;

namespace NPPJpegCompression
{
	public partial class Form1 : Form
	{
		public Form1()
		{
			InitializeComponent();
		}

		private void btn_OpenNPP_Click(object sender, EventArgs e)
		{
			OpenFileDialog ofd = new OpenFileDialog();
			ofd.Filter = "JPEG files|*.jpg";
			if (ofd.ShowDialog() != System.Windows.Forms.DialogResult.OK)
				return;

			try
			{
				pic_Image.Image = JpegNPP.LoadJpeg(ofd.FileName);
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message);
			}
		}

		private void btn_openImageNet_Click(object sender, EventArgs e)
		{
			OpenFileDialog ofd = new OpenFileDialog();
			ofd.Filter = "JPEG files|*.jpg";
			if (ofd.ShowDialog() != System.Windows.Forms.DialogResult.OK)
				return;

			pic_Image.Image = new Bitmap(ofd.FileName);
		}

		private void trk_Size_Scroll(object sender, EventArgs e)
		{
			txt_Resize.Text = trk_Size.Value.ToString() + " %";
		}

		private void btn_SaveJpegNPP_Click(object sender, EventArgs e)
		{
			if ((Bitmap)pic_Image.Image == null) return;

			SaveFileDialog sfd = new SaveFileDialog();
			sfd.Filter = "JPEG files|*.jpg";

			if (sfd.ShowDialog() != System.Windows.Forms.DialogResult.OK)
				return;

			try
			{
				JpegNPP.SaveJpeg(sfd.FileName, trk_JpegQuality.Value, (Bitmap)pic_Image.Image);
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message);
			}
		}

		private void trk_JpegQuality_Scroll(object sender, EventArgs e)
		{
			txt_JpegQuality.Text = trk_JpegQuality.Value.ToString();
		}

		private void btn_Resize_Click(object sender, EventArgs e)
		{
			if ((Bitmap)pic_Image.Image == null) return;

			Bitmap bmp = (Bitmap)pic_Image.Image;
			int w = bmp.Width;
			int h = bmp.Height;

			if ((w <= 16 || h <= 16) && trk_Size.Value < 100)
			{
				MessageBox.Show("Image is too small for resizing.");
				return;
			}
			
			int newW = (int)(trk_Size.Value / 100.0f * w);
			int newH = (int)(trk_Size.Value / 100.0f * h);

			if (newW % 16 != 0)
			{
				newW = newW - (newW % 16);
			}
			if (newW < 16) newW = 16;
			
			if (newH % 16 != 0)
			{
				newH = newH - (newH % 16);
			}
			if (newH < 16) newH = 16;
			
			double ratioW = newW / (double)w;
			double ratioH = newH / (double)h;

			if (ratioW == 1 && ratioH == 1)
				return;

			if (bmp.PixelFormat != System.Drawing.Imaging.PixelFormat.Format24bppRgb)
			{
				MessageBox.Show("Only three channel color images are supported!");
				return;
			}

			NPPImage_8uC3 imgIn = new NPPImage_8uC3(w, h);
			NPPImage_8uC3 imgOut = new NPPImage_8uC3(newW, newH);

			InterpolationMode interpol = InterpolationMode.SuperSampling;
			if (ratioH >= 1 || ratioW >= 1)
				interpol = InterpolationMode.Lanczos;

			imgIn.CopyToDevice(bmp);
			imgIn.ResizeSqrPixel(imgOut, ratioW, ratioH, 0, 0, interpol);
			Bitmap bmpRes = new Bitmap(newW, newH, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
			imgOut.CopyToHost(bmpRes);
			pic_Image.Image = bmpRes;

			imgIn.Dispose();
			imgOut.Dispose();
		}
	}
}
