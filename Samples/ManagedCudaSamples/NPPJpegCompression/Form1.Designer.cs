namespace NPPJpegCompression
{
	partial class Form1
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.btn_OpenNPP = new System.Windows.Forms.Button();
			this.btn_openImageNet = new System.Windows.Forms.Button();
			this.btn_SaveJpegNPP = new System.Windows.Forms.Button();
			this.trk_JpegQuality = new System.Windows.Forms.TrackBar();
			this.txt_JpegQuality = new System.Windows.Forms.TextBox();
			this.label1 = new System.Windows.Forms.Label();
			this.pic_Image = new System.Windows.Forms.PictureBox();
			this.trk_Size = new System.Windows.Forms.TrackBar();
			this.label2 = new System.Windows.Forms.Label();
			this.txt_Resize = new System.Windows.Forms.TextBox();
			this.btn_Resize = new System.Windows.Forms.Button();
			((System.ComponentModel.ISupportInitialize)(this.trk_JpegQuality)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.pic_Image)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trk_Size)).BeginInit();
			this.SuspendLayout();
			// 
			// btn_OpenNPP
			// 
			this.btn_OpenNPP.Location = new System.Drawing.Point(12, 12);
			this.btn_OpenNPP.Name = "btn_OpenNPP";
			this.btn_OpenNPP.Size = new System.Drawing.Size(137, 23);
			this.btn_OpenNPP.TabIndex = 0;
			this.btn_OpenNPP.Text = "Open Jpeg with NPP";
			this.btn_OpenNPP.UseVisualStyleBackColor = true;
			this.btn_OpenNPP.Click += new System.EventHandler(this.btn_OpenNPP_Click);
			// 
			// btn_openImageNet
			// 
			this.btn_openImageNet.Location = new System.Drawing.Point(12, 41);
			this.btn_openImageNet.Name = "btn_openImageNet";
			this.btn_openImageNet.Size = new System.Drawing.Size(137, 23);
			this.btn_openImageNet.TabIndex = 1;
			this.btn_openImageNet.Text = "Open image with .net";
			this.btn_openImageNet.UseVisualStyleBackColor = true;
			this.btn_openImageNet.Click += new System.EventHandler(this.btn_openImageNet_Click);
			// 
			// btn_SaveJpegNPP
			// 
			this.btn_SaveJpegNPP.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.btn_SaveJpegNPP.Location = new System.Drawing.Point(694, 12);
			this.btn_SaveJpegNPP.Name = "btn_SaveJpegNPP";
			this.btn_SaveJpegNPP.Size = new System.Drawing.Size(137, 23);
			this.btn_SaveJpegNPP.TabIndex = 2;
			this.btn_SaveJpegNPP.Text = "Save Jpeg NPP";
			this.btn_SaveJpegNPP.UseVisualStyleBackColor = true;
			this.btn_SaveJpegNPP.Click += new System.EventHandler(this.btn_SaveJpegNPP_Click);
			// 
			// trk_JpegQuality
			// 
			this.trk_JpegQuality.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.trk_JpegQuality.Location = new System.Drawing.Point(643, 41);
			this.trk_JpegQuality.Maximum = 100;
			this.trk_JpegQuality.Minimum = 1;
			this.trk_JpegQuality.Name = "trk_JpegQuality";
			this.trk_JpegQuality.Size = new System.Drawing.Size(188, 45);
			this.trk_JpegQuality.TabIndex = 3;
			this.trk_JpegQuality.Value = 75;
			this.trk_JpegQuality.Scroll += new System.EventHandler(this.trk_JpegQuality_Scroll);
			// 
			// txt_JpegQuality
			// 
			this.txt_JpegQuality.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.txt_JpegQuality.Location = new System.Drawing.Point(643, 15);
			this.txt_JpegQuality.Name = "txt_JpegQuality";
			this.txt_JpegQuality.ReadOnly = true;
			this.txt_JpegQuality.Size = new System.Drawing.Size(45, 20);
			this.txt_JpegQuality.TabIndex = 4;
			this.txt_JpegQuality.Text = "75";
			this.txt_JpegQuality.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
			// 
			// label1
			// 
			this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(595, 50);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(42, 13);
			this.label1.TabIndex = 5;
			this.label1.Text = "Quality:";
			// 
			// pic_Image
			// 
			this.pic_Image.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
			this.pic_Image.Location = new System.Drawing.Point(12, 83);
			this.pic_Image.Name = "pic_Image";
			this.pic_Image.Size = new System.Drawing.Size(819, 542);
			this.pic_Image.TabIndex = 6;
			this.pic_Image.TabStop = false;
			// 
			// trk_Size
			// 
			this.trk_Size.Location = new System.Drawing.Point(205, 41);
			this.trk_Size.Maximum = 200;
			this.trk_Size.Minimum = 10;
			this.trk_Size.Name = "trk_Size";
			this.trk_Size.Size = new System.Drawing.Size(152, 45);
			this.trk_Size.TabIndex = 7;
			this.trk_Size.Value = 100;
			this.trk_Size.Scroll += new System.EventHandler(this.trk_Size_Scroll);
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(169, 50);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(30, 13);
			this.label2.TabIndex = 8;
			this.label2.Text = "Size:";
			// 
			// txt_Resize
			// 
			this.txt_Resize.Location = new System.Drawing.Point(172, 14);
			this.txt_Resize.Name = "txt_Resize";
			this.txt_Resize.ReadOnly = true;
			this.txt_Resize.Size = new System.Drawing.Size(45, 20);
			this.txt_Resize.TabIndex = 9;
			this.txt_Resize.Text = "100 %";
			this.txt_Resize.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
			// 
			// btn_Resize
			// 
			this.btn_Resize.Location = new System.Drawing.Point(223, 12);
			this.btn_Resize.Name = "btn_Resize";
			this.btn_Resize.Size = new System.Drawing.Size(134, 23);
			this.btn_Resize.TabIndex = 10;
			this.btn_Resize.Text = "Resize";
			this.btn_Resize.UseVisualStyleBackColor = true;
			this.btn_Resize.Click += new System.EventHandler(this.btn_Resize_Click);
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(843, 637);
			this.Controls.Add(this.pic_Image);
			this.Controls.Add(this.btn_Resize);
			this.Controls.Add(this.txt_Resize);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.trk_Size);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.txt_JpegQuality);
			this.Controls.Add(this.trk_JpegQuality);
			this.Controls.Add(this.btn_SaveJpegNPP);
			this.Controls.Add(this.btn_openImageNet);
			this.Controls.Add(this.btn_OpenNPP);
			this.Name = "Form1";
			this.Text = "Form1";
			((System.ComponentModel.ISupportInitialize)(this.trk_JpegQuality)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.pic_Image)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trk_Size)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button btn_OpenNPP;
		private System.Windows.Forms.Button btn_openImageNet;
		private System.Windows.Forms.Button btn_SaveJpegNPP;
		private System.Windows.Forms.TrackBar trk_JpegQuality;
		private System.Windows.Forms.TextBox txt_JpegQuality;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.PictureBox pic_Image;
		private System.Windows.Forms.TrackBar trk_Size;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox txt_Resize;
		private System.Windows.Forms.Button btn_Resize;
	}
}

