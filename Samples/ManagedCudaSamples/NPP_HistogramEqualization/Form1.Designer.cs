namespace NPP_HistogramEqualization
{
	partial class Form1
	{
		/// <summary>
		/// Erforderliche Designervariable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Verwendete Ressourcen bereinigen.
		/// </summary>
		/// <param name="disposing">True, wenn verwaltete Ressourcen gelöscht werden sollen; andernfalls False.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Vom Windows Form-Designer generierter Code

		/// <summary>
		/// Erforderliche Methode für die Designerunterstützung.
		/// Der Inhalt der Methode darf nicht mit dem Code-Editor geändert werden.
		/// </summary>
		private void InitializeComponent()
		{
			this.txt_info = new System.Windows.Forms.TextBox();
			this.pictureBox_src = new System.Windows.Forms.PictureBox();
			this.pictureBox_dest = new System.Windows.Forms.PictureBox();
			this.btn_open = new System.Windows.Forms.Button();
			this.btn_calc = new System.Windows.Forms.Button();
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.btn_Save = new System.Windows.Forms.Button();
			this.hist_rb_src = new System.Windows.Forms.PictureBox();
			this.hist_g_src = new System.Windows.Forms.PictureBox();
			this.hist_b_src = new System.Windows.Forms.PictureBox();
			this.hist_rb_dest = new System.Windows.Forms.PictureBox();
			this.hist_g_dest = new System.Windows.Forms.PictureBox();
			this.hist_b_dest = new System.Windows.Forms.PictureBox();
			this.label3 = new System.Windows.Forms.Label();
			this.label4 = new System.Windows.Forms.Label();
			this.label5 = new System.Windows.Forms.Label();
			this.label6 = new System.Windows.Forms.Label();
			this.label7 = new System.Windows.Forms.Label();
			this.label8 = new System.Windows.Forms.Label();
			this.label9 = new System.Windows.Forms.Label();
			this.label10 = new System.Windows.Forms.Label();
			this.label11 = new System.Windows.Forms.Label();
			this.label12 = new System.Windows.Forms.Label();
			this.label13 = new System.Windows.Forms.Label();
			this.label14 = new System.Windows.Forms.Label();
			this.label15 = new System.Windows.Forms.Label();
			this.lbl_max = new System.Windows.Forms.Label();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_src)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_dest)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_rb_src)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_g_src)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_b_src)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_rb_dest)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_g_dest)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_b_dest)).BeginInit();
			this.SuspendLayout();
			// 
			// txt_info
			// 
			this.txt_info.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
						| System.Windows.Forms.AnchorStyles.Right)));
			this.txt_info.Location = new System.Drawing.Point(12, 12);
			this.txt_info.Multiline = true;
			this.txt_info.Name = "txt_info";
			this.txt_info.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
			this.txt_info.Size = new System.Drawing.Size(1536, 64);
			this.txt_info.TabIndex = 0;
			// 
			// pictureBox_src
			// 
			this.pictureBox_src.Location = new System.Drawing.Point(12, 111);
			this.pictureBox_src.Name = "pictureBox_src";
			this.pictureBox_src.Size = new System.Drawing.Size(768, 576);
			this.pictureBox_src.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
			this.pictureBox_src.TabIndex = 1;
			this.pictureBox_src.TabStop = false;
			// 
			// pictureBox_dest
			// 
			this.pictureBox_dest.Location = new System.Drawing.Point(786, 111);
			this.pictureBox_dest.Name = "pictureBox_dest";
			this.pictureBox_dest.Size = new System.Drawing.Size(768, 576);
			this.pictureBox_dest.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
			this.pictureBox_dest.TabIndex = 1;
			this.pictureBox_dest.TabStop = false;
			// 
			// btn_open
			// 
			this.btn_open.Location = new System.Drawing.Point(637, 82);
			this.btn_open.Name = "btn_open";
			this.btn_open.Size = new System.Drawing.Size(93, 23);
			this.btn_open.TabIndex = 2;
			this.btn_open.Text = "Open Image";
			this.btn_open.UseVisualStyleBackColor = true;
			this.btn_open.Click += new System.EventHandler(this.btn_open_Click);
			// 
			// btn_calc
			// 
			this.btn_calc.Location = new System.Drawing.Point(736, 82);
			this.btn_calc.Name = "btn_calc";
			this.btn_calc.Size = new System.Drawing.Size(93, 23);
			this.btn_calc.TabIndex = 3;
			this.btn_calc.Text = "Calculate!";
			this.btn_calc.UseVisualStyleBackColor = true;
			this.btn_calc.Click += new System.EventHandler(this.btn_calc_Click);
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(12, 92);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(42, 13);
			this.label1.TabIndex = 4;
			this.label1.Text = "Original";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(1517, 92);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(37, 13);
			this.label2.TabIndex = 4;
			this.label2.Text = "Result";
			// 
			// btn_Save
			// 
			this.btn_Save.Location = new System.Drawing.Point(835, 82);
			this.btn_Save.Name = "btn_Save";
			this.btn_Save.Size = new System.Drawing.Size(93, 23);
			this.btn_Save.TabIndex = 5;
			this.btn_Save.Text = "Save image";
			this.btn_Save.UseVisualStyleBackColor = true;
			this.btn_Save.Click += new System.EventHandler(this.btn_Save_Click);
			// 
			// hist_rb_src
			// 
			this.hist_rb_src.Location = new System.Drawing.Point(12, 695);
			this.hist_rb_src.Name = "hist_rb_src";
			this.hist_rb_src.Size = new System.Drawing.Size(256, 256);
			this.hist_rb_src.TabIndex = 6;
			this.hist_rb_src.TabStop = false;
			// 
			// hist_g_src
			// 
			this.hist_g_src.Location = new System.Drawing.Point(268, 695);
			this.hist_g_src.Name = "hist_g_src";
			this.hist_g_src.Size = new System.Drawing.Size(256, 256);
			this.hist_g_src.TabIndex = 6;
			this.hist_g_src.TabStop = false;
			// 
			// hist_b_src
			// 
			this.hist_b_src.Location = new System.Drawing.Point(524, 695);
			this.hist_b_src.Name = "hist_b_src";
			this.hist_b_src.Size = new System.Drawing.Size(256, 256);
			this.hist_b_src.TabIndex = 6;
			this.hist_b_src.TabStop = false;
			// 
			// hist_rb_dest
			// 
			this.hist_rb_dest.Location = new System.Drawing.Point(786, 695);
			this.hist_rb_dest.Name = "hist_rb_dest";
			this.hist_rb_dest.Size = new System.Drawing.Size(256, 256);
			this.hist_rb_dest.TabIndex = 6;
			this.hist_rb_dest.TabStop = false;
			// 
			// hist_g_dest
			// 
			this.hist_g_dest.Location = new System.Drawing.Point(1042, 695);
			this.hist_g_dest.Name = "hist_g_dest";
			this.hist_g_dest.Size = new System.Drawing.Size(256, 256);
			this.hist_g_dest.TabIndex = 6;
			this.hist_g_dest.TabStop = false;
			// 
			// hist_b_dest
			// 
			this.hist_b_dest.Location = new System.Drawing.Point(1298, 695);
			this.hist_b_dest.Name = "hist_b_dest";
			this.hist_b_dest.Size = new System.Drawing.Size(256, 256);
			this.hist_b_dest.TabIndex = 6;
			this.hist_b_dest.TabStop = false;
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(755, 954);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(25, 13);
			this.label3.TabIndex = 7;
			this.label3.Text = "255";
			// 
			// label4
			// 
			this.label4.AutoSize = true;
			this.label4.Location = new System.Drawing.Point(499, 954);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(25, 13);
			this.label4.TabIndex = 7;
			this.label4.Text = "255";
			// 
			// label5
			// 
			this.label5.AutoSize = true;
			this.label5.Location = new System.Drawing.Point(243, 954);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(25, 13);
			this.label5.TabIndex = 7;
			this.label5.Text = "255";
			// 
			// label6
			// 
			this.label6.AutoSize = true;
			this.label6.Location = new System.Drawing.Point(1017, 954);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(25, 13);
			this.label6.TabIndex = 7;
			this.label6.Text = "255";
			// 
			// label7
			// 
			this.label7.AutoSize = true;
			this.label7.Location = new System.Drawing.Point(1273, 954);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(25, 13);
			this.label7.TabIndex = 7;
			this.label7.Text = "255";
			// 
			// label8
			// 
			this.label8.AutoSize = true;
			this.label8.Location = new System.Drawing.Point(1527, 954);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(25, 13);
			this.label8.TabIndex = 7;
			this.label8.Text = "255";
			// 
			// label9
			// 
			this.label9.AutoSize = true;
			this.label9.Location = new System.Drawing.Point(1295, 954);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(13, 13);
			this.label9.TabIndex = 7;
			this.label9.Text = "0";
			// 
			// label10
			// 
			this.label10.AutoSize = true;
			this.label10.Location = new System.Drawing.Point(1039, 954);
			this.label10.Name = "label10";
			this.label10.Size = new System.Drawing.Size(13, 13);
			this.label10.TabIndex = 7;
			this.label10.Text = "0";
			// 
			// label11
			// 
			this.label11.AutoSize = true;
			this.label11.Location = new System.Drawing.Point(783, 954);
			this.label11.Name = "label11";
			this.label11.Size = new System.Drawing.Size(13, 13);
			this.label11.TabIndex = 7;
			this.label11.Text = "0";
			// 
			// label12
			// 
			this.label12.AutoSize = true;
			this.label12.Location = new System.Drawing.Point(521, 954);
			this.label12.Name = "label12";
			this.label12.Size = new System.Drawing.Size(13, 13);
			this.label12.TabIndex = 7;
			this.label12.Text = "0";
			// 
			// label13
			// 
			this.label13.AutoSize = true;
			this.label13.Location = new System.Drawing.Point(265, 954);
			this.label13.Name = "label13";
			this.label13.Size = new System.Drawing.Size(13, 13);
			this.label13.TabIndex = 7;
			this.label13.Text = "0";
			// 
			// label14
			// 
			this.label14.AutoSize = true;
			this.label14.Location = new System.Drawing.Point(9, 954);
			this.label14.Name = "label14";
			this.label14.Size = new System.Drawing.Size(13, 13);
			this.label14.TabIndex = 7;
			this.label14.Text = "0";
			// 
			// label15
			// 
			this.label15.AutoSize = true;
			this.label15.Location = new System.Drawing.Point(0, 938);
			this.label15.Name = "label15";
			this.label15.Size = new System.Drawing.Size(13, 13);
			this.label15.TabIndex = 7;
			this.label15.Text = "0";
			// 
			// lbl_max
			// 
			this.lbl_max.AutoSize = true;
			this.lbl_max.Location = new System.Drawing.Point(0, 695);
			this.lbl_max.Name = "lbl_max";
			this.lbl_max.Size = new System.Drawing.Size(13, 13);
			this.lbl_max.TabIndex = 7;
			this.lbl_max.Text = "0";
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(1560, 970);
			this.Controls.Add(this.lbl_max);
			this.Controls.Add(this.label5);
			this.Controls.Add(this.label4);
			this.Controls.Add(this.label14);
			this.Controls.Add(this.label13);
			this.Controls.Add(this.label12);
			this.Controls.Add(this.label11);
			this.Controls.Add(this.label10);
			this.Controls.Add(this.label9);
			this.Controls.Add(this.label8);
			this.Controls.Add(this.label7);
			this.Controls.Add(this.label6);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.hist_b_dest);
			this.Controls.Add(this.hist_g_dest);
			this.Controls.Add(this.hist_rb_dest);
			this.Controls.Add(this.hist_b_src);
			this.Controls.Add(this.hist_g_src);
			this.Controls.Add(this.hist_rb_src);
			this.Controls.Add(this.btn_Save);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.btn_calc);
			this.Controls.Add(this.btn_open);
			this.Controls.Add(this.pictureBox_dest);
			this.Controls.Add(this.pictureBox_src);
			this.Controls.Add(this.txt_info);
			this.Controls.Add(this.label15);
			this.Name = "Form1";
			this.Text = "NPP Histogram equalization";
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
			this.Load += new System.EventHandler(this.Form1_Load);
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_src)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox_dest)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_rb_src)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_g_src)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_b_src)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_rb_dest)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_g_dest)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.hist_b_dest)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.TextBox txt_info;
		private System.Windows.Forms.PictureBox pictureBox_src;
		private System.Windows.Forms.PictureBox pictureBox_dest;
		private System.Windows.Forms.Button btn_open;
		private System.Windows.Forms.Button btn_calc;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Button btn_Save;
		private System.Windows.Forms.PictureBox hist_rb_src;
		private System.Windows.Forms.PictureBox hist_g_src;
		private System.Windows.Forms.PictureBox hist_b_src;
		private System.Windows.Forms.PictureBox hist_rb_dest;
		private System.Windows.Forms.PictureBox hist_g_dest;
		private System.Windows.Forms.PictureBox hist_b_dest;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.Label lbl_max;
	}
}

