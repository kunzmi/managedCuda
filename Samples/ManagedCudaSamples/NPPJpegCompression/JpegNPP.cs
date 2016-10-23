using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.NPP;

namespace NPPJpegCompression
{
	public static class JpegNPP
	{
		const int BUFFER_SIZE = 4 << 23; //32 MegaBytes

		#region Data structures
		private struct FrameHeader
		{
			public byte nSamplePrecision;
			public ushort nHeight;
			public ushort nWidth;
			public byte nComponents;
			public byte[] aComponentIdentifier;
			public byte[] aSamplingFactors;
			public byte[] aQuantizationTableSelector;

		};

		private struct ScanHeader
		{
			public byte nComponents;
			public byte[] aComponentSelector;
			public byte[] aHuffmanTablesSelector;
			public byte nSs;
			public byte nSe;
			public byte nA;

		};

		private class QuantizationTable
		{
			public enum QuantizationType
			{
				Zero,
				Luminance,
				Chroma
			}

			public byte nPrecisionAndIdentifier;
			public byte[] aTable;

			public QuantizationTable() :
				this(QuantizationType.Zero, 0)
			{ }

			//Tables as given in JPEG standard / LibJPEG
			public QuantizationTable(QuantizationType type, int quality)
			{
				switch (type)
				{
					case QuantizationType.Zero:
						aTable = new byte[64];
						nPrecisionAndIdentifier = 0;
						break;
					case QuantizationType.Luminance:
						aTable = new byte[] {   16,  11,  10,  16,  24,  40,  51,  61,
                                                12,  12,  14,  19,  26,  58,  60,  55,
                                                14,  13,  16,  24,  40,  57,  69,  56,
                                                14,  17,  22,  29,  51,  87,  80,  62,
                                                18,  22,  37,  56,  68, 109, 103,  77,
                                                24,  35,  55,  64,  81, 104, 113,  92,
                                                49,  64,  78,  87, 103, 121, 120, 101,
                                                72,  92,  95,  98, 112, 100, 103,  99};
						nPrecisionAndIdentifier = 0;
						break;
					case QuantizationType.Chroma:
						aTable = new byte[] {   17,  18,  24,  47,  99,  99,  99,  99,
                                                18,  21,  26,  66,  99,  99,  99,  99,
                                                24,  26,  56,  99,  99,  99,  99,  99,
                                                47,  66,  99,  99,  99,  99,  99,  99,
                                                99,  99,  99,  99,  99,  99,  99,  99,
                                                99,  99,  99,  99,  99,  99,  99,  99,
                                                99,  99,  99,  99,  99,  99,  99,  99,
                                                99,  99,  99,  99,  99,  99,  99,  99 };
						nPrecisionAndIdentifier = 1;
						break;
					default:
						aTable = new byte[64];
						break;
				}

				if (type != QuantizationType.Zero)
				{
					if (quality <= 0) quality = 1;
					if (quality > 100) quality = 100;

					if (quality < 50)
						quality = 5000 / quality;
					else
						quality = 200 - quality * 2;

					for (int i = 0; i < aTable.Length; i++)
					{
						int temp = ((int)aTable[i] * quality + 50) / 100;
						/* limit the values to the valid range */
						if (temp <= 0L) temp = 1;
						if (temp > 32767L) temp = 32767; /* max quantizer needed for 12 bits */
						bool force_baseline = true;
						if (force_baseline && temp > 255L)
							temp = 255;		/* limit to baseline range if requested */
						aTable[i] = (byte)temp;
					}
				}
			}

		};

		private class HuffmanTable
		{
			public enum HuffmanType
			{
				Zero,
				LuminanceDC,
				ChromaDC,
				LuminanceAC,
				ChromaAC
			}

			public byte nClassAndIdentifier;
			public byte[] aCodes; //aCodes and aTable must be one continuous memory segment!
			//public byte[] aTable;

			public HuffmanTable() :
				this(HuffmanType.Zero)
			{ }

			//Tables as given in JPEG standard / LibJPEG
			public HuffmanTable(HuffmanType type)
			{
				switch (type)
				{
					case HuffmanType.Zero:
						aCodes = new byte[16 + 256];
						nClassAndIdentifier = 0;
						break;
					case HuffmanType.LuminanceDC:
						aCodes = new byte[16 + 256] { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, //bits
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, //vals
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0
                        };
						nClassAndIdentifier = 0;
						break;
					case HuffmanType.ChromaDC:
						aCodes = new byte[16 + 256] { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, //bits
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, //vals
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0
                        };
						nClassAndIdentifier = 1;
						break;
					case HuffmanType.LuminanceAC:
						aCodes = new byte[16 + 256] { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d, //bits
                            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, //vals
                            0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
                            0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
                            0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
                            0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
                            0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                            0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
                            0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
                            0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
                            0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                            0xf9, 0xfa, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                        };
						nClassAndIdentifier = 16;
						break;
					case HuffmanType.ChromaAC:
						aCodes = new byte[16 + 256] { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77, //bits
                            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, //vals
                            0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
                            0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
                            0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
                            0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
                            0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                            0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
                            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
                            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
                            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                            0xf9, 0xfa, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                        };
						nClassAndIdentifier = 17;
						break;
					default:
						break;
				}
			}

		};
		#endregion

		#region internal methods (more or less 1:1 from NPP Jpeg sample)
		private static void writeMarker(byte nMarker, byte[] pData, ref int pos)
		{
			pData[pos] = 0x0ff; pos++;
			pData[pos] = nMarker; pos++;
		}

		private static void write(byte[] pData, ushort value, ref int pos)
		{
			byte s1, s2;
			s1 = (byte)(value / 256);
			s2 = (byte)(value - (s1 * 256));
			
			pData[pos] = s1; pos++;
			pData[pos] = s2; pos++;
		}

		private static void write(byte[] pData, byte value, ref int pos)
		{
			pData[pos] = value; pos++;
		}

		private static void writeJFIFTag(byte[] pData, ref int pos)
		{
			byte[] JFIF_TAG = new byte[]
			{
				0x4a, 0x46, 0x49, 0x46, 0x00,
				0x01, 0x02,
				0x00,
				0x00, 0x01, 0x00, 0x01,
				0x00, 0x00
			};

			writeMarker(0x0e0, pData, ref pos);
			write(pData, (ushort)(JFIF_TAG.Length + 2), ref pos);
			for (int i = 0; i < JFIF_TAG.Length; i++)
			{
				pData[pos + i] = JFIF_TAG[i];
			}

			pos += JFIF_TAG.Length;
		}

		private static void writeFrameHeader(FrameHeader header, byte[] pData, ref int pos)
		{
			byte[] pTemp = new byte[128];
			int posTemp = 0;
			write(pTemp, header.nSamplePrecision, ref posTemp);
			write(pTemp, header.nHeight, ref posTemp);
			write(pTemp, header.nWidth, ref posTemp);
			write(pTemp, header.nComponents, ref posTemp);


			for (int c = 0; c < header.nComponents; ++c)
			{
				write(pTemp, header.aComponentIdentifier[c], ref posTemp);
				write(pTemp, header.aSamplingFactors[c], ref posTemp);
				write(pTemp, header.aQuantizationTableSelector[c], ref posTemp);
			}

			ushort nLength = (ushort)(posTemp);

			writeMarker(0x0C0, pData, ref pos);
			write(pData, (ushort)(nLength + 2), ref pos);
			for (int i = 0; i < nLength; i++)
			{
				pData[pos + i] = pTemp[i];
			}
			pos += nLength;
		}

		private static void writeScanHeader(ScanHeader header, byte[] pData, ref int pos)
		{
			byte[] pTemp = new byte[128];
			int posTemp = 0;

			write(pTemp, header.nComponents, ref posTemp);

			for (int c = 0; c < header.nComponents; ++c)
			{
				write(pTemp, header.aComponentSelector[c], ref posTemp);
				write(pTemp, header.aHuffmanTablesSelector[c], ref posTemp);
			}

			write(pTemp, header.nSs, ref posTemp);
			write(pTemp, header.nSe, ref posTemp);
			write(pTemp, header.nA, ref posTemp);

			ushort nLength = (ushort)(posTemp);

			writeMarker(0x0DA, pData, ref pos);
			write(pData, (ushort)(nLength + 2), ref pos);
			for (int i = 0; i < nLength; i++)
			{
				pData[pos + i] = pTemp[i];
			}
			pos += nLength;
		}

		private static void writeQuantizationTable(QuantizationTable table, byte[] pData, ref int pos)
		{
			writeMarker(0x0DB, pData, ref pos);
			write(pData, (ushort)(65 + 2), ref pos);

			write(pData, table.nPrecisionAndIdentifier, ref pos);
			for (int i = 0; i < 64; i++)
			{
				pData[pos + i] = table.aTable[i];
			}
			pos += 64;
		}

		private static void writeHuffmanTable(HuffmanTable table, byte[] pData, ref int pos)
		{
			writeMarker(0x0C4, pData, ref pos);

			// Number of Codes for Bit Lengths [1..16]
			int nCodeCount = 0;

			for (int i = 0; i < 16; ++i)
			{
				nCodeCount += table.aCodes[i];
			}

			write(pData, (ushort)(17 + nCodeCount + 2), ref pos);

			write(pData, table.nClassAndIdentifier, ref pos);
			for (int i = 0; i < 16; i++)
			{
				pData[pos + i] = table.aCodes[i];
			}
			pos += 16;
			for (int i = 0; i < nCodeCount; i++)
			{
				pData[pos + i] = table.aCodes[i + 16];
			}
			pos += nCodeCount;
		}

		private static int DivUp(int x, int d)
		{
			return (x + d - 1) / d;
		}

		private static ushort readUShort(byte[] pData, ref int pos)
		{
			byte s1 = pData[pos], s2 = pData[pos + 1];
			pos += 2;

			return (ushort)(256 * s1 + s2);
		}

		private static byte readByte(byte[] pData, ref int pos)
		{
			byte s1 = pData[pos];
			pos++;

			return s1;
		}

		private static int nextMarker(byte[] pData, ref int nPos, int nLength)
		{
			if (nPos >= nLength)
				return -1;
			byte c = pData[nPos];
			nPos++;

			do
			{
				while (c != 0xffu && nPos < nLength)
				{
					c = pData[nPos];
					nPos++;
				}

				if (nPos >= nLength)
					return -1;

				c = pData[nPos];
				nPos++;
			}
			while (c == 0 || c == 0x0ffu);

			return c;
		}

		private static void readFrameHeader(byte[] pData, ref int p, ref FrameHeader header)
		{
			int pos = p;
			readUShort(pData, ref pos);
			header.nSamplePrecision = readByte(pData, ref pos);
			header.nHeight = readUShort(pData, ref pos);
			header.nWidth = readUShort(pData, ref pos);
			header.nComponents = readByte(pData, ref pos);


			for (int c = 0; c < header.nComponents; ++c)
			{
				header.aComponentIdentifier[c] = readByte(pData, ref pos);
				header.aSamplingFactors[c] = readByte(pData, ref pos);
				header.aQuantizationTableSelector[c] = readByte(pData, ref pos);
			}

		}

		private static void readScanHeader(byte[] pData, ref int p, ref ScanHeader header)
		{
			int pos = p;
			readUShort(pData, ref pos);

			header.nComponents = readByte(pData, ref pos);

			for (int c = 0; c < header.nComponents; ++c)
			{
				header.aComponentSelector[c] = readByte(pData, ref pos);
				header.aHuffmanTablesSelector[c] = readByte(pData, ref pos);
			}

			header.nSs = readByte(pData, ref pos);
			header.nSe = readByte(pData, ref pos);
			header.nA = readByte(pData, ref pos);
		}

		private static void readQuantizationTables(byte[] pData, ref int p, QuantizationTable[] pTables)
		{
			int pos = p;
			int nLength = readUShort(pData, ref pos) - 2;

			while (nLength > 0)
			{
				byte nPrecisionAndIdentifier = readByte(pData, ref pos);
				int nIdentifier = nPrecisionAndIdentifier & 0x0f;

				pTables[nIdentifier].nPrecisionAndIdentifier = nPrecisionAndIdentifier;
				for (int i = 0; i < 64; i++)
				{
					pTables[nIdentifier].aTable[i] = readByte(pData, ref pos);
				}
				nLength -= 65;
			}
		}

		private static void readHuffmanTables(byte[] pData, ref int p, HuffmanTable[] pTables)
		{
			int pos = p;
			int nLength = readUShort(pData, ref pos) - 2;

			while (nLength > 0)
			{
				byte nClassAndIdentifier = readByte(pData, ref pos);
				int nClass = nClassAndIdentifier >> 4; // AC or DC
				int nIdentifier = nClassAndIdentifier & 0x0f;
				int nIdx = nClass * 2 + nIdentifier;
				pTables[nIdx].nClassAndIdentifier = nClassAndIdentifier;

				// Number of Codes for Bit Lengths [1..16]
				int nCodeCount = 0;

				for (int i = 0; i < 16; ++i)
				{
					pTables[nIdx].aCodes[i] = readByte(pData, ref pos);
					nCodeCount += pTables[nIdx].aCodes[i];
				}
				for (int i = 0; i < nCodeCount; i++)
				{
					pTables[nIdx].aCodes[i + 16] = readByte(pData, ref pos);
				}

				nLength -= 17 + nCodeCount;
			}
		}

		private static void readRestartInterval(byte[] pData, ref int pos, ref int nRestartInterval)
		{
			int p = pos;
			readUShort(pData, ref p);
			nRestartInterval = readUShort(pData, ref p);
		}
		#endregion


		public static Bitmap LoadJpeg(string aFilename)
		{
			JPEGCompression compression = new JPEGCompression();
			byte[] pJpegData = File.ReadAllBytes(aFilename);
			int nInputLength = pJpegData.Length;


			// Check if this is a valid JPEG file
			int nPos = 0;
			int nMarker = nextMarker(pJpegData, ref nPos, nInputLength);

			if (nMarker != 0x0D8)
			{
				throw new ArgumentException(aFilename + " is not a JPEG file.");
			}

			nMarker = nextMarker(pJpegData, ref nPos, nInputLength);

			// Parsing and Huffman Decoding (on host)
			FrameHeader oFrameHeader = new FrameHeader();

			oFrameHeader.aComponentIdentifier = new byte[3];
			oFrameHeader.aSamplingFactors = new byte[3];
			oFrameHeader.aQuantizationTableSelector = new byte[3];

			QuantizationTable[] aQuantizationTables = new QuantizationTable[4];
			aQuantizationTables[0] = new QuantizationTable();
			aQuantizationTables[1] = new QuantizationTable();
			aQuantizationTables[2] = new QuantizationTable();
			aQuantizationTables[3] = new QuantizationTable();


			CudaDeviceVariable<byte>[] pdQuantizationTables = new CudaDeviceVariable<byte>[4];
			pdQuantizationTables[0] = new CudaDeviceVariable<byte>(64);
			pdQuantizationTables[1] = new CudaDeviceVariable<byte>(64);
			pdQuantizationTables[2] = new CudaDeviceVariable<byte>(64);
			pdQuantizationTables[3] = new CudaDeviceVariable<byte>(64);

			HuffmanTable[] aHuffmanTables = new HuffmanTable[4];
			aHuffmanTables[0] = new HuffmanTable();
			aHuffmanTables[1] = new HuffmanTable();
			aHuffmanTables[2] = new HuffmanTable();
			aHuffmanTables[3] = new HuffmanTable();


			ScanHeader oScanHeader = new ScanHeader();
			oScanHeader.aComponentSelector = new byte[3];
			oScanHeader.aHuffmanTablesSelector = new byte[3];


			int nMCUBlocksH = 0;
			int nMCUBlocksV = 0;

			int nRestartInterval = -1;

			NppiSize[] aSrcSize = new NppiSize[3];

			short[][] aphDCT = new short[3][];
			NPPImage_16sC1[] apdDCT = new NPPImage_16sC1[3];
			int[] aDCTStep = new int[3];

			NPPImage_8uC1[] apSrcImage = new NPPImage_8uC1[3];
			int[] aSrcImageStep = new int[3];

			NPPImage_8uC1[] apDstImage = new NPPImage_8uC1[3];
			int[] aDstImageStep = new int[3];
			NppiSize[] aDstSize = new NppiSize[3];

			//Same read routine as in NPP JPEG sample from Nvidia
			while (nMarker != -1)
			{
				if (nMarker == 0x0D8)
				{
					// Embeded Thumbnail, skip it
					int nNextMarker = nextMarker(pJpegData, ref nPos, nInputLength);

					while (nNextMarker != -1 && nNextMarker != 0x0D9)
					{
						nNextMarker = nextMarker(pJpegData, ref nPos, nInputLength);
					}
				}

				if (nMarker == 0x0DD)
				{
					readRestartInterval(pJpegData, ref nPos, ref nRestartInterval);
				}

				if ((nMarker == 0x0C0) | (nMarker == 0x0C2))
				{
					//Assert Baseline for this Sample
					//Note: NPP does support progressive jpegs for both encode and decode
					if (nMarker != 0x0C0)
					{
						pdQuantizationTables[0].Dispose();
						pdQuantizationTables[1].Dispose();
						pdQuantizationTables[2].Dispose();
						pdQuantizationTables[3].Dispose();

						throw new ArgumentException(aFilename + " is not a Baseline-JPEG file.");
					}

					// Baseline or Progressive Frame Header
					readFrameHeader(pJpegData, ref nPos, ref oFrameHeader);
					//Console.WriteLine("Image Size: " + oFrameHeader.nWidth + "x" + oFrameHeader.nHeight + "x" + (int)(oFrameHeader.nComponents));

					//Assert 3-Channel Image for this Sample
					if (oFrameHeader.nComponents != 3)
					{
						pdQuantizationTables[0].Dispose();
						pdQuantizationTables[1].Dispose();
						pdQuantizationTables[2].Dispose();
						pdQuantizationTables[3].Dispose();

						throw new ArgumentException(aFilename + " is not a three channel JPEG file.");
					}

					// Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
					for (int i = 0; i < oFrameHeader.nComponents; ++i)
					{
						nMCUBlocksV = Math.Max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] >> 4);
						nMCUBlocksH = Math.Max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] & 0x0f);
					}

					for (int i = 0; i < oFrameHeader.nComponents; ++i)
					{
						NppiSize oBlocks = new NppiSize();
						NppiSize oBlocksPerMCU = new NppiSize(oFrameHeader.aSamplingFactors[i] & 0x0f, oFrameHeader.aSamplingFactors[i] >> 4);

						oBlocks.width = (int)Math.Ceiling((oFrameHeader.nWidth + 7) / 8 *
												  (float)(oBlocksPerMCU.width) / nMCUBlocksH);
						oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

						oBlocks.height = (int)Math.Ceiling((oFrameHeader.nHeight + 7) / 8 *
												   (float)(oBlocksPerMCU.height) / nMCUBlocksV);
						oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

						aSrcSize[i].width = oBlocks.width * 8;
						aSrcSize[i].height = oBlocks.height * 8;

						// Allocate Memory
						apdDCT[i] = new NPPImage_16sC1(oBlocks.width * 64, oBlocks.height);
						aDCTStep[i] = apdDCT[i].Pitch;

						apSrcImage[i] = new NPPImage_8uC1(aSrcSize[i].width, aSrcSize[i].height);
						aSrcImageStep[i] = apSrcImage[i].Pitch;

						aphDCT[i] = new short[aDCTStep[i] * oBlocks.height];
					}
				}

				if (nMarker == 0x0DB)
				{
					// Quantization Tables
					readQuantizationTables(pJpegData, ref nPos, aQuantizationTables);
				}

				if (nMarker == 0x0C4)
				{
					// Huffman Tables
					readHuffmanTables(pJpegData, ref nPos, aHuffmanTables);
				}

				if (nMarker == 0x0DA)
				{
					// Scan
					readScanHeader(pJpegData, ref nPos, ref oScanHeader);
					nPos += 6 + oScanHeader.nComponents * 2;

					int nAfterNextMarkerPos = nPos;
					int nAfterScanMarker = nextMarker(pJpegData, ref nAfterNextMarkerPos, nInputLength);

					if (nRestartInterval > 0)
					{
						while (nAfterScanMarker >= 0x0D0 && nAfterScanMarker <= 0x0D7)
						{
							// This is a restart marker, go on
							nAfterScanMarker = nextMarker(pJpegData, ref nAfterNextMarkerPos, nInputLength);
						}
					}

					NppiDecodeHuffmanSpec[] apHuffmanDCTableDec = new NppiDecodeHuffmanSpec[3];
					NppiDecodeHuffmanSpec[] apHuffmanACTableDec = new NppiDecodeHuffmanSpec[3];

					for (int i = 0; i < 3; ++i)
					{
						apHuffmanDCTableDec[i] = JPEGCompression.DecodeHuffmanSpecInitAllocHost(aHuffmanTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, NppiHuffmanTableType.nppiDCTable);
						apHuffmanACTableDec[i] = JPEGCompression.DecodeHuffmanSpecInitAllocHost(aHuffmanTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f) + 2].aCodes, NppiHuffmanTableType.nppiACTable);
					}

					byte[] img = new byte[nAfterNextMarkerPos - nPos - 2];
					Buffer.BlockCopy(pJpegData, nPos, img, 0, nAfterNextMarkerPos - nPos - 2);

					
					JPEGCompression.DecodeHuffmanScanHost(img, nRestartInterval, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f, aphDCT[0], aphDCT[1], aphDCT[2], aDCTStep, apHuffmanDCTableDec, apHuffmanACTableDec, aSrcSize);

					for (int i = 0; i < 3; ++i)
					{
						JPEGCompression.DecodeHuffmanSpecFreeHost(apHuffmanDCTableDec[i]);
						JPEGCompression.DecodeHuffmanSpecFreeHost(apHuffmanACTableDec[i]);
					}
				}

				nMarker = nextMarker(pJpegData, ref nPos, nInputLength);
			}
			

			// Copy DCT coefficients and Quantization Tables from host to device
			for (int i = 0; i < 4; ++i)
			{
				pdQuantizationTables[i].CopyToDevice(aQuantizationTables[i].aTable);
			}

			for (int i = 0; i < 3; ++i)
			{
				apdDCT[i].CopyToDevice(aphDCT[i], aDCTStep[i]);
			}

			// Inverse DCT
			for (int i = 0; i < 3; ++i)
			{
				compression.DCTQuantInv8x8LS(apdDCT[i], apSrcImage[i], aSrcSize[i], pdQuantizationTables[oFrameHeader.aQuantizationTableSelector[i]]);
			}

			//Alloc final image
			NPPImage_8uC3 res = new NPPImage_8uC3(apSrcImage[0].Width, apSrcImage[0].Height);

			//Copy Y color plane to first channel
			apSrcImage[0].Copy(res, 0);

			//Cb anc Cr channel might be smaller
			if ((oFrameHeader.aSamplingFactors[0] & 0x0f) == 1 && oFrameHeader.aSamplingFactors[0] >> 4 == 1)
			{
				//Color planes are of same size as Y channel
				apSrcImage[1].Copy(res, 1);
				apSrcImage[2].Copy(res, 2);
			}
			else
			{
				//rescale color planes to full size
				double scaleX = oFrameHeader.aSamplingFactors[0] & 0x0f;
				double scaleY = oFrameHeader.aSamplingFactors[0] >> 4;

				apSrcImage[1].ResizeSqrPixel(apSrcImage[0], scaleX, scaleY, 0, 0, InterpolationMode.Lanczos);
				apSrcImage[0].Copy(res, 1);
				apSrcImage[2].ResizeSqrPixel(apSrcImage[0], scaleX, scaleY, 0, 0, InterpolationMode.Lanczos);
				apSrcImage[0].Copy(res, 2);				
			}


			//System.Drawing.Bitmap is ordered BGR not RGB
			//The NPP routine YCbCR to BGR needs clampled input values, following the YCbCr standard.
			//But JPEG uses unclamped values ranging all from [0..255], thus use our own color matrix:
			float[,] YCbCrToBgr = new float[3, 4]
			{{1.0f, 1.772f,     0.0f,    -226.816f  },
			 {1.0f, -0.34414f, -0.71414f, 135.45984f},
			 {1.0f, 0.0f,       1.402f,  -179.456f  }};

			//Convert from YCbCr to BGR
			res.ColorTwist(YCbCrToBgr);
			
			Bitmap bmp = new Bitmap(apSrcImage[0].Width, apSrcImage[0].Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
			res.CopyToHost(bmp);

			//Cleanup:
			res.Dispose();
			apSrcImage[2].Dispose();
			apSrcImage[1].Dispose();
			apSrcImage[0].Dispose();

			apdDCT[2].Dispose();
			apdDCT[1].Dispose();
			apdDCT[0].Dispose();

			pdQuantizationTables[0].Dispose();
			pdQuantizationTables[1].Dispose();
			pdQuantizationTables[2].Dispose();
			pdQuantizationTables[3].Dispose();

			compression.Dispose();

			return bmp;
		}

		public static void SaveJpeg(string aFilename, int aQuality, Bitmap aImage)
		{
			if (aImage.PixelFormat != System.Drawing.Imaging.PixelFormat.Format24bppRgb)
			{
				throw new ArgumentException("Only three channel color images are supported.");
			}

			if (aImage.Width % 16 != 0 || aImage.Height % 16 != 0)
			{
				throw new ArgumentException("The provided bitmap must have a height and width of a multiple of 16.");
			}
			
			JPEGCompression compression = new JPEGCompression();

			NPPImage_8uC3 src = new NPPImage_8uC3(aImage.Width, aImage.Height);
			NPPImage_8uC1 srcY = new NPPImage_8uC1(aImage.Width, aImage.Height);
			NPPImage_8uC1 srcCb = new NPPImage_8uC1(aImage.Width / 2, aImage.Height / 2);
			NPPImage_8uC1 srcCr = new NPPImage_8uC1(aImage.Width / 2, aImage.Height / 2);
			src.CopyToDevice(aImage);

			//System.Drawing.Bitmap is ordered BGR not RGB
			//The NPP routine BGR to YCbCR outputs the values in clamped range, following the YCbCr standard.
			//But JPEG uses unclamped values ranging all from [0..255], thus use our own color matrix:
			float[,] BgrToYCbCr = new float[3, 4]
			{{0.114f,     0.587f,    0.299f,   0},
			 {0.5f,      -0.33126f, -0.16874f, 128},
			 {-0.08131f, -0.41869f,  0.5f,     128}};


			src.ColorTwist(BgrToYCbCr);

			//Reduce size of of Cb and Cr channel
			src.Copy(srcY, 2);
			srcY.Resize(srcCr, 0.5, 0.5, InterpolationMode.SuperSampling);
			src.Copy(srcY, 1);
			srcY.Resize(srcCb, 0.5, 0.5, InterpolationMode.SuperSampling);
			src.Copy(srcY, 0);
			

			FrameHeader oFrameHeader = new FrameHeader();
			oFrameHeader.nComponents = 3;
			oFrameHeader.nHeight = (ushort)aImage.Height;
			oFrameHeader.nSamplePrecision = 8;
			oFrameHeader.nWidth = (ushort)aImage.Width;
			oFrameHeader.aComponentIdentifier = new byte[] { 1, 2, 3 };
			oFrameHeader.aSamplingFactors = new byte[] { 34, 17, 17 }; //Y channel is twice the sice of Cb/Cr channel
			oFrameHeader.aQuantizationTableSelector = new byte[] { 0, 1, 1 };

			//Get quantization tables from JPEG standard with quality scaling
			QuantizationTable[] aQuantizationTables = new QuantizationTable[2];
			aQuantizationTables[0] = new QuantizationTable(QuantizationTable.QuantizationType.Luminance, aQuality);
			aQuantizationTables[1] = new QuantizationTable(QuantizationTable.QuantizationType.Chroma, aQuality);


			CudaDeviceVariable<byte>[] pdQuantizationTables = new CudaDeviceVariable<byte>[2];
			pdQuantizationTables[0] = aQuantizationTables[0].aTable;
			pdQuantizationTables[1] = aQuantizationTables[1].aTable;


			//Get Huffman tables from JPEG standard
			HuffmanTable[] aHuffmanTables = new HuffmanTable[4];
			aHuffmanTables[0] = new HuffmanTable(HuffmanTable.HuffmanType.LuminanceDC);
			aHuffmanTables[1] = new HuffmanTable(HuffmanTable.HuffmanType.ChromaDC);
			aHuffmanTables[2] = new HuffmanTable(HuffmanTable.HuffmanType.LuminanceAC);
			aHuffmanTables[3] = new HuffmanTable(HuffmanTable.HuffmanType.ChromaAC);

			//Set header
			ScanHeader oScanHeader = new ScanHeader();
			oScanHeader.nA = 0;
			oScanHeader.nComponents = 3;
			oScanHeader.nSe = 63;
			oScanHeader.nSs = 0;
			oScanHeader.aComponentSelector = new byte[] { 1, 2, 3 };
			oScanHeader.aHuffmanTablesSelector = new byte[] { 0, 17, 17 };


			NPPImage_16sC1[] apdDCT = new NPPImage_16sC1[3];
			
			NPPImage_8uC1[] apDstImage = new NPPImage_8uC1[3];
			NppiSize[] aDstSize = new NppiSize[3];
			aDstSize[0] = new NppiSize(srcY.Width, srcY.Height);
			aDstSize[1] = new NppiSize(srcCb.Width, srcCb.Height);
			aDstSize[2] = new NppiSize(srcCr.Width, srcCr.Height);


			// Compute channel sizes as stored in the output JPEG (8x8 blocks & MCU block layout)
			NppiSize oDstImageSize = new NppiSize();
			float frameWidth = (float)Math.Floor((float)oFrameHeader.nWidth);
			float frameHeight = (float)Math.Floor((float)oFrameHeader.nHeight);

			oDstImageSize.width = (int)Math.Max(1.0f, frameWidth);
			oDstImageSize.height = (int)Math.Max(1.0f, frameHeight);

			//Console.WriteLine("Output Size: " + oDstImageSize.width + "x" + oDstImageSize.height + "x" + (int)(oFrameHeader.nComponents));

			apDstImage[0] = srcY;
			apDstImage[1] = srcCb;
			apDstImage[2] = srcCr;

			int nMCUBlocksH = 0;
			int nMCUBlocksV = 0;

			// Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
			for (int i = 0; i < oFrameHeader.nComponents; ++i)
			{
				nMCUBlocksV = Math.Max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] >> 4);
				nMCUBlocksH = Math.Max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] & 0x0f);
			}

			for (int i = 0; i < oFrameHeader.nComponents; ++i)
			{
				NppiSize oBlocks = new NppiSize();
				NppiSize oBlocksPerMCU = new NppiSize(oFrameHeader.aSamplingFactors[i] & 0x0f, oFrameHeader.aSamplingFactors[i] >> 4);

				oBlocks.width = (int)Math.Ceiling((oFrameHeader.nWidth + 7) / 8 *
										  (float)(oBlocksPerMCU.width) / nMCUBlocksH);
				oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

				oBlocks.height = (int)Math.Ceiling((oFrameHeader.nHeight + 7) / 8 *
										   (float)(oBlocksPerMCU.height) / nMCUBlocksV);
				oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

				// Allocate Memory
				apdDCT[i] = new NPPImage_16sC1(oBlocks.width * 64, oBlocks.height);

			}



			/***************************
			*
			*   Output
			*
			***************************/


			// Forward DCT
			for (int i = 0; i < 3; ++i)
			{
				compression.DCTQuantFwd8x8LS(apDstImage[i], apdDCT[i], aDstSize[i], pdQuantizationTables[oFrameHeader.aQuantizationTableSelector[i]]);
			}


			// Huffman Encoding
			CudaDeviceVariable<byte> pdScan = new CudaDeviceVariable<byte>(BUFFER_SIZE);
			int nScanLength = 0;

			int nTempSize = JPEGCompression.EncodeHuffmanGetSize(aDstSize[0], 3);
			CudaDeviceVariable<byte> pJpegEncoderTemp = new CudaDeviceVariable<byte>(nTempSize);

			NppiEncodeHuffmanSpec[] apHuffmanDCTableEnc = new NppiEncodeHuffmanSpec[3];
			NppiEncodeHuffmanSpec[] apHuffmanACTableEnc = new NppiEncodeHuffmanSpec[3];

			for (int i = 0; i < 3; ++i)
			{
				apHuffmanDCTableEnc[i] = JPEGCompression.EncodeHuffmanSpecInitAlloc(aHuffmanTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, NppiHuffmanTableType.nppiDCTable);
				apHuffmanACTableEnc[i] = JPEGCompression.EncodeHuffmanSpecInitAlloc(aHuffmanTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f) + 2].aCodes, NppiHuffmanTableType.nppiACTable);
			}

			JPEGCompression.EncodeHuffmanScan(apdDCT, 0, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f, pdScan, ref nScanLength, apHuffmanDCTableEnc, apHuffmanACTableEnc, aDstSize, pJpegEncoderTemp);
			
			for (int i = 0; i < 3; ++i)
			{
				JPEGCompression.EncodeHuffmanSpecFree(apHuffmanDCTableEnc[i]);
				JPEGCompression.EncodeHuffmanSpecFree(apHuffmanACTableEnc[i]);
			}

			// Write JPEG to byte array, as in original sample code
			byte[] pDstOutput = new byte[BUFFER_SIZE];
			int pos = 0;

			oFrameHeader.nWidth = (ushort)oDstImageSize.width;
			oFrameHeader.nHeight = (ushort)oDstImageSize.height;

			writeMarker(0x0D8, pDstOutput, ref pos);
			writeJFIFTag(pDstOutput, ref pos);
			writeQuantizationTable(aQuantizationTables[0], pDstOutput, ref pos);
			writeQuantizationTable(aQuantizationTables[1], pDstOutput, ref pos);
			writeFrameHeader(oFrameHeader, pDstOutput, ref pos);
			writeHuffmanTable(aHuffmanTables[0], pDstOutput, ref pos);
			writeHuffmanTable(aHuffmanTables[1], pDstOutput, ref pos);
			writeHuffmanTable(aHuffmanTables[2], pDstOutput, ref pos);
			writeHuffmanTable(aHuffmanTables[3], pDstOutput, ref pos);
			writeScanHeader(oScanHeader, pDstOutput, ref pos);

			pdScan.CopyToHost(pDstOutput, 0, pos, nScanLength);

			pos += nScanLength;
			writeMarker(0x0D9, pDstOutput, ref pos);
			
			FileStream fs = new FileStream(aFilename, FileMode.Create, FileAccess.Write);
			fs.Write(pDstOutput, 0, pos);
			fs.Close();

			//cleanup:
			fs.Dispose();
			pJpegEncoderTemp.Dispose();
			pdScan.Dispose();
			apdDCT[2].Dispose();
			apdDCT[1].Dispose();
			apdDCT[0].Dispose();
			pdQuantizationTables[1].Dispose();
			pdQuantizationTables[0].Dispose();

			srcCr.Dispose();
			srcCb.Dispose();
			srcY.Dispose();
			src.Dispose();
			compression.Dispose();
		}

	}
}
