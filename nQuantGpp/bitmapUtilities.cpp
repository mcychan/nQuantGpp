#include "stdafx.h"
#include "bitmapUtilities.h"

void CalcDitherPixel(int* pDitherPixel, const Vec4b& c, const uchar* clamp, const short* rowerr, int cursor, const bool noBias)
{
	if (noBias) {
		pDitherPixel[0] = clamp[((rowerr[cursor] + 0x1008) >> 4) + c[0]];
		pDitherPixel[1] = clamp[((rowerr[cursor + 1] + 0x1008) >> 4) + c[1]];
		pDitherPixel[2] = clamp[((rowerr[cursor + 2] + 0x1008) >> 4) + c[2]];
		pDitherPixel[3] = clamp[((rowerr[cursor + 3] + 0x1008) >> 4) + c[3]];
	}
	else {
		pDitherPixel[0] = clamp[((rowerr[cursor] + 0x2010) >> 5) + c[0]];
		pDitherPixel[1] = clamp[((rowerr[cursor + 1] + 0x1008) >> 4) + c[1]];
		pDitherPixel[2] = clamp[((rowerr[cursor + 2] + 0x2010) >> 5) + c[2]];
		pDitherPixel[3] = c[3];
	}
}

bool dither_image(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, const bool& hasSemiTransparency, const int& transparentPixelIndex, const uint nMaxColors, Mat1b qPixels)
{
	uint pixelIndex = 0;
	auto width = pixels4b.cols;
	auto height = pixels4b.rows;

	const int DJ = 4;
	const int BLOCK_SIZE = 256;
	const int DITHER_MAX = 20;
	const int err_len = (width + 2) * DJ;
	auto clamp = make_unique <uchar[]>(DJ * BLOCK_SIZE);
	auto erowErr = make_unique<short[]>(err_len);
	auto orowErr = make_unique<short[]>(err_len);
	auto limtb = make_unique<char[]>(2 * BLOCK_SIZE);
	auto lookup = make_unique<short[]>(65536);
	auto pDitherPixel = make_unique<int[]>(DJ);

	for (int i = 0; i < BLOCK_SIZE; ++i) {
		clamp[i] = 0;
		clamp[i + BLOCK_SIZE] = static_cast<uchar>(i);
		clamp[i + BLOCK_SIZE * 2] = UCHAR_MAX;
		clamp[i + BLOCK_SIZE * 3] = UCHAR_MAX;

		limtb[i] = -DITHER_MAX;
		limtb[i + BLOCK_SIZE] = DITHER_MAX;
	}
	for (int i = -DITHER_MAX; i <= DITHER_MAX; ++i) {
		limtb[i + BLOCK_SIZE] = i;
		if (nMaxColors > 16 && i % 4 == 3)
			limtb[i + BLOCK_SIZE] = 0;
	}

	auto row0 = erowErr.get();
	auto row1 = orowErr.get();

	bool noBias = (transparentPixelIndex >= 0 || hasSemiTransparency) || nMaxColors < 64;
	int dir = 1;
	for (int i = 0; i < height; ++i) {
		if (dir < 0)
			pixelIndex += width - 1;

		int cursor0 = DJ, cursor1 = width * DJ;
		row1[cursor1] = row1[cursor1 + 1] = row1[cursor1 + 2] = row1[cursor1 + 3] = 0;
		for (int j = 0; j < width; ++j) {
			int y = pixelIndex / width, x = pixelIndex % width;
			auto& pixel = pixels4b(y, x);

			CalcDitherPixel(pDitherPixel.get(), pixel, clamp.get(), row0, cursor0, noBias);
			int b_pix = pDitherPixel[0];
			int g_pix = pDitherPixel[1];
			int r_pix = pDitherPixel[2];
			int a_pix = pDitherPixel[3];
			Vec4b c1(b_pix, g_pix, r_pix, a_pix);
			auto& qPixel = qPixels(y, x);
			if (noBias && a_pix > 0xF0) {
				int offset = GetArgbIndex(c1, hasSemiTransparency, transparentPixelIndex >= 0);
				if (!lookup[offset])
					lookup[offset] = ditherFn(palette, c1, i + j) + 1;
				qPixel = (uchar) lookup[offset] - 1;
			}
			else
				qPixel = (uchar) ditherFn(palette, c1, i + j);

			Vec4b c2;
			GrabPixel(c2, palette, qPixel, 0);

			b_pix = limtb[c1[0] - c2[0] + BLOCK_SIZE];
			g_pix = limtb[c1[1] - c2[1] + BLOCK_SIZE];
			r_pix = limtb[c1[2] - c2[2] + BLOCK_SIZE];
			a_pix = limtb[c1[3] - c2[3] + BLOCK_SIZE];

			int k = r_pix * 2;
			row1[cursor1 - DJ] = r_pix;
			row1[cursor1 + DJ] += (r_pix += k);
			row1[cursor1] += (r_pix += k);
			row0[cursor0 + DJ] += (r_pix + k);

			k = g_pix * 2;
			row1[cursor1 + 1 - DJ] = g_pix;
			row1[cursor1 + 1 + DJ] += (g_pix += k);
			row1[cursor1 + 1] += (g_pix += k);
			row0[cursor0 + 1 + DJ] += (g_pix + k);

			k = b_pix * 2;
			row1[cursor1 + 2 - DJ] = b_pix;
			row1[cursor1 + 2 + DJ] += (b_pix += k);
			row1[cursor1 + 2] += (b_pix += k);
			row0[cursor0 + 2 + DJ] += (b_pix + k);

			k = a_pix * 2;
			row1[cursor1 + 3 - DJ] = a_pix;
			row1[cursor1 + 3 + DJ] += (a_pix += k);
			row1[cursor1 + 3] += (a_pix += k);
			row0[cursor0 + 3 + DJ] += (a_pix + k);

			cursor0 += DJ;
			cursor1 -= DJ;
			pixelIndex += dir;
		}
		if ((i % 2) == 1)
			pixelIndex += width + 1;

		dir *= -1;
		swap(row0, row1);
	}
	return true;
}

bool dithering_image(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, const bool& hasSemiTransparency, const int& transparentPixelIndex, const uint nMaxColors, Mat qPixels)
{
	uint pixelIndex = 0;
	auto width = pixels4b.cols;
	auto height = pixels4b.rows;
	
	bool hasTransparency = (transparentPixelIndex >= 0 || hasSemiTransparency);
	const int DJ = 4;
	const int BLOCK_SIZE = 256;
	const int DITHER_MAX = 20;
	const int err_len = (width + 2) * DJ;
	auto clamp = make_unique<uchar[]>(DJ * BLOCK_SIZE);
	auto erowErr = make_unique<short[]>(err_len);
	auto orowErr = make_unique<short[]>(err_len);
	auto limtb = make_unique<char[]>(2 * BLOCK_SIZE);
	auto lookup = make_unique<short[]>(65536);
	auto pDitherPixel = make_unique<int[]>(DJ);

	for (int i = 0; i < BLOCK_SIZE; ++i) {
		clamp[i] = 0;
		clamp[i + BLOCK_SIZE] = static_cast<uchar>(i);
		clamp[i + BLOCK_SIZE * 2] = UCHAR_MAX;
		clamp[i + BLOCK_SIZE * 3] = UCHAR_MAX;

		limtb[i] = -DITHER_MAX;
		limtb[i + BLOCK_SIZE] = DITHER_MAX;
	}
	for (int i = -DITHER_MAX; i <= DITHER_MAX; ++i) {
		limtb[i + BLOCK_SIZE] = i;
		if(nMaxColors > 16 && i % 4 == 3)
			limtb[i + BLOCK_SIZE] = 0;
	}

	auto row0 = erowErr.get();
	auto row1 = orowErr.get();
	int dir = 1;
	for (int i = 0; i < height; ++i) {
		if (dir < 0)
			pixelIndex += width - 1;

		int cursor0 = DJ, cursor1 = width * DJ;
		row1[cursor1] = row1[cursor1 + 1] = row1[cursor1 + 2] = row1[cursor1 + 3] = 0;
		for (int j = 0; j < width; ++j) {
			int y = pixelIndex / width, x = pixelIndex % width;
			auto& pixel = pixels4b(y, x);

			CalcDitherPixel(pDitherPixel.get(), pixel, clamp.get(), row0, cursor0, hasTransparency);
			int b_pix = pDitherPixel[0];
			int g_pix = pDitherPixel[1];
			int r_pix = pDitherPixel[2];
			int a_pix = pDitherPixel[3];
			Vec4b c1(b_pix, g_pix, r_pix, a_pix);
			auto qPixelIndex = ditherFn(palette, c1, i + j);

			Vec4b c2;
			GrabPixel(c2, palette, qPixelIndex, 0);
			SetPixel(qPixels, y, x, c2);

			b_pix = limtb[BLOCK_SIZE + c1[0] - c2[0]];
			g_pix = limtb[BLOCK_SIZE + c1[1] - c2[1]];
			r_pix = limtb[BLOCK_SIZE + c1[2] - c2[2]];
			a_pix = limtb[BLOCK_SIZE + c1[3] - c2[3]];

			int k = r_pix * 2;
			row1[cursor1 - DJ] = r_pix;
			row1[cursor1 + DJ] += (r_pix += k);
			row1[cursor1] += (r_pix += k);
			row0[cursor0 + DJ] += (r_pix + k);

			k = g_pix * 2;
			row1[cursor1 + 1 - DJ] = g_pix;
			row1[cursor1 + 1 + DJ] += (g_pix += k);
			row1[cursor1 + 1] += (g_pix += k);
			row0[cursor0 + 1 + DJ] += (g_pix + k);

			k = b_pix * 2;
			row1[cursor1 + 2 - DJ] = b_pix;
			row1[cursor1 + 2 + DJ] += (b_pix += k);
			row1[cursor1 + 2] += (b_pix += k);
			row0[cursor0 + 2 + DJ] += (b_pix + k);

			k = a_pix * 2;
			row1[cursor1 + 3 - DJ] = a_pix;
			row1[cursor1 + 3 + DJ] += (a_pix += k);
			row1[cursor1 + 3] += (a_pix += k);
			row0[cursor0 + 3 + DJ] += (a_pix + k);

			cursor0 += DJ;
			cursor1 -= DJ;
			pixelIndex += dir;
		}
		if ((i % 2) == 1)
			pixelIndex += (width + 1);

		dir *= -1;
		swap(row0, row1);
	}
	return true;
}


// ---------------------------------------------------------------------------------------------------------
// NOTE copy from: **OFFICIAL** png documentation has a link to "sample crc code"
// https://www.w3.org/TR/PNG-CRCAppendix.html
// ---------------------------------------------------------------------------------------------------------

/* Table of CRCs of all 8-bit messages. */
unsigned long crc_table[256];

/* Flag: has the table been computed? Initially false. */
int crc_table_computed = 0;

/* Make the table for a fast CRC. */
void make_crc_table() {
	unsigned long c;
	int n, k;

	for (n = 0; n < 256; ++n) {
		c = (unsigned long)n;
		for (k = 0; k < 8; k++) {
			if (c & 1)
				c = 0xedb88320L ^ (c >> 1);
			else
				c >>= 1;
		}
		crc_table[n] = c;
	}
	crc_table_computed = 1;
}

/* Update a running CRC with the bytes buf[0..len-1]--the CRC
   should be initialized to all 1's, and the transmitted value
   is the 1's complement of the final running CRC (see the
   crc() routine below)). */

unsigned long update_crc(unsigned long crc, unsigned char *buf, int len) {
	unsigned long c = crc;
	int n;

	if (!crc_table_computed)
		make_crc_table();
	for (n = 0; n < len; n++) {
		c = crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
	}
	return c;
}

/* Return the CRC of the bytes buf[0..len-1]. */
unsigned long crc(unsigned char* buf, int len) {
	return update_crc(0xffffffffL, buf, len) ^ 0xffffffffL;
}

const uchar PNG_SIGNATURE[] = {137, 80, 78, 71, 13, 10, 26, 10};

// NOTE png needs big endian https://stackoverflow.com/questions/2384111/png-file-format-endianness
uint32_t to_uint32_big_endian(const uchar* a) {
	return a[3] + (((uint32_t) a[2]) << 8) + (((uint32_t) a[1]) << 16) + (((uint32_t) a[0]) << 24);
}

void from_uint32_big_endian(uchar *a, uint32_t n) {
	a[0] = (n >> 24) & 0xFF;
	a[1] = (n >> 16) & 0xFF;
	a[2] = (n >> 8) & 0xFF;
	a[3] = n & 0xFF;
}

int png_iterator_first_chunk(const vector<uchar>& bytes) {
	uint n = sizeof(PNG_SIGNATURE) / sizeof(PNG_SIGNATURE[0]);
	for (int j = 0; j < n; ++j) {
		CV_Assert(bytes[j] == PNG_SIGNATURE[j]);
	}
	return n;
}

int png_iterator_next_chunk(const vector<uchar>& bytes, int chunk_begin_loc, string& chunk_type) {
	int idx = chunk_begin_loc;

	uint32_t length = to_uint32_big_endian(&bytes[idx]);
	idx += 4;

	string type(bytes.begin() + idx, bytes.begin() + idx + 4);
	idx += 4;

	chunk_type = type;
	return chunk_begin_loc + (int) length + 12;
}

void create_PLTE_chunk(vector<uchar>& bytes, const Mat palette) {
	CV_Assert(palette.cols == 1);
	CV_Assert(palette.rows <= 256); // since only 8-bit depth!

	// *3: since 3 bytes per channel (color image)
	uint32_t length = palette.rows * 3;
	if(palette.channels() < 4)
		CV_Assert(palette.type() == CV_8UC3);
	else
		CV_Assert(palette.type() == CV_8UC4);

	// +12: by definition of chunk "actual" length vs "recorded" length
	bytes.resize(length + 12);

	int idx = 0;

	from_uint32_big_endian(&bytes[idx], length);
	idx += 4;

	bytes[idx + 0] = 'P';
	bytes[idx + 1] = 'L';
	bytes[idx + 2] = 'T';
	bytes[idx + 3] = 'E';
	idx += 4;

	// I know not that fast, but palette is pretty small so readability is also important
	for (int j = 0; j < palette.rows; ++j) {
		for (int ch = 0; ch < 3; ++ch) {
			if(palette.channels() < 4)
				bytes[idx + ch] = palette.at<Vec3b>(j, 0)[2 - ch];
			else
				bytes[idx + ch] = palette.at<Vec4b>(j, 0)[2 - ch];
		}
		idx += 3;
	}

	auto crc_val = crc(&bytes[4], (int) length + 4);
	from_uint32_big_endian(&bytes[idx], crc_val);
	idx += 4;

	CV_Assert(idx == bytes.size());
}

bool create_tRNS_chunk(vector<uchar>& bytes, const Mat palette) {
	if(palette.channels() < 4)
		return false;
	
	CV_Assert(palette.cols == 1);
	CV_Assert(palette.rows <= 256); // since only 8-bit depth!

	uint32_t length = palette.rows;
	CV_Assert(palette.type() == CV_8UC4);

	// +12: by definition of chunk "actual" length vs "recorded" length
	bytes.resize(length + 12);

	int idx = 0;

	from_uint32_big_endian(&bytes[idx], length);
	idx += 4;

	bytes[idx + 0] = 't';
	bytes[idx + 1] = 'R';
	bytes[idx + 2] = 'N';
	bytes[idx + 3] = 'S';
	idx += 4;

	for (int j = 0; j < palette.rows; ++j)
		bytes[idx++] = palette.at<Vec4b>(j, 0)[3];

	auto crc_val = crc(&bytes[4], (int) length + 4);
	from_uint32_big_endian(&bytes[idx], crc_val);
	idx += 4;

	CV_Assert(idx == bytes.size());
	return true;
}

void change_IHDR_colortype_and_crc(vector<uchar>& bytes, int ihdr_start_loc, int ihdr_end_loc, const int nMaxColors) {
	const int ihdr_data_loc = ihdr_start_loc + 4 + 4;
	const int ihdr_bitdepth_loc = ihdr_data_loc + 8;
	const int ihdr_colortype_loc = ihdr_data_loc + 9;
	const int crc_loc = ihdr_end_loc - 4;

	if (nMaxColors < 3)
		CV_Assert(bytes[ihdr_bitdepth_loc] == 1);
	else
		CV_Assert(bytes[ihdr_bitdepth_loc] == 8);

	// a. change colortype to "indexed color"
	bytes[ihdr_colortype_loc] = 3;

	// b. re-calculate the CRC value
	int chunk_length = ihdr_end_loc - ihdr_start_loc - 12;
	auto crc_val = crc(&bytes[ihdr_start_loc + 4], chunk_length + 4);
	from_uint32_big_endian(&bytes[crc_loc], crc_val);
}

void ProcessImagePixels(vector<uchar>& bytes, const Mat palette, const Mat1b qPixels, const bool& hasTransparent)
{
	CV_Assert(qPixels.type() == CV_8UC1);
	CV_Assert(palette.cols == 1);
	if(hasTransparent)
		CV_Assert(palette.type() == CV_8UC4);
	else
		CV_Assert(palette.type() == CV_8UC3);

	imencode(".png", qPixels, bytes,
		// use higher compression, thus smaller
		 {IMWRITE_PNG_COMPRESSION, 9, IMWRITE_PNG_BILEVEL, palette.rows < 3});

	int idx = png_iterator_first_chunk(bytes);
	for (;;) {
		string chunk_type;
		int next_chunk_start_idx = png_iterator_next_chunk(bytes, idx, chunk_type);

		if (chunk_type == "IHDR") {
			// change 1: change THDR "color" flag
			change_IHDR_colortype_and_crc(bytes, idx, next_chunk_start_idx, palette.rows);

			// change 2: insert the PLTE chunk **after** THDR chunk
			vector<uchar> plte_chunk_bytes;
			create_PLTE_chunk(plte_chunk_bytes, palette);
			bytes.insert(bytes.begin() + next_chunk_start_idx, plte_chunk_bytes.begin(), plte_chunk_bytes.end());

			// no need to manipulate the data after that. let's stop!
			if (palette.channels() > 3) {
				vector<uchar> trns_chunk_bytes;
				if (create_tRNS_chunk(trns_chunk_bytes, palette))
					bytes.insert(bytes.begin() + next_chunk_start_idx, trns_chunk_bytes.begin(), trns_chunk_bytes.end());
			}
			break;
		}

		if (next_chunk_start_idx >= bytes.size())
			break;

		idx = next_chunk_start_idx;
	}
}

void SetPixel(Mat pixels, int row, int col, Vec4b& pixel)
{
	if(pixels.channels() == 4)
		pixels.at<Vec4b>(row, col) = pixel;
	else
		pixels.at<Vec3b>(row, col) = Vec3b(pixel[0], pixel[1], pixel[2]);
}

bool GrabPixels(const Mat source, Mat4b pixels, int& semiTransCount, int& transparentPixelIndex, Vec4b& transparentColor, const uchar alphaThreshold, const uint nMaxColors)
{
	bool hasAlpha = source.channels() > 3;
	int pixelIndex = 0;
	for (int y = 0; y < source.rows; ++y)
	{
		for (int x = 0; x < source.cols; ++x)
		{
			Vec4b pixel;
			GrabPixel(pixel, source, y, x);
			auto pixelBlue = pixel[0];
			auto pixelGreen = pixel[1];
			auto pixelRed = pixel[2];
			auto pixelAlpha = hasAlpha ? pixel[3] : UCHAR_MAX;
			if (transparentPixelIndex > -1 && GetArgb8888(transparentColor) == GetArgb8888(pixel)) {
				pixelAlpha = 0;
				pixel = Vec4b(pixelBlue, pixelGreen, pixelRed, pixelAlpha);
			}

			if (pixelAlpha < 0xE0) {
				if (pixelAlpha == 0) {
					transparentPixelIndex = pixelIndex;
					if (nMaxColors > 2)
						transparentColor = pixel;
					else
						pixel = transparentColor;
				}
				else if (pixelAlpha > alphaThreshold)
					++semiTransCount;
			}
			pixels(y, x) = pixel;
			++pixelIndex;
		}
	}

	return !source.empty();
}

int GrabPixels(const Mat source, Mat4b pixels, bool& hasSemiTransparency, int& transparentPixelIndex, Vec4b& transparentColor, const uchar alphaThreshold, const uint nMaxColors)
{
	int semiTransCount = 0;
	GrabPixels(source, pixels, semiTransCount, transparentPixelIndex, transparentColor, alphaThreshold, nMaxColors);
	hasSemiTransparency = semiTransCount > 0;
	return semiTransCount;
}

void GrabPixel(Vec4b& pixel, const Mat pixels, int row, int col)
{
	if (pixels.channels() == 4)
		pixel = pixels.at<Vec4b>(row, col);
	else {
		auto& bgr = pixels.at<Vec3b>(row, col);
		pixel = Vec4b(bgr[0], bgr[1], bgr[2], UCHAR_MAX);
	}
}
