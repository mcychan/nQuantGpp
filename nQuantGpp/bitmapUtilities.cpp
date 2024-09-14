#include "stdafx.h"
#include "bitmapUtilities.h"
#include "ApngWriter.h"

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


void ProcessImagePixels(vector<uchar>& bytes, const Mat palette, const Mat1b qPixels, const bool& hasTransparent)
{
	PngEncode::AddImage(bytes, palette, qPixels, hasTransparent);
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
