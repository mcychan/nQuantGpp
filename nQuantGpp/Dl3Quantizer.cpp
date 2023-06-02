/*
 * DL3 Quantization
 * ================
 *
 * Author: Dennis Lee   E-mail: denlee@ecf.utoronto.ca
 *
 * Copyright (C) 1993-1997 Dennis Lee
 * Copyright (c) 2023 Miller Cy Chan
 *
 * C implementation of DL3 Quantization.
 * DL3 Quantization is a 2-pass color quantizer that uses an
 * exhaustive search technique to minimize error introduced at
 * each step during palette reduction.
 *
 * I believe DL3 Quant offers the highest quality of all existing
 * color quantizers.  It is truly 'optimal' except for a few provisos.
 * These provisos and other information about DL3 Quant can be found
 * in DLQUANT.TXT, which is included in this distribution.
 *
 *
 * NOTES
 * =====
 *
 * The dithering code is based on code from the IJG's jpeg library.
 *
 * DL3 Quantization can take a long time to reduce a palette.
 * Times can range from seconds to minutes or even hours depending on
 * the image and the computer used.  This eliminates DL3 Quant for
 * typical usage unless the user has a very fast computer and/or has a
 * lot of patience.  However, the reward is a quantized image that is
 * the best currently possible.  The number of colors in the source image,
 * not the image size, determines the time required to quantize it.
 *
 * This source code may be freely copied, modified, and redistributed,
 * provided this copyright notice is attached.
 * Compiled versions of this code, modified or not, are free for
 * personal use.  Compiled versions used in distributed software
 * is also free, but a notification must be sent to the author.
 * An e-mail to denlee@ecf.utoronto.ca will do.
 *
 */

#include "stdafx.h"
#include "Dl3Quantizer.h"
#include "bitmapUtilities.h"
#include "BlueNoise.h"
#include <unordered_map>

namespace Dl3Quant
{
	uchar alphaThreshold = 0xF;
	bool hasSemiTransparency = false;
	int m_transparentPixelIndex = -1;
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);
	unordered_map<ARGB, vector<ushort> > closestMap;
	unordered_map<ARGB, ushort> nearestMap;

	using namespace std;

	struct CUBE3 {
		int a, r, g, b;
		int aa, rr, gg, bb;
		uint cc;
		uint pixel_count = 0;
		double err;
	};

	void setARGB(CUBE3& rec)
	{
		uint v = rec.pixel_count, v2 = v >> 1;
		rec.aa = (rec.a + v2) / v;
		rec.rr = (rec.r + v2) / v;
		rec.gg = (rec.g + v2) / v;
		rec.bb = (rec.b + v2) / v;
	}

	double calc_err(CUBE3* rgb_table3, const int* squares3, const uint& c1, const uint& c2)
	{
		auto P1 = rgb_table3[c1].pixel_count;
		auto P2 = rgb_table3[c2].pixel_count;
		auto P3 = P1 + P2;

		if (P3 == 0)
			return UINT_MAX;

		int A3 = (rgb_table3[c1].a + rgb_table3[c2].a + (P3 >> 1)) / P3;
		int R3 = (rgb_table3[c1].r + rgb_table3[c2].r + (P3 >> 1)) / P3;
		int G3 = (rgb_table3[c1].g + rgb_table3[c2].g + (P3 >> 1)) / P3;
		int B3 = (rgb_table3[c1].b + rgb_table3[c2].b + (P3 >> 1)) / P3;

		int A1 = rgb_table3[c1].aa;
		int R1 = rgb_table3[c1].rr;
		int G1 = rgb_table3[c1].gg;
		int B1 = rgb_table3[c1].bb;

		int A2 = rgb_table3[c2].aa;
		int R2 = rgb_table3[c2].rr;
		int G2 = rgb_table3[c2].gg;
		int B2 = rgb_table3[c2].bb;

		auto dist1 = squares3[A3 - A1] + squares3[R3 - R1] + squares3[G3 - G1] + squares3[B3 - B1];
		dist1 *= P1;

		auto dist2 = squares3[A2 - A3] + squares3[R2 - R3] + squares3[G2 - G3] + squares3[B2 - B3];
		dist2 *= P2;

		return (dist1 + dist2);
	}

	void build_table3(CUBE3* rgb_table3, const Vec4b& pixel)
	{
		int index = GetArgbIndex(pixel, hasSemiTransparency, m_transparentPixelIndex >= 0);

		rgb_table3[index].a += pixel[3];
		rgb_table3[index].r += pixel[2];
		rgb_table3[index].g += pixel[1];
		rgb_table3[index].b += pixel[0];
		rgb_table3[index].pixel_count++;
	}

	uint build_table3(CUBE3* rgb_table3, const Mat pixels)
	{
		for (uint y = 0; y < pixels.rows; ++y)
		{
			for (uint x = 0; x < pixels.cols; ++x) {
				Vec4b pixel;
				GrabPixel(pixel, pixels, y, x);
				build_table3(rgb_table3, pixel);
			}
		}

		uint tot_colors = 0;
		for (int i = 0; i < USHRT_MAX + 1; ++i) {
			if (rgb_table3[i].pixel_count > 0) {
				setARGB(rgb_table3[i]);
				rgb_table3[tot_colors] = rgb_table3[i];
				++tot_colors;
			}
		}
		return tot_colors;
	}

	void recount_next(CUBE3* rgb_table3, const int* squares3, const uint& tot_colors, const uint& i)
	{
		uint c2 = 0;
		double err = UINT_MAX;
		for (uint j = i + 1; j < tot_colors; ++j) {
			auto cur_err = calc_err(rgb_table3, squares3, i, j);
			if (cur_err < err) {
				err = cur_err;
				c2 = j;
			}
		}
		rgb_table3[i].err = err;
		rgb_table3[i].cc = c2;
	}

	void recount_dist(CUBE3* rgb_table3, const int* squares3, const uint& tot_colors, const uint& c1)
	{
		recount_next(rgb_table3, squares3, tot_colors, c1);
		for (int i = 0; i < c1; ++i) {
			if (rgb_table3[i].cc == c1)
				recount_next(rgb_table3, squares3, tot_colors, i);
			else {
				auto cur_err = calc_err(rgb_table3, squares3, i, c1);
				if (cur_err < rgb_table3[i].err) {
					rgb_table3[i].err = cur_err;
					rgb_table3[i].cc = c1;
				}
			}
		}
	}

	void reduce_table3(CUBE3* rgb_table3, const int* squares3, uint tot_colors, const uint& num_colors)
	{
		uint i = 0;
		for (; i < (tot_colors - 1); ++i)
			recount_next(rgb_table3, squares3, tot_colors, i);

		rgb_table3[i].err = UINT_MAX;
		rgb_table3[i].cc = tot_colors;

		uint c1 = 0, grand_total = tot_colors - num_colors;
		while (tot_colors > num_colors) {
			uint err = UINT_MAX;
			for (i = 0; i < tot_colors; ++i) {
				if (rgb_table3[i].err < err) {
					err = rgb_table3[i].err;
					c1 = i;
				}
			}
			auto c2 = rgb_table3[c1].cc;
			rgb_table3[c2].a += rgb_table3[c1].a;
			rgb_table3[c2].r += rgb_table3[c1].r;
			rgb_table3[c2].g += rgb_table3[c1].g;
			rgb_table3[c2].b += rgb_table3[c1].b;
			rgb_table3[c2].pixel_count += rgb_table3[c1].pixel_count;
			setARGB(rgb_table3[c2]);

			rgb_table3[c1] = rgb_table3[--tot_colors];
			rgb_table3[tot_colors - 1].err = UINT_MAX;
			rgb_table3[tot_colors - 1].cc = tot_colors;

			for (i = 0; i < c1; ++i) {
				if (rgb_table3[i].cc == tot_colors)
					rgb_table3[i].cc = c1;
			}

			for (i = c1 + 1; i < tot_colors; ++i) {
				if (rgb_table3[i].cc == tot_colors)
					recount_next(rgb_table3, squares3, tot_colors, i);
			}

			recount_dist(rgb_table3, squares3, tot_colors, c1);
			if (c2 != tot_colors)
				recount_dist(rgb_table3, squares3, tot_colors, c2);
		}
	}

	ushort nearestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
	{
		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			c = m_transparentColor;
		
		const auto nMaxColors = palette.rows;
		if (nMaxColors > 2 && m_transparentPixelIndex >= 0 && c[3] > alphaThreshold)
			k = 1;

		auto argb = GetArgb8888(c0);
		auto got = nearestMap.find(argb);
		if (got == nearestMap.end()) {
			double mindist = SHRT_MAX;
			for (uint i = k; i < palette.rows; i++) {
				Vec4b c2;
				GrabPixel(c2, palette, i, 0);
				double curdist = sqr(c2[3] - c[3]);
				if (curdist > mindist)
					continue;

				curdist += sqr(c2[2] - c[2]);
				if (curdist > mindist)
					continue;

				curdist += sqr(c2[1] - c[1]);
				if (curdist > mindist)
					continue;

				curdist += sqr(c2[0] - c[0]);
				if (curdist > mindist)
					continue;

				mindist = curdist;
				k = i;
			}

			nearestMap[argb] = k;
		}
		else
			k = got->second;

		return k;
	}

	ushort closestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
	{
		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			c = m_transparentColor;

		const auto nMaxColors = palette.rows;
		auto argb = GetArgb8888(c0);

		vector<ushort> closest(5);
		auto got = closestMap.find(argb);
		if (got == closestMap.end()) {
			closest[2] = closest[3] = SHRT_MAX;

			for (; k < nMaxColors; ++k) {
				Vec4b c2;
				GrabPixel(c2, palette, k, 0);
				closest[4] = abs(c[3] - c2[3]) + abs(c[2] - c2[2]) + abs(c[1] - c2[1]) + abs(c[0] - c2[0]);
				if (closest[4] < closest[2]) {
					closest[1] = closest[0];
					closest[3] = closest[2];
					closest[0] = k;
					closest[2] = closest[4];
				}
				else if (closest[4] < closest[3]) {
					closest[1] = k;
					closest[3] = closest[4];
				}
			}

			if (closest[3] == SHRT_MAX)
				closest[2] = 0;
		}
		else
			closest = got->second;

		if (closest[2] == 0 || (rand() % (closest[3] + closest[2])) <= closest[3])
			k = closest[0];
		else
			k = closest[1];

		closestMap[argb] = closest;
		return k;
	}

	inline auto GetColorIndex(const Vec4b& c)
	{
		return GetArgbIndex(c, hasSemiTransparency, m_transparentPixelIndex >= 0);
	}

	bool quantize_image(const Mat4b pixels, const Mat palette, const uint nMaxColors, Mat1b qPixels, const bool dither)
	{
		auto width = pixels.cols;
		auto height = pixels.rows;
		if (dither)
			return dither_image(pixels, palette, nearestColorIndex, hasSemiTransparency, m_transparentPixelIndex, nMaxColors, qPixels);

		DitherFn ditherFn = (m_transparentPixelIndex >= 0 || nMaxColors < 256) ? nearestColorIndex : closestColorIndex;
		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				auto& pixel = pixels(j, i);
				qPixels(j, i) = (uchar) closestColorIndex(palette, pixel, i + j);
			}
		}

		BlueNoise::dither(pixels, palette, ditherFn, GetColorIndex, qPixels);
		return true;
	}

	void GetQuantizedPalette(Mat palette, const CUBE3* rgb_table3)
	{
		for (uint k = 0; k < palette.rows; ++k) {
			uint sum = rgb_table3[k].pixel_count;
			if (sum > 0) {
				if(palette.channels() == 4)
					palette.at<Vec4b>(k, 0) = Vec4b(rgb_table3[k].bb, rgb_table3[k].gg, rgb_table3[k].rr, rgb_table3[k].aa);
				else
					palette.at<Vec3b>(k, 0) = Vec3b(rgb_table3[k].bb, rgb_table3[k].gg, rgb_table3[k].rr);

				if (m_transparentPixelIndex >= 0 && GetArgb8888(palette.at<Vec4b>(k, 0)) == GetArgb8888(m_transparentColor))
					swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(k, 0));
			}
		}
	}

	Mat Dl3Quantizer::QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);

		Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
		GrabPixels(srcImg, pixels4b, hasSemiTransparency, m_transparentPixelIndex, m_transparentColor, nMaxColors);

		Mat palette(nMaxColors, 1, srcImg.type(), scalar);

		if (nMaxColors > 2) {
			auto rgb_table3 = make_unique<CUBE3[]>(USHRT_MAX + 1);
			auto tot_colors = build_table3(rgb_table3.get(), pixels4b);
			int sqr_tbl[UCHAR_MAX + UCHAR_MAX + 1];

			for (int i = (-UCHAR_MAX); i <= UCHAR_MAX; ++i)
				sqr_tbl[i + UCHAR_MAX] = i * i;

			auto squares3 = &sqr_tbl[UCHAR_MAX];

			reduce_table3(rgb_table3.get(), squares3, tot_colors, nMaxColors);

			GetQuantizedPalette(palette, rgb_table3.get());
		}
		else {
			if (m_transparentPixelIndex >= 0)
				palette.at<Vec4b>(0, 0) = m_transparentColor;
			else
				palette.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);
		}

		if (nMaxColors > 256) {
			Mat qPixels(bitmapHeight, bitmapWidth, srcImg.type());
			dithering_image(pixels4b, palette, nearestColorIndex, hasSemiTransparency, m_transparentPixelIndex, nMaxColors, qPixels);
			closestMap.clear();
			nearestMap.clear();
			return qPixels;
		}

		Mat1b qPixels(bitmapHeight, bitmapWidth);
		quantize_image(pixels4b, palette, nMaxColors, qPixels, dither);
		closestMap.clear();
		nearestMap.clear();

		if (m_transparentPixelIndex >= 0) {
			auto k = qPixels(m_transparentPixelIndex / bitmapWidth, m_transparentPixelIndex % bitmapWidth);
			if (nMaxColors > 2)
				palette.at<Vec4b>(k, 0) = m_transparentColor;
			else if (GetArgb8888(palette.at<Vec4b>(k, 0)) != GetArgb8888(m_transparentColor))
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(1, 0));
		}
		ProcessImagePixels(bytes, palette, qPixels, m_transparentPixelIndex >= 0);
		return palette;
	}

}
