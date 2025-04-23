/* Fast pairwise nearest neighbor based algorithm for multilevel thresholding
Copyright (C) 2004-2016 Mark Tyler and Dmitry Groshev
Copyright (c) 2018-2024 Miller Cy Chan
* error measure; time used is proportional to number of bins squared - WJ */

#include "stdafx.h"
#include "PnnQuantizer.h"
#include "bitmapUtilities.h"
#include "GilbertCurve.h"
#include "BlueNoise.h"
#include <unordered_map>

namespace PnnQuant
{
	uchar alphaThreshold = 0xF;
	bool hasSemiTransparency = false;
	int m_transparentPixelIndex = -1;
	double ratio = .5, weight = 1.0;
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);
	double PR = .299, PG = .587, PB = .114, PA = .3333;
	unordered_map<ARGB, vector<ushort> > closestMap;
	unordered_map<ARGB, ushort > nearestMap;

	static const float coeffs[3][3] = {
		{0.299f, 0.587f, 0.114f},
		{-0.14713f, -0.28886f, 0.436f},
		{0.615f, -0.51499f, -0.10001f}
	};

	struct pnnbin {
		float ac = 0, rc = 0, gc = 0, bc = 0, err = 0;
		float cnt = 0;
		int nn = 0, fw = 0, bk = 0, tm = 0, mtm = 0;
	};

	void find_nn(pnnbin* bins, int idx)
	{
		int nn = 0;
		float err = 1e100;

		auto& bin1 = bins[idx];
		auto n1 = bin1.cnt;
		auto wa = bin1.ac;
		auto wr = bin1.rc;
		auto wg = bin1.gc;
		auto wb = bin1.bc;

		int start = 0;
		if (BlueNoise::TELL_BLUE_NOISE[idx & 4095] > -88)
			start = (PG < coeffs[0][1]) ? 3 : 1;

		for (int i = bin1.fw; i; i = bins[i].fw) {
			auto n2 = bins[i].cnt, nerr2 = (n1 * n2) / (n1 + n2);
			if (nerr2 >= err)
				continue;

			auto nerr = 0.0;
			if (hasSemiTransparency) {
				nerr += nerr2 * PA * sqr(bins[i].ac - wa);
				if (nerr >= err)
					continue;
			}

			nerr += nerr2 * (1 - ratio) * PR * sqr(bins[i].rc - wr);
			if (nerr >= err)
				continue;

			nerr += nerr2 * (1 - ratio) * PG * sqr(bins[i].gc - wg);
			if (nerr >= err)
				continue;

			nerr += nerr2 * (1 - ratio) * PB * sqr(bins[i].bc - wb);
			if (nerr >= err)
				continue;

			for (int j = start; j < 3; ++j) {
				nerr += nerr2 * ratio * sqr(coeffs[j][0] * (bins[i].rc - wr));
				if (nerr >= err)
					break;

				nerr += nerr2 * ratio * sqr(coeffs[j][1] * (bins[i].gc - wg));
				if (nerr >= err)
					break;

				nerr += nerr2 * ratio * sqr(coeffs[j][2] * (bins[i].bc - wb));
				if (nerr >= err)
					break;
			}

			if (nerr >= err)
				continue;
			err = nerr;
			nn = i;
		}
		bin1.err = err;
		bin1.nn = nn;
	}

	typedef float (*QuanFn)(const float& cnt);
	QuanFn getQuanFn(const uint& nMaxColors, const short quan_rt) {
		if (quan_rt > 0) {
			if (nMaxColors < 64)
				return[](const float& cnt) {
					return (float)(int) sqrt(cnt);
				};
			return[](const float& cnt) {
				return (float) sqrt(cnt);
			};
		}
		if (quan_rt < 0)
			return[](const float& cnt) { return (float)(int)cbrt(cnt); };
		return[](const float& cnt) { return cnt; };
	}

	void pnnquan(const Mat pixels, Mat palette, uint& nMaxColors)
	{
		short quan_rt = 1;
		vector<pnnbin> bins(USHRT_MAX + 1);

		/* Build histogram */
		for (int y = 0; y < pixels.rows; ++y)
		{
			for (int x = 0; x < pixels.cols; ++x)
			{
				Vec4b c;
				GrabPixel(c, pixels, y, x);
				if (c[3] <= alphaThreshold)
					c = m_transparentColor;

				int index = GetArgbIndex(c, hasSemiTransparency, nMaxColors < 64 || m_transparentPixelIndex >= 0);
				auto& tb = bins[index];
				tb.ac += c[3];
				tb.rc += c[2];
				tb.gc += c[1];
				tb.bc += c[0];
				tb.cnt += 1.0;
			}
		}

		/* Cluster nonempty bins at one end of array */
		int maxbins = 0;

		for (int i = 0; i < bins.size(); ++i) {
			if (bins[i].cnt <= 0)
				continue;

			auto& tb = bins[i];
			double d = 1.0 / tb.cnt;
			tb.ac *= d;
			tb.rc *= d;
			tb.gc *= d;
			tb.bc *= d;
			
			bins[maxbins++] = tb;
		}

		if (nMaxColors < 16)
			quan_rt = -1;

		weight = min(0.9, nMaxColors * 1.0 / maxbins);
		if (weight < .03 && PG < 1 && PG >= coeffs[0][1]) {
			PR = PG = PB = PA = 1;
			if (nMaxColors >= 64)
				quan_rt = 0;
		}

		auto quanFn = getQuanFn(nMaxColors, quan_rt);

		int j = 0;
		for (; j < maxbins - 1; ++j) {
			bins[j].fw = j + 1;
			bins[j + 1].bk = j;

			bins[j].cnt = quanFn(bins[j].cnt);
		}
		bins[j].cnt = quanFn(bins[j].cnt);

		auto heap = make_unique<int[]>(bins.size() + 1);
		int h, l, l2;
		/* Initialize nearest neighbors and build heap of them */
		for (int i = 0; i < maxbins; ++i) {
			find_nn(bins.data(), i);
			/* Push slot on heap */
			auto err = bins[i].err;
			for (l = ++heap[0]; l > 1; l = l2) {
				l2 = l >> 1;
				if (bins[h = heap[l2]].err <= err)
					break;
				heap[l] = h;
			}
			heap[l] = i;
		}

		/* Merge bins which increase error the least */
		int extbins = maxbins - nMaxColors;
		for (int i = 0; i < extbins; ) {
			int b1;
			
			/* Use heap to find which bins to merge */
			for (;;) {
				auto& tb = bins[b1 = heap[1]]; /* One with least error */
				/* Is stored error up to date? */
				if ((tb.tm >= tb.mtm) && (bins[tb.nn].mtm <= tb.tm))
					break;
				if (tb.mtm == USHRT_MAX) /* Deleted node */
					b1 = heap[1] = heap[heap[0]--];
				else /* Too old error value */
				{
					find_nn(bins.data(), b1);
					tb.tm = i;
				}
				/* Push slot down */
				auto err = bins[b1].err;
				for (l = 1; (l2 = l + l) <= heap[0]; l = l2) {
					if ((l2 < heap[0]) && (bins[heap[l2]].err > bins[heap[l2 + 1]].err))
						++l2;
					if (err <= bins[h = heap[l2]].err)
						break;
					heap[l] = h;
				}
				heap[l] = b1;
			}

			/* Do a merge */
			auto& tb = bins[b1];
			auto& nb = bins[tb.nn];
			auto n1 = tb.cnt;
			auto n2 = nb.cnt;
			auto d = 1.0f / (n1 + n2);
			tb.ac = d * rint(n1 * tb.ac + n2 * nb.ac);
			tb.rc = d * rint(n1 * tb.rc + n2 * nb.rc);
			tb.gc = d * rint(n1 * tb.gc + n2 * nb.gc);
			tb.bc = d * rint(n1 * tb.bc + n2 * nb.bc);
			tb.cnt += n2;
			tb.mtm = ++i;

			/* Unchain deleted bin */
			bins[nb.bk].fw = nb.fw;
			bins[nb.fw].bk = nb.bk;
			nb.mtm = USHRT_MAX;
		}

		/* Fill palette */
		uint k = 0;
		for (int i = 0; k < nMaxColors; ++k) {
			auto alpha = (hasSemiTransparency || m_transparentPixelIndex > -1) ? rint(bins[i].ac) : UCHAR_MAX;
			Vec4b c1((uchar) bins[i].bc, (uchar) bins[i].gc, (uchar) bins[i].rc, alpha);
			SetPixel(palette, k, 0, c1);

			i = bins[i].fw;
		}
	}

	ushort nearestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
	{
		auto argb = GetArgb8888(c0);
		auto got = nearestMap.find(argb);
		if (got != nearestMap.end())
			return got->second;

		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			c = m_transparentColor;

		const auto nMaxColors = palette.rows;
		if (nMaxColors > 2 && m_transparentPixelIndex >= 0 && c[3] > alphaThreshold)
			k = 1;
		
		auto pr = PR, pg = PG, pb = PB, pa = PA;
		if(nMaxColors < 3)
			pr = pg = pb = pa = 1;

		double mindist = INT_MAX;		
		for (uint i = k; i < nMaxColors; ++i) {
			Vec4b c2;
			GrabPixel(c2, palette, i, 0);
			double curdist = pa * sqr(c2[3] - c[3]);
			if (curdist > mindist)
				continue;

			curdist += pr * sqr(c2[2] - c[2]);
			if (curdist > mindist)
				continue;

			curdist += pg * sqr(c2[1] - c[1]);
			if (curdist > mindist)
				continue;

			curdist += pb * sqr(c2[0] - c[0]);
			if (curdist > mindist)
				continue;

			mindist = curdist;
			k = i;
		}
		nearestMap[argb] = k;
		return k;
	}

	ushort closestColorIndex(const Mat palette, const Vec4b& c, const uint pos)
	{
		ushort k = 0;
		if (c[3] <= alphaThreshold)
			return nearestColorIndex(palette, c, pos);

		const auto nMaxColors = (ushort) palette.rows;
		vector<ushort> closest(4);
		auto argb = GetArgb8888(c);
		auto got = closestMap.find(argb);
		if (got == closestMap.end()) {
			closest[2] = closest[3] = USHRT_MAX;

			auto pr = PR, pg = PG, pb = PB, pa = PA;
			if(nMaxColors < 3)
				pr = pg = pb = pa = 1;

			for (; k < nMaxColors; ++k) {
				Vec4b c2;
				GrabPixel(c2, palette, k, 0);
				auto err = pr * sqr(c2[2] - c[2]);
				if (err >= closest[3])
					break;

				err += pg* sqr(c2[1] - c[1]);
				if (err >= closest[3])
					break;

				err += pb* sqr(c2[0] - c[0]);
				if (err >= closest[3])
					break;

				if (hasSemiTransparency)
					err += pa * sqr(c2[3] - c[3]);

				if (err < closest[2]) {
					closest[1] = closest[0];
					closest[3] = closest[2];
					closest[0] = k;
					closest[2] = err;
				}
				else if (err < closest[3]) {
					closest[1] = k;
					closest[3] = err;
				}
			}

			if (closest[3] == USHRT_MAX)
				closest[1] = closest[0];

			closestMap[argb] = closest;
		}
		else
			closest = got->second;

		auto MAX_ERR = palette.rows << 2;
		int idx = (pos + 1) % 2;
		if (closest[3] * .67 < (closest[3] - closest[2]))
			idx = 0;
		else if (closest[0] > closest[1])
			idx = pos % 2;

		if (closest[idx + 2] >= MAX_ERR || (m_transparentPixelIndex >= 0 && closest[idx] == 0))
			return nearestColorIndex(palette, c, pos);
		return closest[idx];
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
				qPixels(j, i) = (uchar) ditherFn(palette, pixel, i + j);
			}
		}

		BlueNoise::dither(pixels, palette, ditherFn, GetColorIndex, qPixels);
		return true;
	}	

	Mat PnnQuantizer::QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);
		
		Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
		int semiTransCount = 0;
		GrabPixels(srcImg, pixels4b, semiTransCount, m_transparentPixelIndex, m_transparentColor, alphaThreshold, nMaxColors);
		hasSemiTransparency = semiTransCount > 0;
		
		Mat palette(nMaxColors, 1, srcImg.type(), scalar);

		if (nMaxColors <= 32)
			PR = PG = PB = PA = 1;
		else {
			PR = coeffs[0][0]; PG = coeffs[0][1]; PB = coeffs[0][2];
		}

		if (nMaxColors > 2)
			pnnquan(pixels4b, palette, nMaxColors);
		else {
			if (m_transparentPixelIndex >= 0)
				palette.at<Vec4b>(0, 0) = m_transparentColor;
			else
				palette.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);
		}

		
		DitherFn ditherFn = dither ? nearestColorIndex : closestColorIndex;
		if (hasSemiTransparency)
			weight *= -1;

		vector<float> saliencies;
		if (nMaxColors > 256) {
			Mat qPixels(bitmapHeight, bitmapWidth, srcImg.type());
			Peano::GilbertCurve::dither(pixels4b, palette, ditherFn, GetColorIndex, qPixels, saliencies.data(), weight, dither);

			closestMap.clear();
			nearestMap.clear();
			return qPixels;
		}

		Mat1b qPixels(bitmapHeight, bitmapWidth);
		Peano::GilbertCurve::dither(pixels4b, palette, ditherFn, GetColorIndex, qPixels, saliencies.data(), weight, dither);

		if(!dither && nMaxColors > 32)
			BlueNoise::dither(pixels4b, palette, ditherFn, GetColorIndex, qPixels);

		if (m_transparentPixelIndex >= 0) {
			auto k = qPixels.at<ushort>(m_transparentPixelIndex / bitmapWidth, m_transparentPixelIndex % bitmapWidth);
			if (nMaxColors > 2)
				palette.at<Vec4b>(k, 0) = m_transparentColor;
			else if (GetArgb8888(palette.at<Vec4b>(k, 0)) != GetArgb8888(m_transparentColor))
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(1, 0));
		}
		closestMap.clear();
		nearestMap.clear();

		ProcessImagePixels(bytes, palette, qPixels, m_transparentPixelIndex >= 0);
		return palette;
	}

}
