/* Fast pairwise nearest neighbor based algorithm for multilevel thresholding
Copyright (C) 2004-2016 Mark Tyler and Dmitry Groshev
Copyright (c) 2018-2021 Miller Cy Chan
* error measure; time used is proportional to number of bins squared - WJ */

#include "stdafx.h"
#include "PnnLABQuantizer.h"
#include "bitmapUtilities.h"
#include "BlueNoise.h"
#include "GilbertCurve.h"
#include <ctime>
#include <unordered_map>

namespace PnnLABQuant
{
	double PR = 0.299, PG = 0.587, PB = 0.114, PA = .3333;
	uchar alphaThreshold = 0xF;	
	double weight = 1.0;
	
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);

	static const float coeffs[3][3] = {
		{0.299f, 0.587f, 0.114f},
		{-0.14713f, -0.28886f, 0.436f},
		{0.615f, -0.51499f, -0.10001f}
	};

	PnnLABQuantizer::PnnLABQuantizer() {
	}

	PnnLABQuantizer::PnnLABQuantizer(const PnnLABQuantizer& quantizer) {
		hasSemiTransparency = quantizer.hasSemiTransparency;
		m_transparentPixelIndex = quantizer.m_transparentPixelIndex;
		saliencies = quantizer.saliencies;
		pixelMap.insert(quantizer.pixelMap.begin(), quantizer.pixelMap.end());
		isGA = true;
		proportional = quantizer.proportional;
	}

	void PnnLABQuantizer::GetLab(const Vec4b& pixel, CIELABConvertor::Lab& lab1)
	{
		auto argb = GetArgb8888(pixel);
		auto got = pixelMap.find(argb);
		if (got == pixelMap.end()) {
			CIELABConvertor::RGB2LAB(pixel, lab1);
			pixelMap[argb] = lab1;
		}
		else
			lab1 = got->second;
	}

	void PnnLABQuantizer::find_nn(pnnbin* bins, int idx, bool texicab)
	{
		int nn = 0;
		double err = 1e100;

		auto& bin1 = bins[idx];
		auto n1 = bin1.cnt;
		CIELABConvertor::Lab lab1;
		lab1.alpha = bin1.ac, lab1.L = bin1.Lc, lab1.A = bin1.Ac, lab1.B = bin1.Bc;
		for (int i = bin1.fw; i; i = bins[i].fw) {
			auto n2 = bins[i].cnt;
			auto nerr2 = (n1 * n2) / (n1 + n2);
			if (nerr2 >= err)
				continue;

			CIELABConvertor::Lab lab2;
			lab2.alpha = bins[i].ac, lab2.L = bins[i].Lc, lab2.A = bins[i].Ac, lab2.B = bins[i].Bc;
			auto alphaDiff = hasSemiTransparency ? sqr(lab2.alpha - lab1.alpha) / exp(1.5) : 0;
			auto nerr = nerr2 * alphaDiff;
			if (nerr >= err)
				continue;

			if (hasSemiTransparency || !texicab) {
				nerr += (1 - ratio) * nerr2 * sqr(lab2.L - lab1.L);
				if (nerr >= err)
					continue;

				nerr += (1 - ratio) * nerr2 * sqr(lab2.A - lab1.A);
				if (nerr >= err)
					continue;

				nerr += (1 - ratio) * nerr2 * sqr(lab2.B - lab1.B);
			}
			else {
				nerr += (1 - ratio) * nerr2 * abs(lab2.L - lab1.L);
				if (nerr >= err)
					continue;

				nerr += (1 - ratio) * nerr2 * sqrt(sqr(lab2.A - lab1.A) + sqr(lab2.B - lab1.B));
			}

			if (nerr >= err)
				continue;

			auto deltaL_prime_div_k_L_S_L = CIELABConvertor::L_prime_div_k_L_S_L(lab1, lab2);
			nerr += ratio * nerr2 * sqr(deltaL_prime_div_k_L_S_L);
			if (nerr >= err)
				continue;

			double a1Prime, a2Prime, CPrime1, CPrime2;
			auto deltaC_prime_div_k_L_S_L = CIELABConvertor::C_prime_div_k_L_S_L(lab1, lab2, a1Prime, a2Prime, CPrime1, CPrime2);
			nerr += ratio * nerr2 * sqr(deltaC_prime_div_k_L_S_L);
			if (nerr >= err)
				continue;

			double barCPrime, barhPrime;
			auto deltaH_prime_div_k_L_S_L = CIELABConvertor::H_prime_div_k_L_S_L(lab1, lab2, a1Prime, a2Prime, CPrime1, CPrime2, barCPrime, barhPrime);
			nerr += ratio * nerr2 * sqr(deltaH_prime_div_k_L_S_L);
			if (nerr >= err)
				continue;

			nerr += ratio * nerr2 * CIELABConvertor::R_T(barCPrime, barhPrime, deltaC_prime_div_k_L_S_L, deltaH_prime_div_k_L_S_L);
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
			if (quan_rt > 1)
				return[](const float& cnt) { return (float)(int)pow(cnt, 0.75); };
			if (nMaxColors < 64)
				return[](const float& cnt) {
					return (float)(int) sqrt(cnt);
				};
			return[](const float& cnt) {
				return (float) sqrt(cnt);
			};
		}
		return[](const float& cnt) { return cnt; };
	}

	void PnnLABQuantizer::pnnquan(const Mat4b pixels, Mat palette, uint& nMaxColors)
	{
		short quan_rt = 1;
		vector<pnnbin> bins(USHRT_MAX + 1);
		auto length = (size_t) pixels.rows * pixels.cols;
		saliencies.resize(length);
		auto saliencyBase = .1f;

		/* Build histogram */
		for (int y = 0; y < pixels.rows; ++y)
		{
			for (int x = 0; x < pixels.cols; ++x)
			{
				auto c = pixels(y, x);
				if (c[3] <= alphaThreshold)
					c = m_transparentColor;

				int index = GetArgbIndex(c, hasSemiTransparency, m_transparentPixelIndex >= 0);

				CIELABConvertor::Lab lab1;
				GetLab(c, lab1);
				auto& tb = bins[index];
				tb.ac += c[3];
				tb.Lc += lab1.L;
				tb.Ac += lab1.A;
				tb.Bc += lab1.B;
				tb.cnt += 1.0;
				if (lab1.alpha > alphaThreshold && nMaxColors < 32) {
					int i = x + y * pixels.cols;
					saliencies[i] = saliencyBase + (1 - saliencyBase) * lab1.L / 100.0f;
				}
			}
		}

		/* Cluster nonempty bins at one end of array */
		int maxbins = 0;

		for (int i = 0; i < bins.size(); ++i) {
			if (bins[i].cnt <= 0.0)
				continue;

			auto d = 1.0f / bins[i].cnt;
			bins[i].ac *= d;
			bins[i].Lc *= d;
			bins[i].Ac *= d;
			bins[i].Bc *= d;

			bins[maxbins++] = bins[i];
		}

		proportional = sqr(nMaxColors) / maxbins;
		if ((m_transparentPixelIndex >= 0 || hasSemiTransparency) && nMaxColors < 32)
			quan_rt = -1;

		weight = min(0.9, nMaxColors * 1.0 / maxbins);
		if (weight > .0015 && weight < .002)
			quan_rt = 2;
		if (weight < .04 && PG < 1 && PG >= coeffs[0][1]) {
			auto delta = exp(1.75) * weight;
			PG -= delta;
			PB += delta;
			if (nMaxColors >= 64)
				quan_rt = 0;
		}

		if (pixelMap.size() <= nMaxColors) {
			/* Fill palette */
			nMaxColors = pixelMap.size();
			palette = palette.rowRange(0, nMaxColors);
			int k = 0;
			for (const auto& [pixel, lab] : pixelMap) {
				uchar red = pixel & 0xff,
				green = (pixel >> 8) & 0xff,
				blue = (pixel >> 16) & 0xff,
				alpha = (pixel >> 24) & 0xff;
				if (palette.channels() == 4)
					palette.at<Vec4b>(k, 0) = Vec4b(blue, green, red, alpha);
				else
					palette.at<Vec3b>(k, 0) = Vec3b(blue, green, red);

				if (k > 0 && alpha == 0)
					swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(k, 0));
				++k;
			}

			return;
		}

		auto quanFn = getQuanFn(nMaxColors, quan_rt);

		int j = 0;
		for (; j < maxbins - 1; ++j) {
			bins[j].fw = j + 1;
			bins[j + 1].bk = j;

			bins[j].cnt = quanFn(bins[j].cnt);
		}
		bins[j].cnt = quanFn(bins[j].cnt);

		const bool texicab = proportional > .025;

		if(!isGA) {
			if (hasSemiTransparency)
				ratio = .5;
			else if (quan_rt != 0 && nMaxColors < 64) {
				if (proportional > .018 && proportional < .022)
					ratio = min(1.0, proportional + weight * exp(3.622));
				else if (proportional > .1)
					ratio = min(1.0, 1.0 - weight);
				else if (proportional > .04)
					ratio = min(1.0, weight * exp(2.44));
				else if (proportional > .028)
					ratio = min(1.0, weight * exp(3.225));
				else {
					auto beta = (maxbins % 2 == 0) ? 2 : 1;
					ratio = min(1.0, proportional + beta * weight * exp(1.947));
				}
			}
			else if (nMaxColors > 256)
				ratio = min(1.0, 1 - 1.0 / proportional);
			else
				ratio = min(1.0, max(.98, 1 - weight * .7));

			if (!hasSemiTransparency && quan_rt < 0)
				ratio = min(1.0, weight * exp(1.947));
		}

		int h, l, l2;
		/* Initialize nearest neighbors and build heap of them */
		auto heap = make_unique<int[]>(bins.size() + 1);
		for (int i = 0; i < maxbins; ++i) {
			find_nn(bins.data(), i, texicab);
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

		if (!isGA && quan_rt > 0 && nMaxColors < 64 && (proportional < .023 || proportional > .05) && proportional < .1)
			ratio = min(1.0, proportional - weight * exp(2.107));
		else if (isGA)
			ratio = ratioY;

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
					find_nn(bins.data(), b1, texicab && proportional < 1);
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
			auto d = 1.0 / (n1 + n2);
			tb.ac = d * (n1 * tb.ac + n2 * nb.ac);
			tb.Lc = d * (n1 * tb.Lc + n2 * nb.Lc);
			tb.Ac = d * (n1 * tb.Ac + n2 * nb.Ac);
			tb.Bc = d * (n1 * tb.Bc + n2 * nb.Bc);
			tb.cnt += n2;
			tb.mtm = ++i;

			/* Unchain deleted bin */
			bins[nb.bk].fw = nb.fw;
			bins[nb.fw].bk = nb.bk;
			nb.mtm = USHRT_MAX;
		}

		/* Fill palette */
		short k = 0;
		for (int i = 0;; ++k) {
			CIELABConvertor::Lab lab1;
			lab1.alpha = (hasSemiTransparency || m_transparentPixelIndex > -1) ? rint(bins[i].ac) : UCHAR_MAX;
			lab1.L = bins[i].Lc, lab1.A = bins[i].Ac, lab1.B = bins[i].Bc;
			Vec4b c1;
			CIELABConvertor::LAB2RGB(c1, lab1);
			SetPixel(palette, k, 0, c1);

			if (!(i = bins[i].fw))
				break;
		}

		if (k < nMaxColors - 1) {
			nMaxColors = k + 1;
			palette = palette.rowRange(0, nMaxColors);
		}
	}

	ushort PnnLABQuantizer::nearestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
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
		if (nMaxColors > 2 && hasAlpha() && c[3] > alphaThreshold)
			k = 1;

		double mindist = INT_MAX;
		CIELABConvertor::Lab lab1, lab2;
		GetLab(c, lab1);
		
		for (uint i = k; i < nMaxColors; ++i) {
			Vec4b c2;
			GrabPixel(c2, palette, i, 0);
			auto curdist = hasSemiTransparency ? sqr(c2[3] - c[3]) / exp(1.5) : 0;
			if (curdist > mindist)
				continue;

			GetLab(c2, lab2);
			if (nMaxColors <= 4) {
				curdist += sqr(c2[2] - c[2]);
				if (curdist > mindist)
					continue;

				curdist += sqr(c2[1] - c[1]);
				if (curdist > mindist)
					continue;

				curdist += sqr(c2[0] - c[0]);
				if (hasSemiTransparency) {
					if (curdist > mindist)
						continue;
					curdist += sqr(c2[3] - c[3]);
				}
			}
			else if (hasSemiTransparency) {
				curdist += sqr(lab2.L - lab1.L);
				if (curdist > mindist)
					continue;

				curdist += sqr(lab2.A - lab1.A);
				if (curdist > mindist)
					continue;

				curdist += sqr(lab2.B - lab1.B);
			}
			else if (nMaxColors > 32) {
				curdist += abs(lab2.L - lab1.L);
				if (curdist > mindist)
					continue;

				curdist += sqrt(sqr(lab2.A - lab1.A) + sqr(lab2.B - lab1.B));
			}
			else {
				auto deltaL_prime_div_k_L_S_L = CIELABConvertor::L_prime_div_k_L_S_L(lab1, lab2);
				curdist += sqr(deltaL_prime_div_k_L_S_L);
				if (curdist > mindist)
					continue;

				double a1Prime, a2Prime, CPrime1, CPrime2;
				auto deltaC_prime_div_k_L_S_L = CIELABConvertor::C_prime_div_k_L_S_L(lab1, lab2, a1Prime, a2Prime, CPrime1, CPrime2);
				curdist += sqr(deltaC_prime_div_k_L_S_L);
				if (curdist > mindist)
					continue;

				double barCPrime, barhPrime;
				auto deltaH_prime_div_k_L_S_L = CIELABConvertor::H_prime_div_k_L_S_L(lab1, lab2, a1Prime, a2Prime, CPrime1, CPrime2, barCPrime, barhPrime);
				curdist += sqr(deltaH_prime_div_k_L_S_L);
				if (curdist > mindist)
					continue;

				curdist += CIELABConvertor::R_T(barCPrime, barhPrime, deltaC_prime_div_k_L_S_L, deltaH_prime_div_k_L_S_L);
			}

			if (curdist > mindist)
				continue;
			mindist = curdist;
			k = i;
		}
		nearestMap[argb] = k;
		return k;
	}

	ushort PnnLABQuantizer::closestColorIndex(const Mat palette, const Vec4b& c, const uint pos)
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
			
			int start = 0;
			if(BlueNoise::TELL_BLUE_NOISE[pos & 4095] > -88)
				start = 1;
			
			for (; k < nMaxColors; ++k) {
				Vec4b c2;
				GrabPixel(c2, palette, k, 0);
				
				auto err = PR * (1 - ratio) * sqr(c2[2] - c[2]);
				if (err >= closest[3])
					continue;

				err += PG * (1 - ratio) * sqr(c2[1] - c[1]);
				if (err >= closest[3])
					continue;

				err += PB * (1 - ratio) * sqr(c2[0] - c[0]);
				if (err >= closest[3])
					continue;

				if (hasSemiTransparency) {
					err += PA * (1 - ratio) * sqr(c2[3] - c[3]);
					start = 1;
				}
				
				for (int i = start; i < 3; ++i) {
					err += ratio * sqr(coeffs[i][0] * (c2[2] - c[2]));
					if (err >= closest[3])
						break;
						
					err += ratio * sqr(coeffs[i][1] * (c2[1] - c[1]));
					if (err >= closest[3])
						break;
						
					err += ratio * sqr(coeffs[i][2] * (c2[0] - c[0]));
					if (err >= closest[3])
						break;
				}

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

		auto MAX_ERR = palette.rows;
		if(PG < coeffs[0][1] && BlueNoise::TELL_BLUE_NOISE[pos & 4095] > -88)
			return nearestColorIndex(palette, c, pos);

		int idx = 1;
		if (closest[2] == 0 || (rand() % (int)ceil(closest[3] + closest[2])) <= closest[3])
			idx = 0;

		if (closest[idx + 2] >= MAX_ERR || (hasAlpha() && closest[idx + 2] == 0))
			return nearestColorIndex(palette, c, pos);
		return closest[idx];
	}

	bool PnnLABQuantizer::quantize_image(const Mat4b pixels, const Mat palette, const uint nMaxColors, Mat1b qPixels, const bool dither)
	{
		auto width = pixels.cols;
		auto height = pixels.rows;

		auto NearestColorIndex = [this, nMaxColors](const Mat palette, const Vec4b& c, const uint pos) -> ushort {
			if (hasAlpha() || nMaxColors < 64)
				return nearestColorIndex(palette, c, pos);
			return closestColorIndex(palette, c, pos);
		};

		if (dither)
			return dither_image(pixels, palette, NearestColorIndex, hasSemiTransparency, m_transparentPixelIndex, nMaxColors, qPixels);

		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				auto& pixel = pixels(j, i);
				qPixels(j, i) = (uchar) NearestColorIndex(palette, pixel, i + j);
			}
		}
		return true;
	}
	
	void PnnLABQuantizer::clear()
	{
		closestMap.clear();
		nearestMap.clear();
	}

	bool PnnLABQuantizer::IsGA() const {
		return isGA;
	}

	bool PnnLABQuantizer::hasAlpha() const {
		return m_transparentPixelIndex >= 0;
	}
	
	void PnnLABQuantizer::setRatio(double ratioX, double ratioY) {
		ratio = min(1.0, ratioX);
		this->ratioY = min(1.0, ratioY);
		clear();
	}

	void PnnLABQuantizer::grabPixels(const Mat srcImg, Mat4b pixels, uint& nMaxColors, bool& hasSemiTransparency)
	{
		int semiTransCount = 0;
		GrabPixels(srcImg, pixels, semiTransCount, m_transparentPixelIndex, m_transparentColor, alphaThreshold, nMaxColors);
		this->hasSemiTransparency = hasSemiTransparency = semiTransCount > 0;
	}


	Mat PnnLABQuantizer::QuantizeImage(const Mat4b pixels4b, Mat palette, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		if (nMaxColors <= 32)
			PR = PG = PB = PA = 1;
		else {
			PR = coeffs[0][0]; PG = coeffs[0][1]; PB = coeffs[0][2];
		}

		auto bitmapWidth = pixels4b.cols;
		auto bitmapHeight = pixels4b.rows;
		
		if (nMaxColors > 2) {
			if(m_palette.empty())
				pnnquan(pixels4b, palette, nMaxColors);
		}
		else {
			if (m_transparentPixelIndex >= 0)
				palette.at<Vec4b>(0, 0) = m_transparentColor;
			else
				palette.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);
		}

		Mat1b qPixels(bitmapHeight, bitmapWidth);
		if (hasSemiTransparency)
			weight *= -1;

		auto GetColorIndex = [&](const Vec4b& c) -> int {
			return GetArgbIndex(c, hasSemiTransparency, hasAlpha());
		};
		DitherFn ditherFn;
		if (hasAlpha() || nMaxColors < 64)
			ditherFn = [&](const Mat palette, const Vec4b& c, const uint pos) -> unsigned short {
			return nearestColorIndex(palette, c, pos);
		};
		else
			ditherFn = [&](const Mat palette, const Vec4b& c, const uint pos) -> unsigned short {
			return closestColorIndex(palette, c, pos);
		};

		Peano::GilbertCurve::dither(pixels4b, palette, ditherFn, GetColorIndex, qPixels, saliencies.data(), weight);

		if (nMaxColors > 256) {
			auto NearestColorIndex = [&](const Mat palette, const Vec4b& c, const uint pos) -> ushort {
				return nearestColorIndex(palette, c, pos);
			};

			Mat qPixels(bitmapHeight, bitmapWidth, palette.type());
			dithering_image(pixels4b, palette, NearestColorIndex, hasSemiTransparency, m_transparentPixelIndex, nMaxColors, qPixels);

			pixelMap.clear();
			clear();
			return qPixels;
		}

		if (!dither) {
			const auto delta = sqr(nMaxColors) / pixelMap.size();
			auto weight = delta > 0.023 ? 1.0f : (float)(36.921 * delta + 0.906);
			BlueNoise::dither(pixels4b, palette, ditherFn, GetColorIndex, qPixels, weight);
		}

		if (m_transparentPixelIndex >= 0) {
			auto k = qPixels(m_transparentPixelIndex / bitmapWidth, m_transparentPixelIndex % bitmapWidth);
			if (nMaxColors > 2)
				palette.at<Vec4b>(k, 0) = m_transparentColor;
			else if (GetArgb8888(palette.at<Vec4b>(k, 0)) != GetArgb8888(m_transparentColor))
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(1, 0));
		}
		pixelMap.clear();
		clear();

		ProcessImagePixels(bytes, palette, qPixels, hasAlpha());
		return palette;
	}

	Mat PnnLABQuantizer::QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
		grabPixels(srcImg, pixels4b, nMaxColors, hasSemiTransparency);

		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);
		Mat palette(nMaxColors, 1, srcImg.type(), scalar);
		return this->QuantizeImage(pixels4b, palette, bytes, nMaxColors, dither);
	}

}
