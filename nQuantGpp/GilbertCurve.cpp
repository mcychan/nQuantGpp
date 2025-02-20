/* Generalized Hilbert ("gilbert") space-filling curve for rectangular domains of arbitrary (non-power of two) sizes.
Copyright (c) 2023-2025 Miller Cy Chan
* A general rectangle with a known orientation is split into three regions ("up", "right", "down"), for which the function calls itself recursively, until a trivial path can be produced. */

#include "stdafx.h"
#include "GilbertCurve.h"
#include "BlueNoise.h"
#include "CIELABConvertor.h"

#include <memory>
#include <list>
#include <algorithm>

namespace Peano
{
	struct ErrorBox
	{
		double yDiff = 0;
		float p[4] = { 0 };

		ErrorBox() {
		}
		ErrorBox(const Vec4b& c) {
			p[0] = c[0];
			p[1] = c[1];
			p[2] = c[2];
			p[3] = c[3];
		}
		inline float& operator[](int index)
		{
			return p[index];
		}
		inline uchar length() const
		{
			return 4;
		}
	};

	bool sortedByYDiff;
	uint m_width, m_height;
	float beta;
	const Mat4b* m_pPixels4b;
	const Mat* m_pPalette;
	Mat* m_qPixels;
	DitherFn m_ditherFn;
	float* m_saliencies;
	GetColorIndexFn m_getColorIndexFn;
	list<ErrorBox> errorq;
	vector<float> m_weights;
	ushort* m_lookup;
	static uchar DITHER_MAX = 9, ditherMax;
	static int margin, nMaxColors, thresold;
	static const float BLOCK_SIZE = 343.0f;

	template <typename T> int sign(T val) {
		return (T(0) < val) - (val < T(0));
	}

	template<typename T>
	void insert_in_order(list<T>& items, T& element, int (*compareFn)(const T& a, const T& b)) {
		auto begin = items.begin();
		auto end = items.end();
		while (begin != end && compareFn(*begin, element) < 0)
			++begin;

		items.insert(begin, element);
	}

	void initWeights(int size) {
		/* Dithers all pixels of the image in sequence using
		 * the Gilbert path, and distributes the error in
		 * a sequence of pixels size.
		 */
		errorq.resize(size);
		const auto weightRatio = (float)pow(BLOCK_SIZE + 1.0f, 1.0f / (size - 1.0f));
		auto weight = 1.0f;
		auto sumweight = 0.0f;
		m_weights.resize(size);
		for (int c = 0; c < size; ++c) {
			sumweight += (m_weights[size - c - 1] = 1.0f / weight);
			weight *= weightRatio;
		}

		weight = 0.0f; /* Normalize */
		for (int c = 0; c < size; ++c)
			weight += (m_weights[c] /= sumweight);
		m_weights[0] += 1.0f - weight;
	}

	inline int compare(const ErrorBox& o1, const ErrorBox& o2)
	{
		return sign(o1.yDiff - o2.yDiff);
	}

	void ditherPixel(int x, int y)
	{
		int bidx = x + y * m_width;
		auto& pixel = m_pPixels4b->at<Vec4b>(y, x);

		ErrorBox error(pixel);
		int i = sortedByYDiff ? m_weights.size() - 1 : 0;
		auto maxErr = DITHER_MAX - 1;
		for (auto& eb : errorq) {
			if (i < 0 || i >= m_weights.size())
				break;

			for (int j = 0; j < eb.length(); ++j) {
				error[j] += eb[j] * m_weights[i];
				if (error[j] > maxErr)
					maxErr = error[j];
			}
			i += sortedByYDiff ? -1 : 1;
		}

		auto b_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[0], 0.0f)));
		auto g_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[1], 0.0f)));
		auto r_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[2], 0.0f)));
		auto a_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[3], 0.0f)));

		Vec4b c2(b_pix, g_pix, r_pix, a_pix);
		ushort qPixelIndex = 0;
		if (m_saliencies != nullptr && !sortedByYDiff)
		{
			Vec4b qPixel;
			GrabPixel(qPixel, *m_pPalette, qPixelIndex, 0);
			auto strength = 1 / 3.0f;
			int acceptedDiff = max(2, nMaxColors - margin);
			if (nMaxColors <= 8 && m_saliencies[bidx] > .2f && m_saliencies[bidx] < .25f)
				c2 = BlueNoise::diffuse(pixel, qPixel, beta * 2 / m_saliencies[bidx], strength, x, y);
			else if (nMaxColors <= 8 || CIELABConvertor::Y_Diff(pixel, c2) < (2 * acceptedDiff)) {
				c2 = BlueNoise::diffuse(pixel, qPixel, beta * .5f / m_saliencies[bidx], strength, x, y);
				if (nMaxColors <= 8 && CIELABConvertor::U_Diff(pixel, c2) > (8 * acceptedDiff)) {
					Vec4b c1(b_pix, g_pix, r_pix, a_pix);
					if (m_saliencies[bidx] > .65f)
						c1 = pixel;
					c2 = BlueNoise::diffuse(c1, qPixel, beta * m_saliencies[bidx], strength, x, y);
				}
				if (CIELABConvertor::U_Diff(pixel, c2) > (margin * acceptedDiff))
					c2 = BlueNoise::diffuse(pixel, qPixel, beta / m_saliencies[bidx], strength, x, y);
			}
			
			if (nMaxColors < 3 || margin > 6) {
				if (nMaxColors > 8 && (CIELABConvertor::Y_Diff(pixel, c2) > (beta * acceptedDiff) || CIELABConvertor::U_Diff(pixel, c2) > (2 * acceptedDiff))) {
					auto kappa = m_saliencies[bidx] < .25f ? beta * .4f * m_saliencies[bidx] : beta * .4f / m_saliencies[bidx];
					Vec4b c1(b_pix, g_pix, r_pix, a_pix);
					c2 = BlueNoise::diffuse(c1, qPixel, kappa, strength, x, y);
				}
			}
			else if (nMaxColors > 8 && (CIELABConvertor::Y_Diff(pixel, c2) > (beta * acceptedDiff) || CIELABConvertor::U_Diff(pixel, c2) > acceptedDiff)) {
				if (beta < .3f && (nMaxColors <= 32 || m_saliencies[bidx] < beta))
					c2 = BlueNoise::diffuse(c2, qPixel, beta * .4f * m_saliencies[bidx], strength, x, y);
				else {
					Vec4b c1(b_pix, g_pix, r_pix, a_pix);
					c2 = c1;
				}
			}

			int offset = m_getColorIndexFn(c2);
			if (!m_lookup[offset])
				m_lookup[offset] = m_ditherFn(*m_pPalette, c2, bidx) + 1;
			qPixelIndex = m_lookup[offset] - 1;
		}
		else if (nMaxColors <= 32 && a_pix > 0xF0)
		{
			int offset = m_getColorIndexFn(c2);
			if (!m_lookup[offset])
				m_lookup[offset] = m_ditherFn(*m_pPalette, c2, bidx) + 1;
			qPixelIndex = m_lookup[offset] - 1;

			int acceptedDiff = max(2, nMaxColors - margin);
			if (m_saliencies != nullptr && (CIELABConvertor::Y_Diff(pixel, c2) > acceptedDiff || CIELABConvertor::U_Diff(pixel, c2) > (2 * acceptedDiff))) {
				Vec4b qPixel;
				GrabPixel(qPixel, *m_pPalette, qPixelIndex, 0);
				auto strength = 1 / 3.0f;
				c2 = BlueNoise::diffuse(pixel, qPixel, 1.0f / m_saliencies[bidx], strength, x, y);
				qPixelIndex = m_ditherFn(*m_pPalette, c2, bidx);
			}
		}
		else
			qPixelIndex = m_ditherFn(*m_pPalette, c2, bidx);

		if(errorq.size() >= DITHER_MAX)
			errorq.pop_front();
		else if(!errorq.empty())
			initWeights(errorq.size());

		c2 = m_pPalette->at<Vec4b>(qPixelIndex, 0);
		if (nMaxColors > 256)
			SetPixel(*m_qPixels, y, x, c2);
		else
			m_qPixels->at<uchar>(y, x) = qPixelIndex;

		error[0] = b_pix - c2[0];
		error[1] = g_pix - c2[1];
		error[2] = r_pix - c2[2];
		error[3] = a_pix - c2[3];

		auto denoise = nMaxColors > 2;
		auto diffuse = BlueNoise::TELL_BLUE_NOISE[bidx & 4095] > thresold;
		error.yDiff = sortedByYDiff ? CIELABConvertor::Y_Diff(pixel, c2) : 1;
		auto illusion = !diffuse && BlueNoise::TELL_BLUE_NOISE[(int)(error.yDiff * 4096) & 4095] > thresold;
		auto yDiff = 1.0;
		if (!m_saliencies && !sortedByYDiff)
			yDiff = CIELABConvertor::Y_Diff(pixel, c2);

		int errLength = denoise ? error.length() - 1 : 0;
		for (int j = 0; j < errLength; ++j) {
			if (abs(error.p[j]) / yDiff >= ditherMax) {
				if (diffuse)
					error[j] = (float)tanh(error.p[j] / maxErr * 8) * (ditherMax - 1);
				else {
					if (illusion)
						error[j] = (float)(error.p[j] / maxErr * error.yDiff) * (ditherMax - 1);
					else
						error[j] /= (float)(1 + sqrt(ditherMax));
				}
			}
		}

		if (sortedByYDiff)
			insert_in_order<ErrorBox>(errorq, error, &compare);
		else
			errorq.emplace_back(error);
	}

	void generate2d(int x, int y, int ax, int ay, int bx, int by) {
		int w = abs(ax + ay);
		int h = abs(bx + by);
		int dax = sign(ax);
		int day = sign(ay);
		int dbx = sign(bx);
		int dby = sign(by);

		if (h == 1) {
			for (int i = 0; i < w; ++i) {
				ditherPixel(x, y);
				x += dax;
				y += day;
			}
			return;
		}

		if (w == 1) {
			for (int i = 0; i < h; ++i) {
				ditherPixel(x, y);
				x += dbx;
				y += dby;
			}
			return;
		}

		int ax2 = ax / 2;
		int ay2 = ay / 2;
		int bx2 = bx / 2;
		int by2 = by / 2;

		int w2 = abs(ax2 + ay2);
		int h2 = abs(bx2 + by2);

		if (2 * w > 3 * h) {
			if ((w2 % 2) != 0 && w > 2) {
				ax2 += dax;
				ay2 += day;
			}
			generate2d(x, y, ax2, ay2, bx, by);
			generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by);
			return;
		}

		if ((h2 % 2) != 0 && h > 2) {
			bx2 += dbx;
			by2 += dby;
		}

		generate2d(x, y, bx2, by2, ax2, ay2);
		generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2);
		generate2d(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2));
	}

	void GilbertCurve::dither(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, GetColorIndexFn getColorIndexFn, Mat qPixels, float* saliencies, double weight)
	{
		m_width = pixels4b.cols;
		m_height = pixels4b.rows;
		m_pPixels4b = &pixels4b;
		m_pPalette = &palette;
		m_qPixels = &qPixels;
		m_ditherFn = ditherFn;
		auto hasAlpha = weight < 0;
		m_saliencies = hasAlpha ? nullptr : saliencies;
		m_getColorIndexFn = getColorIndexFn;
		
		weight = abs(weight);
		margin = weight < .0025 ? 12 : weight < .004 ? 8 : 6;
		sortedByYDiff = !hasAlpha && m_saliencies && nMaxColors >= 128 && weight >= .052;
		nMaxColors = palette.cols * palette.rows;
		beta = nMaxColors > 8 ? (float) max(.25, 1 - (.022f + weight) * nMaxColors) : 1;
		if (nMaxColors > 64 || (beta < 1 && weight > .02))
			beta *= .4f;
		DITHER_MAX = weight < .01 ? (weight > .0025) ? (uchar)25 : 16 : 9;
		auto edge = hasAlpha ? 1 : exp(weight) + .25;
		auto deviation = !hasAlpha && weight > .002 ? .25 : 1;
		ditherMax = (hasAlpha || DITHER_MAX > 9) ? (uchar)sqr(sqrt(DITHER_MAX) + edge * deviation) : DITHER_MAX;
		int density = nMaxColors > 16 ? 3200 : 1500;
		if (nMaxColors / weight > 5000 && (weight > .045 || (weight > .01 && nMaxColors <= 64)))
			ditherMax = (uchar)sqr(5 + edge);
		else if (weight < .03 && nMaxColors / weight < density && nMaxColors >= 16 && nMaxColors < 256)
			ditherMax = (uchar)sqr(5 + edge);
		thresold = DITHER_MAX > 9 ? -112 : -64;
		m_weights.clear();
		auto pLookup = make_unique<ushort[]>(USHRT_MAX + 1);
		m_lookup = pLookup.get();

		if(!sortedByYDiff)
			initWeights(DITHER_MAX);

		if (m_width >= m_height)
			generate2d(0, 0, m_width, 0, 0, m_height);
		else
			generate2d(0, 0, 0, m_height, m_width, 0);
	}
}
