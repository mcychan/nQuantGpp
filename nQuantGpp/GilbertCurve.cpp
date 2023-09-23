/* Generalized Hilbert ("gilbert") space-filling curve for rectangular domains of arbitrary (non-power of two) sizes.
Copyright (c) 2023 Miller Cy Chan
* A general rectangle with a known orientation is split into three regions ("up", "right", "down"), for which the function calls itself recursively, until a trivial path can be produced. */

#include "stdafx.h"
#include "GilbertCurve.h"
#include "BlueNoise.h"
#include "CIELABConvertor.h"

#include <memory>
#include <deque>

namespace Peano
{
	struct ErrorBox
	{
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

	uint m_width, m_height;
	const Mat4b* m_pPixels4b;
	const Mat* m_pPalette;
	Mat1b* m_qPixels;
	DitherFn m_ditherFn;
	float* m_saliencies;
	GetColorIndexFn m_getColorIndexFn;
	deque<ErrorBox> errorq;
	float* m_weights;
	short* m_lookup;
	static uchar DITHER_MAX = 9, ditherMax;
	static int nMaxColors, thresold;
	static const float BLOCK_SIZE = 343.0f;

	template <typename T> int sign(T val) {
		return (T(0) < val) - (val < T(0));
	}

	void ditherPixel(int x, int y)
	{
		int bidx = x + y * m_width;
		auto& pixel = m_pPixels4b->at<Vec4b>(y, x);

		ErrorBox error(pixel);
		int i = 0;
		auto maxErr = DITHER_MAX - 1;
		for (auto& eb : errorq) {
			for (int j = 0; j < eb.length(); ++j) {
				error[j] += eb[j] * m_weights[i];
				if (error[j] > maxErr)
					maxErr = error[j];
			}
			++i;
		}

		auto b_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[0], 0.0f)));
		auto g_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[1], 0.0f)));
		auto r_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[2], 0.0f)));
		auto a_pix = static_cast<uchar>(min(UCHAR_MAX, (int) max(error[3], 0.0f)));

		Vec4b c2(b_pix, g_pix, r_pix, a_pix);
		auto& qPixelIndex = m_qPixels->at<uchar>(y, x);
		if (nMaxColors <= 32 && a_pix > 0xF0)
		{
			int offset = m_getColorIndexFn(c2);
			if (!m_lookup[offset])
				m_lookup[offset] = m_ditherFn(*m_pPalette, c2, bidx) + 1;
			qPixelIndex = m_lookup[offset] - 1;

			if (m_saliencies != nullptr && m_saliencies[bidx] > .65f && m_saliencies[bidx] < .7f) {
				Vec4b qPixel;
				GrabPixel(qPixel, *m_pPalette, qPixelIndex, 0);
				auto strength = 1 / 3.0f;
				c2 = BlueNoise::diffuse(pixel, qPixel, 1.0f / m_saliencies[bidx], strength, x, y);
				qPixelIndex = m_ditherFn(*m_pPalette, c2, bidx);
			}
		}
		else
			qPixelIndex = m_ditherFn(*m_pPalette, c2, bidx);

		errorq.pop_front();
		c2 = m_pPalette->at<Vec4b>(qPixelIndex, 0);
		error[0] = b_pix - c2[0];
		error[1] = g_pix - c2[1];
		error[2] = r_pix - c2[2];
		error[3] = a_pix - c2[3];

		auto denoise = nMaxColors > 2;
		auto diffuse = BlueNoise::TELL_BLUE_NOISE[bidx & 4095] > thresold;
		auto yDiff = diffuse ? 1 : CIELABConvertor::Y_Diff(pixel, c2);
		auto illusion = !diffuse && BlueNoise::TELL_BLUE_NOISE[(int)(yDiff * 4096) & 4095] > thresold;

		int errLength = denoise ? error.length() - 1 : 0;
		for (int j = 0; j < errLength; ++j) {
			if (abs(error.p[j]) >= ditherMax) {
				if (diffuse)
					error[j] = (float)tanh(error.p[j] / maxErr * 8) * (ditherMax - 1);
				else {
					if (illusion)
						error[j] = (float)(error.p[j] / maxErr * yDiff) * (ditherMax - 1);
					else
						error[j] /= (float)(1 + sqrt(ditherMax));
				}
			}
		}

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

	void GilbertCurve::dither(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, GetColorIndexFn getColorIndexFn, Mat1b qPixels, float* saliencies, double weight)
	{
		m_width = pixels4b.cols;
		m_height = pixels4b.rows;
		m_pPixels4b = &pixels4b;
		m_pPalette = &palette;
		m_qPixels = &qPixels;
		m_ditherFn = ditherFn;
		m_saliencies = saliencies;
		m_getColorIndexFn = getColorIndexFn;
		auto hasAlpha = weight < 0;
		weight = abs(weight);
		DITHER_MAX = weight < .01 ? (weight > .0025) ? (uchar)25 : 16 : 9;
		auto edge = hasAlpha ? 1 : exp(weight) + .25;
		ditherMax = (hasAlpha || DITHER_MAX > 9) ? (uchar)sqr(sqrt(DITHER_MAX) + edge) : DITHER_MAX;
		nMaxColors = palette.cols * palette.rows;
		if (nMaxColors / weight > 5000 && (weight > .045 || (weight > .01 && nMaxColors <= 64)))
			ditherMax = (uchar)sqr(5 + edge);
		else if (nMaxColors / weight < 3200 && nMaxColors > 16 && nMaxColors < 256)
			ditherMax = (uchar)sqr(5 + edge);
		thresold = DITHER_MAX > 9 ? -112 : -64;
		auto pWeights = make_unique<float[]>(DITHER_MAX);
		m_weights = pWeights.get();
		auto pLookup = make_unique<short[]>(USHRT_MAX + 1);
		m_lookup = pLookup.get();

		/* Dithers all pixels of the image in sequence using
		 * the Gilbert path, and distributes the error in
		 * a sequence of DITHER_MAX pixels.
		 */
		errorq.clear();
		errorq.resize(DITHER_MAX);
		const auto weightRatio = (float)pow(BLOCK_SIZE + 1.0f, 1.0f / (DITHER_MAX - 1.0f));
		weight = 1.0f;
		auto sumweight = 0.0f;
		for (int c = 0; c < DITHER_MAX; ++c) {
			sumweight += (m_weights[DITHER_MAX - c - 1] = 1.0f / weight);
			weight *= weightRatio;
		}

		weight = 0.0f; /* Normalize */
		for (int c = 0; c < DITHER_MAX; ++c)
			weight += (m_weights[c] /= sumweight);
		m_weights[0] += 1.0f - weight;

		if (m_width >= m_height)
			generate2d(0, 0, m_width, 0, 0, m_height);
		else
			generate2d(0, 0, 0, m_height, m_width, 0);
	}
}
