#pragma once
#include "bitmapUtilities.h"

namespace BlueNoise
{
	extern const char TELL_BLUE_NOISE[];
	
	Vec4b diffuse(const Vec4b& pixel, const Vec4b& qPixel, const float weight, const float strength, const int x, const int y);

	void dither(const Mat4b pPixels4b, const Mat pPalette, DitherFn ditherFn, GetColorIndexFn getColorIndexFn, Mat1b qPixels, const float weight = 1.0f);
	
	Vec4b dither_pixel(const Vec4b& pixel, const float saliency, const int x, const int y,
	const float noiseDampener, const float baseSpread, unsigned int frameIndex = 0);

	bool dither_image(const Mat4b pixels4b, const Mat palette, const uint nMaxColors, DitherFn ditherFn,
	Mat1b qPixels, const vector<float>& saliencies, uint frameIndex = 0);
	
	bool dither_image(const Mat4b pixels4b, const Mat palette, const uint nMaxColors, DitherFn ditherFn,
	Mat1b qPixels, uint frameIndex = 0);
}
