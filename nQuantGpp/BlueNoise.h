#pragma once
#include "bitmapUtilities.h"

namespace BlueNoise
{
	extern const char TELL_BLUE_NOISE[];
	
	Vec4b diffuse(const Vec4b& pixel, const Vec4b& qPixel, const float weight, const float strength, const int x, const int y);

	void dither(const Mat4b pPixels4b, const Mat pPalette, DitherFn ditherFn, GetColorIndexFn getColorIndexFn, Mat1b qPixels, const float weight = 1.0f);
}
