#pragma once
#include "bitmapUtilities.h"

namespace Peano
{
	class GilbertCurve
	{
		public:
			static void dither(const Mat4b pixels4b, const Mat pPalette, DitherFn ditherFn, GetColorIndexFn getColorIndexFn, Mat qPixels, float* saliencies, double weight = 1.0);
	};
}
