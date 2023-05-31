#pragma once
#include <memory>
#include <vector>
using namespace std;

// =============================================================
// Quantizer objects and functions
//
// COVERED CODE IS PROVIDED UNDER THIS LICENSE ON AN "AS IS" BASIS, WITHOUT WARRANTY
// OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, WITHOUT LIMITATION, WARRANTIES
// THAT THE COVERED CODE IS FREE OF DEFECTS, MERCHANTABLE, FIT FOR A PARTICULAR PURPOSE
// OR NON-INFRINGING. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE COVERED
// CODE IS WITH YOU. SHOULD ANY COVERED CODE PROVE DEFECTIVE IN ANY RESPECT, YOU (NOT
// THE INITIAL DEVELOPER OR ANY OTHER CONTRIBUTOR) ASSUME THE COST OF ANY NECESSARY
// SERVICING, REPAIR OR CORRECTION. THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL
// PART OF THIS LICENSE. NO USE OF ANY COVERED CODE IS AUTHORIZED HEREUNDER EXCEPT UNDER
// THIS DISCLAIMER.
//
// Use at your own risk!
// =============================================================

namespace nQuant
{
/**
  Xiaolin Wu color quantization algorithm
*/
	enum Pixel : uchar { Blue, Green, Red, Alpha };

	class WuQuantizer
	{
		public:
			Mat QuantizeImage(const Mat srcImg, vector<uchar>& pngBytes, uint& nMaxColors, bool dither = true, uchar alphaThreshold = 0, uchar alphaFader = 1);
	};
}