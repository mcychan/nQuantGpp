#pragma once
#include "CIELABConvertor.h"
#include <memory>
#include <vector>
using namespace std;

namespace PnnLABQuant
{
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

	class PnnLABQuantizer
	{
		private:
			bool hasSemiTransparency = false;
			int m_transparentPixelIndex = -1;
			bool isGA = false, isNano = false;
			double proportional = 1.0, ratio = .5, ratioY = .5;
			unordered_map<ARGB, CIELABConvertor::Lab> pixelMap;
			unordered_map<int, vector<ushort> > closestMap;
			unordered_map<int, ushort> nearestMap;
			vector<float> saliencies;

			struct pnnbin {
				float ac = 0, Lc = 0, Ac = 0, Bc = 0, err = 0;
				float cnt = 0;
				int nn = 0, fw = 0, bk = 0, tm = 0, mtm = 0;
			};

			void find_nn(pnnbin* bins, int idx, bool texicab);
			ushort closestColorIndex(const Mat palette, const Vec4b& c, const uint pos);
			ushort hybridColorIndex(const Mat palette, const Vec4b& c, const uint pos);
			bool quantize_image(const Mat4b pixels, const Mat palette, const uint nMaxColors, Mat1b qPixels, const bool dither);

		public:
			PnnLABQuantizer();
			PnnLABQuantizer(const PnnLABQuantizer& quantizer);
			void clear();
			void pnnquan(const Mat4b pixels, Mat palette, uint& nMaxColors);
			const bool IsGA() const;
			void GetLab(const Vec4b& pixel, CIELABConvertor::Lab& lab1);
			const bool hasAlpha() const;
			inline const double getProportional() const {
				return proportional;
			}
			ushort nearestColorIndex(const Mat palette, const Vec4b& c0, const uint pos);
			void setRatio(double ratioX, double ratioY);
			void grabPixels(const Mat srcImg, Mat4b pixels, uint& nMaxColors, bool& hasSemiTransparency);
			Mat QuantizeImageByPal(const Mat4b pixels, const Mat palette, vector<uchar>& bytes, uint& nMaxColors, bool dither = true);
			Mat QuantizeImage(const Mat4b pixels, Mat palette, vector<uchar>& bytes, uint& nMaxColors, bool dither = true);
			Mat QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither = true);
	};
}