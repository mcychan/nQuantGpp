#pragma once
#include <iostream>
#include <memory>
#include <vector>
using namespace std;


using DitherFn = function<ushort(const Mat, const Vec4b&, const uint)>;

using GetColorIndexFn = function<int(const Vec4b&)>;

void CalcDitherPixel(int* pDitherPixel, const Vec4b& c, const uchar* clamp, const short* rowerr, int cursor, const bool noBias);

bool dither_image(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, const bool& hasSemiTransparency, const int& transparentPixelIndex, const uint nMaxColors, Mat1b qPixels);

bool dithering_image(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, const bool& hasSemiTransparency, const int& transparentPixelIndex, const uint nMaxColors, Mat qPixels);

void ProcessImagePixels(vector<uchar>& bytes, const Mat palette, const Mat1b qPixels, const bool& hasTransparent);

void SetPixel(Mat pixels, int row, int col, Vec4b& pixel);

bool GrabPixels(const Mat source, Mat4b pixels, int& semiTransCount, int& transparentPixelIndex, Vec4b& transparentColor, const uchar alphaThreshold = 0xF, const uint nMaxColors = 2);

int GrabPixels(const Mat source, Mat4b pixels, bool& hasSemiTransparency, int& transparentPixelIndex, Vec4b& transparentColor, const uchar alphaThreshold = 0xF, const uint nMaxColors = 2);

void GrabPixel(Vec4b& pixel, const Mat pixels, int col, int row);

inline ARGB GetArgb1555(const Vec4b& c)
{
	return (c[3] & 0x80) << 8 | (c[2] & 0xF8) << 7 | (c[1] & 0xF8) << 2 | (c[0] >> 3);
}

inline ARGB GetRgb888(const Vec3b& c)
{
	return (c[2] << 16) | (c[1] << 8) | c[0];
}

inline ARGB GetArgb8888(const Vec4b& c)
{
	return (c[3] << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
}

inline int GetArgbIndex(const Vec4b& c, const bool hasSemiTransparency, const bool hasTransparency)
{
	if (hasSemiTransparency)
		return (c[3] & 0xF0) << 8 | (c[2] & 0xF0) << 4 | (c[1] & 0xF0) | (c[0] >> 4);
	if (hasTransparency)
		return GetArgb1555(c);
	return (c[2] & 0xF8) << 8 | (c[1] & 0xFC) << 3 | (c[0] >> 3);
}
