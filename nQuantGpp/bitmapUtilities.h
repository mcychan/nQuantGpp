#pragma once
#include <iostream>
#include <vector>
using namespace std;


using DitherFn = function<ushort(const Mat, const Vec4b&, const uint)>;

using GetColorIndexFn = function<int(const Vec4b&)>;

void CalcDitherPixel(int* pDitherPixel, const Vec4b& c, const uchar* clamp, const short* rowerr, int cursor, const bool noBias);

bool dither_image(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, const bool& hasSemiTransparency, const int& transparentPixelIndex, const uint nMaxColors, Mat1b qPixels);

bool dithering_image(const Mat4b pixels4b, const Mat palette, DitherFn ditherFn, const bool& hasSemiTransparency, const int& transparentPixelIndex, const uint nMaxColors, Mat qPixels);

void ProcessImagePixels(vector<uchar>& bytes, const Mat qPixels, const bool& hasTransparent);

void ProcessImagePixels(vector<uchar>& bytes, const Mat palette, const Mat1b qPixels, const bool& hasTransparent);

void SetPixel(Mat pixels, int row, int col, Vec4b& pixel);

bool GrabPixels(const Mat source, Mat4b pixels, int& semiTransCount, int& transparentPixelIndex, Vec4b& transparentColor, const uchar alphaThreshold = 0xF, const uint nMaxColors = 2);

int GrabPixels(const Mat source, Mat4b pixels, bool& hasSemiTransparency, int& transparentPixelIndex, Vec4b& transparentColor, const uchar alphaThreshold = 0xF, const uint nMaxColors = 2);

void GrabPixel(Vec4b& pixel, const Mat pixels, int col, int row);

inline ARGB GetArgb1555(const Vec4b& c)
{
	uint16_t a = (c[3] >> 7); // Isolate 1 bit for Alpha (0 or 1)
	return (a << 15) | // A: 1 bit to bit 15
		((c[2] >> 3) << 10) | // R: 5 bits to bits 10-14
		((c[1] >> 3) << 5) | // G: 5 bits to bits 5-9
		((c[0] >> 3));         // B: 5 bits to bits 0-4
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
	if (hasSemiTransparency) {
		return ((c[3] >> 4) << 12) | // A: Top 4 bits to 12-15
			((c[2] >> 4) << 8) | // R: Top 4 bits to 8-11
			((c[1] >> 4) << 4) | // G: Top 4 bits to 4-7
			((c[0] >> 4));        // B: Top 4 bits to 0-3
	}
	if (hasTransparency)
		return GetArgb1555(c);
	return ((c[2] >> 3) << 11) | // R: 5 bits to bits 11-15
		((c[1] >> 2) << 5) | // G: 6 bits to bits 5-10
		((c[0] >> 3));        // B: 5 bits to bits 0-4
}

inline void GetArgb(Vec4b& pixel, const Vec4b& c, const bool hasSemiTransparency, const bool hasTransparency)
{
	if (hasSemiTransparency) {
		// Isolate upper 4 bits, then mirror them to lower 4 bits to maintain brightness
		pixel[0] = (c[0] & 0xF0) | (c[0] >> 4); // B
		pixel[1] = (c[1] & 0xF0) | (c[1] >> 4); // G
		pixel[2] = (c[2] & 0xF0) | (c[2] >> 4); // R
		pixel[3] = (c[3] & 0xF0) | (c[3] >> 4); // A
		return;
	}
	if (hasTransparency) {
		// Isolate upper 5 bits, fill lower 3 bits with top 3 bits
		pixel[0] = (c[0] & 0xF8) | (c[0] >> 5); // B (5-bit)
		pixel[1] = (c[1] & 0xF8) | (c[1] >> 5); // G (5-bit)
		pixel[2] = (c[2] & 0xF8) | (c[2] >> 5); // R (5-bit)
		// Alpha is binary: either completely transparent (0) or completely opaque (255)
		pixel[3] = (c[3] >= 128) ? UCHAR_MAX : 0;
		return;
	}

	// B and R get 5 bits; G gets 6 bits
	pixel[0] = (c[0] & 0xF8) | (c[0] >> 5); // B (5-bit)
	pixel[1] = (c[1] & 0xFC) | (c[1] >> 6); // G (6-bit)
	pixel[2] = (c[2] & 0xF8) | (c[2] >> 5); // R (5-bit)
	pixel[3] = UCHAR_MAX;                   // Alpha is dropped / fully opaque
}
