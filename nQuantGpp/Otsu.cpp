/* Otsu's Image Segmentation Method
  Copyright (C) 2009 Tolga Birdal
  Copyright (c) 2023 Miller Cy Chan
*/

#include "stdafx.h"
#include "Otsu.h"
#include "bitmapUtilities.h"
#include "GilbertCurve.h"
#include <unordered_map>

namespace OtsuThreshold
{
	uchar alphaThreshold = 0xF;
	bool hasSemiTransparency = false;
	int m_transparentPixelIndex = -1;
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);
	unordered_map<ARGB, ushort> nearestMap;

	// function is used to compute the q values in the equation
	static float px(int init, int end, int* hist)
	{
		int sum = 0;
		for (int i = init; i <= end; ++i)
			sum += hist[i];

		return (float) sum;
	}

	// function is used to compute the mean values in the equation (mu)
	static float mx(int init, int end, int* hist)
	{
		int sum = 0;
		for (int i = init; i <= end; ++i)
			sum += i * hist[i];

		return (float) sum;
	}

	// finds the maximum element in a vector
	static short findMax(float* vec, int n)
	{
		float maxVec = 0;
		short idx= 0;

		for (short i = 1; i < n - 1; ++i) {
			if (vec[i] > maxVec) {
				maxVec = vec[i];
				idx = i;
			}
		}
		return idx;
	}

	// simply computes the image histogram
	void getHistogram(const vector<Vec4b>& pixels, int* hist)
	{
		for (auto c : pixels) {
			if (c[3] <= alphaThreshold)
				continue;

			hist[c[2]]++;
			hist[c[1]]++;
			hist[c[0]]++;
		}
	}
		
	short getOtsuThreshold(const vector<Vec4b>& pixels)
	{
		float vet[256] = { 0 };
		int hist[256] = { 0 };
		
		getHistogram(pixels, hist);

		// loop through all possible t values and maximize between class variance
		for (int k = 1; k != UCHAR_MAX; ++k) {
			float p1 = px(0, k, hist);
			float p2 = px(k + 1, UCHAR_MAX, hist);
			float p12 = p1 * p2;
			if (p12 == 0) 
				p12 = 1;
			float diff = (mx(0, k, hist) * p2) - (mx(k + 1, UCHAR_MAX, hist) * p1);
			vet[k] = diff * diff / p12;
		}

		return findMax(vet, 256);
	}
	
	void threshold(Mat4b pixels, short thresh, float weight = 1.0f)
	{
		auto maxThresh = (uchar)thresh;
		if (thresh >= 200)
		{
			weight = m_transparentPixelIndex >= 0 ? .9f : .8f;
			maxThresh = (uchar)(thresh * weight);
			thresh = 200;
		}

		auto minThresh = (uchar)(thresh * weight);
		for (uint y = 0; y < pixels.rows; ++y)
		{
			for (uint x = 0; x < pixels.cols; ++x)
			{
				auto& c = pixels(y, x);
				if (c[2] + c[1] + c[0] > maxThresh * 3)
					c = Vec4b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, c[3]);
				else if (m_transparentPixelIndex >= 0 || c[2] + c[1] + c[0] < minThresh * 3)
					c = Vec4b(0, 0, 0, c[3]);
			}
		}
	}
	
	inline auto GetColorIndex(const Vec4b& bgra)
	{
		return GetArgbIndex(bgra, hasSemiTransparency, m_transparentPixelIndex >= 0);
	}

	ushort nearestColorIndex(const Mat palette, const Vec4b& pixel, const uint pos)
	{
		auto argb = GetArgb8888(pixel);
		auto got = nearestMap.find(argb);
		if (got != nearestMap.end())
			return got->second;

		ushort k = 0;
		if (pixel[3] <= alphaThreshold)
			return k;

		double mindist = INT_MAX;
		for (int i = 0; i < palette.rows; ++i) {
			auto& c2 = palette.at<Vec4b>(i, 0);
			auto curdist = sqr(c2[3] - pixel[3]);
			if (curdist > mindist)
				continue;

			curdist += sqr(c2[2] - pixel[2]);
			if (curdist > mindist)
				continue;

			curdist += sqr(c2[1] - pixel[1]);
			if (curdist > mindist)
				continue;

			curdist += sqr(c2[0] - pixel[0]);
			if (curdist > mindist)
				continue;

			mindist = curdist;
			k = i;
		}
		nearestMap[argb] = k;
		return k;
	}

	void Otsu::ConvertToGrayScale(const Mat srcImg, Mat dest)
	{
		double min1 = UCHAR_MAX;
		double max1 = .0;
		
		if(srcImg.channels() == 3) {
			vector<Mat> planes;
			split(srcImg, planes);
			minMaxLoc(planes[1], &min1, &max1);
		}
		else {
			for (uint y = 0; y < srcImg.rows; ++y)
			{
				for (uint x = 0; x < srcImg.cols; ++x)
				{
					Vec4b c0;
					if(srcImg.channels() == 4) {
						c0 = srcImg.at<Vec4b>(y, x);
						auto alfa = c0[3];
						if (alfa <= alphaThreshold)
							continue;
					}
					else {
						auto& pixel = srcImg.at<Vec3b>(y, x);
						c0 = Vec4b(pixel[0], pixel[1], pixel[2], UCHAR_MAX);
					}

					auto green = c0[1];
					if (min1 > green)
						min1 = green;

					if (max1 < green)
						max1 = green;
				}
			}
		}

		for (uint y = 0; y < srcImg.rows; ++y)
		{
			for (uint x = 0; x < srcImg.cols; ++x)
			{				
				auto alfa = UCHAR_MAX;
				Vec4b c0;
				if(srcImg.channels() == 4) {
					c0 = srcImg.at<Vec4b>(y, x);
					alfa = c0[3];
					if (alfa <= alphaThreshold)
						continue;
				}
				else {
					auto& pixel = srcImg.at<Vec3b>(y, x);
					c0 = Vec4b(pixel[0], pixel[1], pixel[2], UCHAR_MAX);
				}

				auto green = c0[1];
				auto grey = (int)((green - min1) * (UCHAR_MAX / (max1 - min1)));
				if (dest.channels() == 4) {
					auto& pixel = dest.at<Vec4b>(y, x);
					pixel[0] = pixel[1] = pixel[2] = grey;
					pixel[3] = alfa;
				}
				else {
					auto& pixel = dest.at<Vec3b>(y, x);
					pixel[0] = pixel[1] = pixel[2] = grey;
				}
			}
		}
	}


	Mat Otsu::ConvertGrayScaleToBinary(const Mat srcImg, vector<uchar>& bytes, bool isGrayscale)
	{		
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);

		Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
		GrabPixels(srcImg, pixels4b, hasSemiTransparency, m_transparentPixelIndex, m_transparentColor, alphaThreshold);
		vector<Vec4b> pixels(pixels4b.begin(), pixels4b.end());

		Mat dest(bitmapHeight, bitmapWidth, srcImg.type(), scalar);
		if (!isGrayscale)
			ConvertToGrayScale(srcImg, dest);

		auto otsuThreshold = getOtsuThreshold(pixels);
		threshold(pixels4b, otsuThreshold);

		Mat palette(2, 1, srcImg.type(), scalar);
		if (srcImg.channels() == 4)
			palette.at<Vec4b>(0, 0) = m_transparentColor;
		else
			palette.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);

		Mat1b qPixels(bitmapHeight, bitmapWidth, CV_8UC1);
		Peano::GilbertCurve::dither(pixels4b, palette, nearestColorIndex, GetColorIndex, qPixels, nullptr, 1.0f);
		if (m_transparentPixelIndex >= 0)
		{
			auto k = qPixels(m_transparentPixelIndex / bitmapWidth, m_transparentPixelIndex % bitmapWidth);
			if (GetArgb8888(palette.at<Vec4b>(k, 0)) != GetArgb8888(m_transparentColor))
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(1, 0));
		}

		nearestMap.clear();
		ProcessImagePixels(bytes, palette, qPixels, m_transparentPixelIndex >= 0);
		return palette;
	}
}
