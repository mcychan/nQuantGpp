/* Otsu's Image Segmentation Method
  Copyright (C) 2009 Tolga Birdal
  Copyright (c) 2023 - 2024 Miller Cy Chan
*/

#include "stdafx.h"
#include "Otsu.h"
#include "bitmapUtilities.h"
#include "GilbertCurve.h"
#define _USE_MATH_DEFINES
#include <math.h>
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
		short idx = 0;

		for (int i = 1; i < n - 1; ++i) {
			if (vec[i] > maxVec) {
				maxVec = vec[i];
				idx = (short) i;
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
	
	void threshold(const Mat4b pixels, Mat4b dest, short thresh, float weight = 1.0f)
	{
		auto maxThresh = (uchar)thresh;
		if (thresh >= 200)
		{
			weight = m_transparentPixelIndex >= 0 ? .9f : .8f;
			maxThresh = (uchar)(thresh * weight);
			thresh = 200;
		}

		auto minThresh = (uchar)(thresh * (m_transparentPixelIndex >= 0 ? .9f : weight));
		for (uint y = 0; y < pixels.rows; ++y)
		{
			for (uint x = 0; x < pixels.cols; ++x)
			{
				auto& d = dest(y, x);
				const auto& c = pixels(y, x);
				
				if (c[3] < alphaThreshold && c[2] + c[1] + c[0] > maxThresh * 3)
					d = Vec4b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, c[3]);
				else if (c[2] + c[1] + c[0] < minThresh * 3)
					d = Vec4b(0, 0, 0, c[3]);
			}
		}
	}

	Mat cannyFilter(const Mat4b pixelsGray, double lowerThreshold, double higherThreshold) {
		const auto width = pixelsGray.cols;
		const auto height = pixelsGray.rows;
		const auto area = (size_t)(width * height);
		auto scalar = Scalar(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);

		Mat4b pixelsCanny(height, width, scalar);

		int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
		int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
		auto G = make_unique<double[]>(area);
		vector<int> theta(area);
		auto largestG = 0.0;

		// perform canny edge detection on everything but the edges
		for (int i = 1; i < height - 1; ++i) {
			for (int j = 1; j < width - 1; ++j) {
				// find gx and gy for each pixel
				auto gxValue = 0.0;
				auto gyValue = 0.0;
				for (int x = -1; x <= 1; ++x) {
					for (int y = -1; y <= 1; ++y) {
						const auto& c = pixelsGray(i + x, j + y);
						gxValue += gx[1 - x][1 - y] * c[0];
						gyValue += gy[1 - x][1 - y] * c[0];
					}
				}

				const int center = i * width + j;
				// calculate G and theta
				G[center] = sqrt(pow(gxValue, 2) + pow(gyValue, 2));
				auto atanResult = atan2(gyValue, gxValue) * 180.0 / M_PI;
				theta[center] = (int)(180.0 + atanResult);

				if (G[center] > largestG)
					largestG = G[center];

				// setting the edges
				if (i == 1) {
					G[center - 1] = G[center];
					theta[center - 1] = theta[center];
				}
				else if (j == 1) {
					G[center - width] = G[center];
					theta[center - width] = theta[center];
				}
				else if (i == height - 1) {
					G[center + 1] = G[center];
					theta[center + 1] = theta[center];
				}
				else if (j == width - 1) {
					G[center + width] = G[center];
					theta[center + width] = theta[center];
				}

				// setting the corners
				if (i == 1 && j == 1) {
					G[center - width - 1] = G[center];
					theta[center - width - 1] = theta[center];
				}
				else if (i == 1 && j == width - 1) {
					G[center - width + 1] = G[center];
					theta[center - width + 1] = theta[center];
				}
				else if (i == height - 1 && j == 1) {
					G[center + width - 1] = G[center];
					theta[center + width - 1] = theta[center];
				}
				else if (i == height - 1 && j == width - 1) {
					G[center + width + 1] = G[center];
					theta[center + width + 1] = theta[center];
				}

				// to the nearest 45 degrees
				theta[center] = rint(theta[center] / 45) * 45;
			}
		}

		largestG *= .5;

		// non-maximum suppression
		for (int i = 1; i < height - 1; ++i) {
			for (int j = 1; j < width - 1; ++j) {
				auto& pixel = pixelsCanny(i, j);
				const int center = i * width + j;
				if (theta[center] == 0 || theta[center] == 180) {
					if (G[center] < G[center - 1] || G[center] < G[center + 1])
						G[center] = 0;
				}
				else if (theta[center] == 45 || theta[center] == 225) {
					if (G[center] < G[center + width + 1] || G[center] < G[center - width - 1])
						G[center] = 0;
				}
				else if (theta[center] == 90 || theta[center] == 270) {
					if (G[center] < G[center + width] || G[center] < G[center - width])
						G[center] = 0;
				}
				else {
					if (G[center] < G[center + width - 1] || G[center] < G[center - width + 1])
						G[center] = 0;
				}

				auto grey = (uchar)(G[center] * (255.0 / largestG));
				pixel[0] = pixel[1] = pixel[2] = ~grey;
			}
		}

		int k = 0;
		auto minThreshold = lowerThreshold * largestG, maxThreshold = higherThreshold * largestG;
		do {
			for (int i = 1; i < height - 1; ++i) {
				for (int j = 1; j < width - 1; ++j) {
					auto& pixel = pixelsCanny(i, j);
					const int center = i * width + j;
					if (G[center] < minThreshold)
						G[center] = 0;
					else if (G[center] >= maxThreshold)
						continue;
					else if (G[center] < maxThreshold) {
						G[center] = 0;
						for (int x = -1; x <= 1; ++x) {
							for (int y = -1; y <= 1; y++) {
								if (x == 0 && y == 0)
									continue;
								if (G[center + x * width + y] >= maxThreshold) {
									G[center] = higherThreshold * largestG;
									k = 0;
									x = 2;
									break;
								}
							}
						}
					}
					
					auto grey = (uchar)(G[center] * 255.0 / largestG);
					pixel[0] = pixel[1] = pixel[2] = ~grey;
				}
			}
		} while (k++ < 100);
		return pixelsCanny;
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
		auto width = srcImg.cols;
		auto height = srcImg.rows;
		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);

		Mat4b pixels4b(height, width, Scalar(0, 0, 0, UCHAR_MAX));
		GrabPixels(srcImg, pixels4b, hasSemiTransparency, m_transparentPixelIndex, m_transparentColor, alphaThreshold);

		auto pixelsGray = pixels4b.clone();
		if (!isGrayscale)
			ConvertToGrayScale(pixels4b, pixelsGray);
		vector<Vec4b> pixels(pixelsGray.begin(), pixelsGray.end());

		auto otsuThreshold = getOtsuThreshold(pixels);
		auto lowerThreshold = 0.03, higherThreshold = 0.1;
		pixels4b = cannyFilter(pixelsGray, lowerThreshold, higherThreshold);
		threshold(pixelsGray, pixels4b, otsuThreshold);

		Mat palette(2, 1, srcImg.type(), scalar);
		if (srcImg.channels() == 4)
			palette.at<Vec4b>(0, 0) = m_transparentColor;
		else
			palette.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);

		Mat1b qPixels(height, width, CV_8UC1);
		Peano::GilbertCurve::dither(pixels4b, palette, nearestColorIndex, GetColorIndex, qPixels, nullptr, 1.0f);
		if (m_transparentPixelIndex >= 0) {
			auto k = qPixels(m_transparentPixelIndex / width, m_transparentPixelIndex % width);
			if (GetArgb8888(palette.at<Vec4b>(k, 0)) != GetArgb8888(m_transparentColor))
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(1, 0));
		}

		nearestMap.clear();
		ProcessImagePixels(bytes, palette, qPixels, m_transparentPixelIndex >= 0);
		return palette;
	}
}
