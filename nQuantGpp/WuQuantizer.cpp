///////////////////////////////////////////////////////////////////////
//	    C Implementation of Wu's Color Quantizer (v. 2)
//	    (see Graphics Gems vol. II, pp. 126-133)
//
// Author:	Xiaolin Wu
// Dept. of Computer Science
// Univ. of Western Ontario
// London, Ontario N6A 5B7
// wu@csd.uwo.ca
//
// Copyright(c) 2023 Miller Cy Chan
// 
// Algorithm: Greedy orthogonal bipartition of RGB space for variance
// 	   minimization aided by inclusion-exclusion tricks.
// 	   For speed no nearest neighbor search is done. Slightly
// 	   better performance can be expected by more sophisticated
// 	   but more expensive versions.
// 
// Free to distribute, comments and suggestions are appreciated.
///////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "WuQuantizer.h"
#include "bitmapUtilities.h"
#include "BlueNoise.h"
#include <unordered_map>

namespace nQuant
{
	/// <summary><para>Shift color values right this many bits.</para><para>This reduces the granularity of the color maps produced, making it much faster.</para></summary>
	/// 3 = value error of 8 (0 and 7 will look the same to it, 0 and 8 different); Takes ~4MB for color tables; ~.25 -> .50 seconds
	/// 2 = value error of 4; Takes ~64MB for color tables; ~3 seconds
	/// RAM usage roughly estimated with: ( ( 256 >> SidePixShift ) ^ 4 ) * 60
	/// Default SidePixShift = 3
	const uchar SIDEPIXSHIFT = 3;
	const uchar MAXSIDEINDEX = 256 / (1 << SIDEPIXSHIFT);
	const uchar SIDESIZE = MAXSIDEINDEX + 1;
	const uint TOTAL_SIDESIZE = SIDESIZE * SIDESIZE * SIDESIZE * SIDESIZE;

	uchar alphaThreshold = 0xF;
	bool hasSemiTransparency = false;
	int m_transparentPixelIndex = -1;
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);
	double PR = .299, PG = .587, PB = .114;
	unordered_map<ARGB, vector<ushort> > closestMap;
	unordered_map<ARGB, ushort> nearestMap;

	struct Box {
		uchar AlphaMinimum = 0;
		uchar AlphaMaximum = 0;
		uchar RedMinimum = 0;
		uchar RedMaximum = 0;
		uchar GreenMinimum = 0;
		uchar GreenMaximum = 0;
		uchar BlueMinimum = 0;
		uchar BlueMaximum = 0;
		uint Size = 0;
	};

	struct CubeCut {
		bool valid;
		uchar position;
		float value;

		CubeCut(bool isValid, uchar cutPoint, float result) {
			valid = isValid;
			position = cutPoint;
			value = result;
		}
	};

	struct ColorData {
		unique_ptr<long[]> weights;
		unique_ptr<long[]> momentsAlpha;
		unique_ptr<long[]> momentsRed;
		unique_ptr<long[]> momentsGreen;
		unique_ptr<long[]> momentsBlue;
		unique_ptr<float[]> moments;

		vector<Vec4b> pixels;
		unique_ptr<Mat> pImage;

		uint height = 0, width = 0, pixelsCount = 0;
		uint pixelFillingCounter = 0;

		ColorData(uint sideSize, uint bitmapWidth, uint bitmapHeight) {
			const int TOTAL_SIDESIZE = sideSize * sideSize * sideSize * sideSize;
			weights = make_unique<long[]>(TOTAL_SIDESIZE);
			momentsAlpha = make_unique<long[]>(TOTAL_SIDESIZE);
			momentsRed = make_unique<long[]>(TOTAL_SIDESIZE);
			momentsGreen = make_unique<long[]>(TOTAL_SIDESIZE);
			momentsBlue = make_unique<long[]>(TOTAL_SIDESIZE);
			moments = make_unique<float[]>(TOTAL_SIDESIZE);
			height = bitmapHeight;
			width = bitmapWidth;
			pixelsCount = bitmapWidth * bitmapHeight;
			pixels.resize(pixelsCount);
		}

		Mat* GetImage() {
			if (pImage.get() == nullptr) {
				pImage = make_unique<Mat>(pixels);
				auto image = pImage->reshape(0, height);
				image.copyTo(*pImage);
			}
			return pImage.get();
		}

		inline void AddPixel(const Vec4b& pixel)
		{
			pixels[pixelFillingCounter++] = pixel;
		}
	};

	inline uint Index(uchar red, uchar green, uchar blue) {
		return red + green * SIDESIZE + blue * SIDESIZE * SIDESIZE;
	}

	inline uint Index(uchar alpha, uchar red, uchar green, uchar blue) {
		return alpha + red * SIDESIZE + green * SIDESIZE * SIDESIZE + blue * SIDESIZE * SIDESIZE * SIDESIZE;
	}

	inline float Volume(const Box& cube, long* moment)
	{
		return (moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] +
			moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMaximum)] +
			moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] +
			moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)]) -
			(moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]);
	}

	inline float Volume(const Box& cube, float* moment)
	{
		return (moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] +
			moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMaximum)] +
			moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] +
			moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] -
			moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)]) -
			(moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]);
	}

	inline float Top(const Box& cube, Pixel direction, uchar position, long* moment)
	{
		switch (direction)
		{
		case Alpha:
			return (moment[Index(position, cube.RedMaximum, cube.GreenMaximum, cube.BlueMaximum)] -
				moment[Index(position, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] -
				moment[Index(position, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] +
				moment[Index(position, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)]) -
				(moment[Index(position, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] -
					moment[Index(position, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] -
					moment[Index(position, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] +
					moment[Index(position, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]);

		case Red:
			return (moment[Index(cube.AlphaMaximum, position, cube.GreenMaximum, cube.BlueMaximum)] -
				moment[Index(cube.AlphaMaximum, position, cube.GreenMinimum, cube.BlueMaximum)] -
				moment[Index(cube.AlphaMinimum, position, cube.GreenMaximum, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMinimum, position, cube.GreenMinimum, cube.BlueMaximum)]) -
				(moment[Index(cube.AlphaMaximum, position, cube.GreenMaximum, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMaximum, position, cube.GreenMinimum, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMinimum, position, cube.GreenMaximum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, position, cube.GreenMinimum, cube.BlueMinimum)]);

		case Green:
			return (moment[Index(cube.AlphaMaximum, cube.RedMaximum, position, cube.BlueMaximum)] -
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, position, cube.BlueMaximum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMaximum, position, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, position, cube.BlueMaximum)]) -
				(moment[Index(cube.AlphaMaximum, cube.RedMaximum, position, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMaximum, cube.RedMinimum, position, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMinimum, cube.RedMaximum, position, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, position, cube.BlueMinimum)]);

		case Blue:
			return (moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMaximum, position)] -
				moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, position)] -
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, position)] +
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, position)]) -
				(moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, position)] -
					moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, position)] -
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, position)] +
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, position)]);

		default:
			return 0;
		}
	}

	inline float Bottom(const Box& cube, Pixel direction, long* moment)
	{
		switch (direction)
		{
		case Alpha:
			return (-moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)]) -
				(-moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]);

		case Red:
			return (-moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMaximum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)]) -
				(-moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]);

		case Green:
			return (-moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)] +
				moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMaximum)] -
				moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMaximum)]) -
				(-moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]);

		case Blue:
			return (-moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMaximum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] +
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] -
				moment[Index(cube.AlphaMaximum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]) -
				(-moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMaximum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, cube.RedMaximum, cube.GreenMinimum, cube.BlueMinimum)] +
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMaximum, cube.BlueMinimum)] -
					moment[Index(cube.AlphaMinimum, cube.RedMinimum, cube.GreenMinimum, cube.BlueMinimum)]);

		default:
			return 0;
		}
	}	

	void Histogram3d(ColorData& colorData, const Vec4b& color, const uchar alphaThreshold, const uchar alphaFader)
	{
		uchar pixelBlue = color[0];
		uchar pixelGreen = color[1];
		uchar pixelRed = color[2];
		uchar pixelAlpha = color[3];

		uchar indexAlpha = static_cast<uchar>((pixelAlpha >> SIDEPIXSHIFT) + 1);
		uchar indexRed = static_cast<uchar>((pixelRed >> SIDEPIXSHIFT) + 1);
		uchar indexGreen = static_cast<uchar>((pixelGreen >> SIDEPIXSHIFT) + 1);
		uchar indexBlue = static_cast<uchar>((pixelBlue >> SIDEPIXSHIFT) + 1);

		if (pixelAlpha > alphaThreshold) {
			if (pixelAlpha < UCHAR_MAX) {
				short alpha = pixelAlpha + (pixelAlpha % alphaFader);
				pixelAlpha = static_cast<uchar>(alpha > UCHAR_MAX ? UCHAR_MAX : alpha);
				indexAlpha = static_cast<uchar>((pixelAlpha >> 3) + 1);
			}

			const int index = Index(indexAlpha, indexRed, indexGreen, indexBlue);
			if (index < TOTAL_SIDESIZE) {
				colorData.weights[index]++;
				colorData.momentsRed[index] += pixelRed;
				colorData.momentsGreen[index] += pixelGreen;
				colorData.momentsBlue[index] += pixelBlue;
				colorData.momentsAlpha[index] += pixelAlpha;
				colorData.moments[index] += sqr(pixelAlpha) + sqr(pixelRed) + sqr(pixelGreen) + sqr(pixelBlue);
			}
		}		

		colorData.AddPixel(Vec4b(pixelBlue, pixelGreen, pixelRed, pixelAlpha));
	}

	void AdjustMoments(const ColorData& colorData, const uint& nMaxColors)
	{
		vector<int> indices;
		for (int i = 0; i < TOTAL_SIDESIZE; ++i) {
			double d = colorData.weights[i];
			if (d > 0)
				indices.emplace_back(i);
		}

		if (sqr(nMaxColors) / indices.size() < .04)
			return;

		for (const auto& i : indices) {
			double d = colorData.weights[i];
			d = (colorData.weights[i] = sqrt(d)) / d;
			colorData.momentsRed[i] *= d;
			colorData.momentsGreen[i] *= d;
			colorData.momentsBlue[i] *= d;
			colorData.momentsAlpha[i] *= d;
			colorData.moments[i] *= d;
		}
	}

	void BuildHistogram(ColorData& colorData, const Mat srcImg, const uint& nMaxColors, uchar alphaThreshold, uchar alphaFader)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;

		int pixelIndex = 0;
		for (uint y = 0; y < bitmapHeight; ++y) {
			for (uint x = 0; x < bitmapWidth; ++x, ++pixelIndex) {
				Vec4b color;
				if (srcImg.channels() == 4 && color[3] < 0xE0) {
					color = srcImg.at<Vec4b>(y, x);
					if (color[3] == 0) {
						m_transparentPixelIndex = pixelIndex;
						m_transparentColor = color;
					}
					else if (color[3] > alphaThreshold)
						hasSemiTransparency = true;
				}
				else {
					auto& c0 = srcImg.at<Vec3b>(y, x);
					color = Vec4b(c0[0], c0[1], c0[2], UCHAR_MAX);
				}
				Histogram3d(colorData, color, alphaThreshold, alphaFader);
			}
		}

		AdjustMoments(colorData, nMaxColors);
	}

	void CalculateMoments(ColorData& data)
	{
		const uint SIDESIZE_3 = SIDESIZE * SIDESIZE * SIDESIZE;
		for (uchar alphaIndex = 1; alphaIndex <= MAXSIDEINDEX; ++alphaIndex)
		{
			auto xarea = make_unique<size_t[]>(SIDESIZE_3);
			auto xareaAlpha = make_unique<size_t[]>(SIDESIZE_3);
			auto xareaRed = make_unique<size_t[]>(SIDESIZE_3);
			auto xareaGreen = make_unique<size_t[]>(SIDESIZE_3);
			auto xareaBlue = make_unique<size_t[]>(SIDESIZE_3);
			auto xarea2 = make_unique<float[]>(SIDESIZE_3);

			for (uchar redIndex = 1; redIndex <= MAXSIDEINDEX; ++redIndex)
			{
				uint area[SIDESIZE] = { 0 };
				uint areaAlpha[SIDESIZE] = { 0 };
				uint areaRed[SIDESIZE] = { 0 };
				uint areaGreen[SIDESIZE] = { 0 };
				uint areaBlue[SIDESIZE] = { 0 };
				float area2[SIDESIZE] = { 0 };

				for (uchar greenIndex = 1; greenIndex <= MAXSIDEINDEX; ++greenIndex) {
					uint line = 0;
					uint lineAlpha = 0;
					uint lineRed = 0;
					uint lineGreen = 0;
					uint lineBlue = 0;
					float line2 = 0.0f;

					for (uchar blueIndex = 1; blueIndex <= MAXSIDEINDEX; ++blueIndex) {
						const uint index = Index(alphaIndex, redIndex, greenIndex, blueIndex);
						line += data.weights[index];
						lineAlpha += data.momentsAlpha[index];
						lineRed += data.momentsRed[index];
						lineGreen += data.momentsGreen[index];
						lineBlue += data.momentsBlue[index];
						line2 += data.moments[index];

						area[blueIndex] += line;
						areaAlpha[blueIndex] += lineAlpha;
						areaRed[blueIndex] += lineRed;
						areaGreen[blueIndex] += lineGreen;
						areaBlue[blueIndex] += lineBlue;
						area2[blueIndex] += line2;

						const uint rgbIndex = Index(redIndex, greenIndex, blueIndex);
						const uint prevRgbIndex = Index(redIndex - 1, greenIndex, blueIndex);
						xarea[rgbIndex] = xarea[prevRgbIndex] + area[blueIndex];
						xareaAlpha[rgbIndex] = xareaAlpha[prevRgbIndex] + areaAlpha[blueIndex];
						xareaRed[rgbIndex] = xareaRed[prevRgbIndex] + areaRed[blueIndex];
						xareaGreen[rgbIndex] = xareaGreen[prevRgbIndex] + areaGreen[blueIndex];
						xareaBlue[rgbIndex] = xareaBlue[prevRgbIndex] + areaBlue[blueIndex];
						xarea2[rgbIndex] = xarea2[prevRgbIndex] + area2[blueIndex];

						const uint prevIndex = Index(alphaIndex - 1, redIndex, greenIndex, blueIndex);
						data.weights[index] = data.weights[prevIndex] + xarea[rgbIndex];
						data.momentsAlpha[index] = data.momentsAlpha[prevIndex] + xareaAlpha[rgbIndex];
						data.momentsRed[index] = data.momentsRed[prevIndex] + xareaRed[rgbIndex];
						data.momentsGreen[index] = data.momentsGreen[prevIndex] + xareaGreen[rgbIndex];
						data.momentsBlue[index] = data.momentsBlue[prevIndex] + xareaBlue[rgbIndex];
						data.moments[index] = data.moments[prevIndex] + xarea2[rgbIndex];
					}
				}
			}
		}
	}

	CubeCut Maximize(const ColorData& data, const Box& cube, Pixel direction, uchar first, uchar last, uint wholeAlpha, uint wholeRed, uint wholeGreen, uint wholeBlue, uint wholeWeight)
	{
		auto bottomAlpha = Bottom(cube, direction, data.momentsAlpha.get());
		auto bottomRed = Bottom(cube, direction, data.momentsRed.get());
		auto bottomGreen = Bottom(cube, direction, data.momentsGreen.get());
		auto bottomBlue = Bottom(cube, direction, data.momentsBlue.get());
		auto bottomWeight = Bottom(cube, direction, data.weights.get());

		bool valid = false;
		auto result = 0.0f;
		uchar cutPoint = 0;

		for (int position = first; position < last; ++position)
		{
			auto halfAlpha = bottomAlpha + Top(cube, direction, position, data.momentsAlpha.get());
			auto halfRed = bottomRed + Top(cube, direction, position, data.momentsRed.get());
			auto halfGreen = bottomGreen + Top(cube, direction, position, data.momentsGreen.get());
			auto halfBlue = bottomBlue + Top(cube, direction, position, data.momentsBlue.get());
			auto halfWeight = bottomWeight + Top(cube, direction, position, data.weights.get());

			if (halfWeight == 0)
				continue;

			auto halfDistance = sqr(halfAlpha) + sqr(halfRed) + sqr(halfGreen) + sqr(halfBlue);
			auto temp = halfDistance / halfWeight;

			halfAlpha = wholeAlpha - halfAlpha;
			halfRed = wholeRed - halfRed;
			halfGreen = wholeGreen - halfGreen;
			halfBlue = wholeBlue - halfBlue;
			halfWeight = wholeWeight - halfWeight;

			if (halfWeight != 0) {
				halfDistance = sqr(halfAlpha) + sqr(halfRed) + sqr(halfGreen) + sqr(halfBlue);
				temp += halfDistance / halfWeight;

				if (temp > result) {
					valid = true;
					result = temp;
					cutPoint = position;
				}
			}
		}

		return CubeCut(valid, cutPoint, result);
	}

	bool Cut(const ColorData& data, Box& first, Box& second)
	{
		auto wholeAlpha = Volume(first, data.momentsAlpha.get());
		auto wholeRed = Volume(first, data.momentsRed.get());
		auto wholeGreen = Volume(first, data.momentsGreen.get());
		auto wholeBlue = Volume(first, data.momentsBlue.get());
		auto wholeWeight = Volume(first, data.weights.get());

		auto maxAlpha = Maximize(data, first, Alpha, static_cast<uchar>(first.AlphaMinimum + 1), first.AlphaMaximum, wholeAlpha, wholeRed, wholeGreen, wholeBlue, wholeWeight);
		auto maxRed = Maximize(data, first, Red, static_cast<uchar>(first.RedMinimum + 1), first.RedMaximum, wholeAlpha, wholeRed, wholeGreen, wholeBlue, wholeWeight);
		auto maxGreen = Maximize(data, first, Green, static_cast<uchar>(first.GreenMinimum + 1), first.GreenMaximum, wholeAlpha, wholeRed, wholeGreen, wholeBlue, wholeWeight);
		auto maxBlue = Maximize(data, first, Blue, static_cast<uchar>(first.BlueMinimum + 1), first.BlueMaximum, wholeAlpha, wholeRed, wholeGreen, wholeBlue, wholeWeight);

		Pixel direction = Blue;
		if ((maxAlpha.value >= maxRed.value) && (maxAlpha.value >= maxGreen.value) && (maxAlpha.value >= maxBlue.value)) {
			if (!maxAlpha.valid)
				return false;
			direction = Alpha;
		}
		else if ((maxRed.value >= maxAlpha.value) && (maxRed.value >= maxGreen.value) && (maxRed.value >= maxBlue.value))
			direction = Red;
		else if ((maxGreen.value >= maxAlpha.value) && (maxGreen.value >= maxRed.value) && (maxGreen.value >= maxBlue.value))
			direction = Green;

		second.AlphaMaximum = first.AlphaMaximum;
		second.RedMaximum = first.RedMaximum;
		second.GreenMaximum = first.GreenMaximum;
		second.BlueMaximum = first.BlueMaximum;

		switch (direction)
		{
		case Alpha:
			second.AlphaMinimum = first.AlphaMaximum = maxAlpha.position;
			second.RedMinimum = first.RedMinimum;
			second.GreenMinimum = first.GreenMinimum;
			second.BlueMinimum = first.BlueMinimum;
			break;

		case Red:
			second.RedMinimum = first.RedMaximum = maxRed.position;
			second.AlphaMinimum = first.AlphaMinimum;
			second.GreenMinimum = first.GreenMinimum;
			second.BlueMinimum = first.BlueMinimum;
			break;

		case Green:
			second.GreenMinimum = first.GreenMaximum = maxGreen.position;
			second.AlphaMinimum = first.AlphaMinimum;
			second.RedMinimum = first.RedMinimum;
			second.BlueMinimum = first.BlueMinimum;
			break;

		case Blue:
			second.BlueMinimum = first.BlueMaximum = maxBlue.position;
			second.AlphaMinimum = first.AlphaMinimum;
			second.RedMinimum = first.RedMinimum;
			second.GreenMinimum = first.GreenMinimum;
			break;
		}

		first.Size = (first.AlphaMaximum - first.AlphaMinimum) * (first.RedMaximum - first.RedMinimum) * (first.GreenMaximum - first.GreenMinimum) * (first.BlueMaximum - first.BlueMinimum);
		second.Size = (second.AlphaMaximum - second.AlphaMinimum) * (second.RedMaximum - second.RedMinimum) * (second.GreenMaximum - second.GreenMinimum) * (second.BlueMaximum - second.BlueMinimum);

		return true;
	}

	float CalculateVariance(const ColorData& data, const Box& cube)
	{
		auto volumeAlpha = Volume(cube, data.momentsAlpha.get());
		auto volumeRed = Volume(cube, data.momentsRed.get());
		auto volumeGreen = Volume(cube, data.momentsGreen.get());
		auto volumeBlue = Volume(cube, data.momentsBlue.get());
		auto volumeMoment = Volume(cube, data.moments.get());
		auto volumeWeight = Volume(cube, data.weights.get());

		float distance = sqr(volumeAlpha) + sqr(volumeRed) + sqr(volumeGreen) + sqr(volumeBlue);

		return volumeWeight != 0.0f ? (volumeMoment - distance / volumeWeight) : 0.0f;
	}

	void SplitData(vector<Box>& boxList, uint& colorCount, ColorData& data)
	{
		int next = 0;
		auto volumeVariance = make_unique<float[]>(colorCount);
		boxList.resize(colorCount);
		boxList[0].AlphaMaximum = MAXSIDEINDEX;
		boxList[0].RedMaximum = MAXSIDEINDEX;
		boxList[0].GreenMaximum = MAXSIDEINDEX;
		boxList[0].BlueMaximum = MAXSIDEINDEX;

		for (int cubeIndex = 1; cubeIndex < colorCount; ++cubeIndex) {
			if (Cut(data, boxList[next], boxList[cubeIndex])) {
				volumeVariance[next] = boxList[next].Size > 1 ? CalculateVariance(data, boxList[next]) : 0.0f;
				volumeVariance[cubeIndex] = boxList[cubeIndex].Size > 1 ? CalculateVariance(data, boxList[cubeIndex]) : 0.0f;
			}
			else {
				volumeVariance[next] = 0.0f;
				--cubeIndex;
			}

			next = 0;
			auto temp = volumeVariance[0];

			for (int index = 1; index <= cubeIndex; ++index) {
				if (volumeVariance[index] <= temp)
					continue;
				temp = volumeVariance[index];
				next = index;
			}

			if (temp > 0.0f)
				continue;

			colorCount = cubeIndex + 1;
			break;
		}
		boxList.resize(colorCount);
	}

	void BuildLookups(Mat palette, vector<Box>& cubes, const ColorData& data)
	{
		uint lookupsCount = 0;
		if (m_transparentPixelIndex >= 0)
			palette.at<Vec4b>(lookupsCount++, 0) = m_transparentColor;
			
		for (auto const& cube : cubes) {
			auto weight = Volume(cube, data.weights.get());

			if (weight <= 0)
				continue;

			auto alpha = static_cast<uchar>(Volume(cube, data.momentsAlpha.get()) / weight);
			auto red = static_cast<uchar>(Volume(cube, data.momentsRed.get()) / weight);
			auto green = static_cast<uchar>(Volume(cube, data.momentsGreen.get()) / weight);
			auto blue = static_cast<uchar>(Volume(cube, data.momentsBlue.get()) / weight);
			if (palette.channels() == 4)
				palette.at<Vec4b>(lookupsCount++, 0) = Vec4b(blue, green, red, alpha);
			else
				palette.at<Vec3b>(lookupsCount++, 0) = Vec3b(blue, green, red);
		}

		if(lookupsCount < (palette.rows * palette.cols))
			palette = palette.rowRange(0, lookupsCount);
	}

	ushort closestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
	{
		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			c = m_transparentColor;

		const auto nMaxColors = palette.rows;
		if (nMaxColors > 2 && m_transparentPixelIndex >= 0 && c[3] > alphaThreshold)
			k = 1;

		auto argb = GetArgb8888(c0);

		vector<ushort> closest(5);
		auto got = closestMap.find(argb);
		if (got == closestMap.end()) {
			closest[2] = closest[3] = SHRT_MAX;

			for (; k < nMaxColors; ++k) {
				Vec4b c2;
				GrabPixel(c2, palette, k, 0);
				closest[4] = abs(c[3] - c2[3]) + abs(c[2] - c2[2]) + abs(c[1] - c2[1]) + abs(c[0] - c2[0]);
				if (closest[4] < closest[2]) {
					closest[1] = closest[0];
					closest[3] = closest[2];
					closest[0] = k;
					closest[2] = closest[4];
				}
				else if (closest[4] < closest[3]) {
					closest[1] = k;
					closest[3] = closest[4];
				}
			}

			if (closest[3] == SHRT_MAX)
				closest[2] = 0;
		}
		else
			closest = got->second;

		if (closest[2] == 0 || (rand() % (closest[3] + closest[2])) <= closest[3])
			k = closest[0];
		else
			k = closest[1];

		closestMap[argb] = closest;
		return k;
	}

	ushort nearestColorIndex(const Mat palette, const Vec4b& c0, const uchar alphaThreshold)
	{
		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			c = m_transparentColor;
		
		const auto nMaxColors = palette.rows;
		if (nMaxColors > 2 && m_transparentPixelIndex >= 0 && c[3] > alphaThreshold)
			k = 1;

		auto argb = GetArgb8888(c0);
		auto got = nearestMap.find(argb);
		if (got == nearestMap.end()) {
			double mindist = SHRT_MAX;
			for (uint i = k; i < palette.rows; i++) {
				Vec4b c2;
				GrabPixel(c2, palette, i, 0);
				double curdist = sqr(c2[3] - c[3]);
				if (curdist > mindist)
					continue;

				curdist += PR * sqr(c2[2] - c[2]);
				if (curdist > mindist)
					continue;

				curdist += PG * sqr(c2[1] - c[1]);
				if (curdist > mindist)
					continue;

				curdist += PB * sqr(c2[0] - c[0]);
				if (curdist > mindist)
					continue;

				mindist = curdist;
				k = i;
			}

			nearestMap[argb] = k;
		}
		else
			k = got->second;

		return k;
	}

	void GetQuantizedPalette(const ColorData& data, Mat palette, const uint colorCount, const uchar alphaThreshold)
	{
		auto alphas = make_unique<size_t[]>(colorCount);
		auto reds = make_unique<size_t[]>(colorCount);
		auto greens = make_unique<size_t[]>(colorCount);
		auto blues = make_unique<size_t[]>(colorCount);
		auto sums = make_unique<size_t[]>(colorCount);

		int pixelsCount = data.pixelsCount;

		for (uint pixelIndex = 0; pixelIndex < pixelsCount; ++pixelIndex) {
			auto pixel = data.pixels[pixelIndex];
			if (pixel[3] <= alphaThreshold)
				pixel = m_transparentColor;

			uint bestMatch = nearestColorIndex(palette, pixel, alphaThreshold);

			alphas[bestMatch] += pixel[3];
			reds[bestMatch] += pixel[2];
			greens[bestMatch] += pixel[1];
			blues[bestMatch] += pixel[0];
			sums[bestMatch]++;
		}
		nearestMap.clear();

		short paletteIndex = (m_transparentPixelIndex < 0) ? 0 : 1;
		for (; paletteIndex < colorCount; ++paletteIndex) {
			if (sums[paletteIndex] > 0) {
				alphas[paletteIndex] /= sums[paletteIndex];
				reds[paletteIndex] /= sums[paletteIndex];
				greens[paletteIndex] /= sums[paletteIndex];
				blues[paletteIndex] /= sums[paletteIndex];
			}

			if(palette.channels() == 4)
				palette.at<Vec4b>(paletteIndex, 0) = Vec4b(blues[paletteIndex], greens[paletteIndex], reds[paletteIndex], alphas[paletteIndex]);
			else
				palette.at<Vec3b>(paletteIndex, 0) = Vec3b(blues[paletteIndex], greens[paletteIndex], reds[paletteIndex]);
		}
	}

	inline auto GetColorIndex(const Vec4b& c)
	{
		return GetArgbIndex(c, hasSemiTransparency, m_transparentPixelIndex >= 0);
	}

	bool quantize_image(const Mat4b pixels, const Mat palette, Mat1b qPixels, const bool dither, uchar alphaThreshold)
	{
		auto width = pixels.cols;
		auto height = pixels.rows;
		if (dither) {
			uint pixelIndex = 0;

			const int DJ = 4;
			const int BLOCK_SIZE = 256;
			const int DITHER_MAX = 20;
			const int err_len = (width + 2) * DJ;
			auto clamp = make_unique <uchar[]>(DJ * BLOCK_SIZE);
			auto erowErr = make_unique<short[]>(err_len);
			auto orowErr = make_unique<short[]>(err_len);
			auto limtb = make_unique<char[]>(2 * BLOCK_SIZE);
			auto pDitherPixel = make_unique<int[]>(DJ);

			for (int i = 0; i < BLOCK_SIZE; ++i) {
				clamp[i] = 0;
				clamp[i + BLOCK_SIZE] = static_cast<uchar>(i);
				clamp[i + BLOCK_SIZE * 2] = UCHAR_MAX;
				clamp[i + BLOCK_SIZE * 3] = UCHAR_MAX;

				limtb[i] = -DITHER_MAX;
				limtb[i + BLOCK_SIZE] = DITHER_MAX;
			}
			for (int i = -DITHER_MAX; i <= DITHER_MAX; i++)
				limtb[i + BLOCK_SIZE] = i;

			auto row0 = erowErr.get();
			auto row1 = orowErr.get();

			bool noBias = (m_transparentPixelIndex >= 0 || hasSemiTransparency) || palette.rows < 64;
			int dir = 1;
			for (int i = 0; i < height; ++i) {
				if (dir < 0)
					pixelIndex += width - 1;

				int cursor0 = DJ, cursor1 = width * DJ;
				row1[cursor1] = row1[cursor1 + 1] = row1[cursor1 + 2] = row1[cursor1 + 3] = 0;
				for (uint j = 0; j < width; ++j) {
					int y = pixelIndex / width, x = pixelIndex % width;
					Vec4b pixel;
					GrabPixel(pixel, pixels, y, x);

					CalcDitherPixel(pDitherPixel.get(), pixel, clamp.get(), row0, cursor0, noBias);
					int b_pix = pDitherPixel[0];
					int g_pix = pDitherPixel[1];
					int r_pix = pDitherPixel[2];
					int a_pix = pDitherPixel[3];
					Vec4b c1(b_pix, g_pix, r_pix, a_pix);
					auto& qPixelIndex = qPixels(y, x);
					qPixelIndex = nearestColorIndex(palette, c1, alphaThreshold);

					Vec4b c2;
					GrabPixel(c2, palette, qPixelIndex, 0);

					b_pix = limtb[c1[0] - c2[0] + BLOCK_SIZE];
					g_pix = limtb[c1[1] - c2[1] + BLOCK_SIZE];
					r_pix = limtb[c1[2] - c2[2] + BLOCK_SIZE];
					a_pix = limtb[c1[3] - c2[3] + BLOCK_SIZE];

					int k = r_pix * 2;
					row1[cursor1 - DJ] = r_pix;
					row1[cursor1 + DJ] += (r_pix += k);
					row1[cursor1] += (r_pix += k);
					row0[cursor0 + DJ] += (r_pix + k);

					k = g_pix * 2;
					row1[cursor1 + 1 - DJ] = g_pix;
					row1[cursor1 + 1 + DJ] += (g_pix += k);
					row1[cursor1 + 1] += (g_pix += k);
					row0[cursor0 + 1 + DJ] += (g_pix + k);

					k = b_pix * 2;
					row1[cursor1 + 2 - DJ] = b_pix;
					row1[cursor1 + 2 + DJ] += (b_pix += k);
					row1[cursor1 + 2] += (b_pix += k);
					row0[cursor0 + 2 + DJ] += (b_pix + k);

					k = a_pix * 2;
					row1[cursor1 + 3 - DJ] = a_pix;
					row1[cursor1 + 3 + DJ] += (a_pix += k);
					row1[cursor1 + 3] += (a_pix += k);
					row0[cursor0 + 3 + DJ] += (a_pix + k);

					cursor0 += DJ;
					cursor1 -= DJ;
					pixelIndex += dir;
				}
				if ((i % 2) == 1)
					pixelIndex += width + 1;

				dir *= -1;
				swap(row0, row1);
			}
			return true;
		}

		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				Vec4b pixel;
				GrabPixel(pixel, pixels, j, i);
				qPixels(j, i) = (uchar) closestColorIndex(palette, pixel, i + j);
			}
		}

		BlueNoise::dither(pixels, palette, closestColorIndex, GetColorIndex, qPixels);
		return true;
	}
	
	Mat WuQuantizer::QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither, uchar alphaThreshold, uchar alphaFader)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);

		Mat palette(nMaxColors, 1, srcImg.type(), scalar);
		
		if (nMaxColors <= 32)
			PR = PG = PB = 1;

		Mat1b qPixels(bitmapHeight, bitmapWidth);
		if (nMaxColors > 2) {
			ColorData colorData(SIDESIZE, bitmapWidth, bitmapHeight);
			BuildHistogram(colorData, srcImg, nMaxColors, alphaThreshold, alphaFader);
			CalculateMoments(colorData);
			vector<Box> cubes;
			SplitData(cubes, nMaxColors, colorData);

			BuildLookups(palette, cubes, colorData);
			cubes.clear();

			nMaxColors = palette.rows;

			GetQuantizedPalette(colorData, palette, nMaxColors, alphaThreshold);
			if (nMaxColors > 256) {
				Mat qPixels(bitmapHeight, bitmapWidth, srcImg.type());
				dithering_image(*colorData.GetImage(), palette, closestColorIndex, hasSemiTransparency, m_transparentPixelIndex, nMaxColors, qPixels);
				return qPixels;
			}
			quantize_image(*colorData.GetImage(), palette, qPixels, dither, alphaThreshold);
		}
		else {
			Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
			GrabPixels(srcImg, pixels4b, hasSemiTransparency, m_transparentPixelIndex, m_transparentColor, 0xF, nMaxColors);
			if (m_transparentPixelIndex >= 0)
				palette.at<Vec4b>(0, 0) = m_transparentColor;
			else
				palette.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);
			quantize_image(pixels4b, palette, qPixels, dither, alphaThreshold);
		}
		
		if (m_transparentPixelIndex >= 0) {
			auto k = qPixels(m_transparentPixelIndex / bitmapWidth, m_transparentPixelIndex % bitmapWidth);
			if (nMaxColors > 2)
				palette.at<Vec4b>(k, 0) = m_transparentColor;
			else if (GetArgb8888(palette.at<Vec4b>(k, 0)) != GetArgb8888(m_transparentColor))
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(1, 0));
		}
		closestMap.clear();
		nearestMap.clear();

		ProcessImagePixels(bytes, palette, qPixels, m_transparentPixelIndex >= 0);
		return palette;
	}

}
