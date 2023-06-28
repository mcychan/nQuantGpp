/* Fast pairwise nearest neighbor based genetic algorithm with CIELAB color space genetic algorithm
* Copyright (c) 2023 Miller Cy Chan */

#include "stdafx.h"
#include "PnnLABGAQuantizer.h"
#include "CIELABConvertor.h"
#include "BlueNoise.h"

#include <numeric>
#include <shared_mutex>
#include <unordered_map>
#include <random>
#include <iomanip>

namespace PnnLABQuant
{	
	int _bitmapWidth, _dp = 1, _type = 0;
	uint _nMaxColors = 256;
	double minRatio = 0, maxRatio = 1.0;

	static unordered_map<string, vector<double> > _fitnessMap;
	static shared_mutex _mutex;

	PnnLABGAQuantizer::PnnLABGAQuantizer(PnnLABQuantizer& pq, Mat srcImg, uint nMaxColors) {
		// increment value when criteria violation occurs
		_objectives.resize(4);
		_bitmapWidth = srcImg.cols;
		srand(_bitmapWidth * srcImg.rows);
		
		m_pq = make_unique<PnnLABQuantizer>(pq);
		if(pq.IsGA())
			return;

		_nMaxColors = nMaxColors;	

		bool hasSemiTransparency = false;
		m_pixels = make_shared<Mat4b>(srcImg.rows, _bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX)); 
		m_pq->grabPixels(srcImg, *m_pixels, _nMaxColors, hasSemiTransparency);
		_type = srcImg.type();
		minRatio = (hasSemiTransparency || nMaxColors < 64) ? .01 : .85;
		maxRatio = min(1.0, nMaxColors / ((nMaxColors < 64) ? 500.0 : 50.0));
		_dp = maxRatio < .1 ? 1000 : 100;
	}

	PnnLABGAQuantizer::PnnLABGAQuantizer(PnnLABQuantizer& pq, const shared_ptr<Mat4b> pixels, int bitmapWidth, uint nMaxColors)
	{
		m_pq = make_unique<PnnLABQuantizer>(pq);
		// increment value when criteria violation occurs
		_objectives.resize(4);
		m_pixels = pixels;
		_bitmapWidth = bitmapWidth;
		srand(pixels->rows * pixels->cols);
		_nMaxColors = nMaxColors;
	}

	auto PnnLABGAQuantizer::findByRatioKey() const
	{
		auto ratioX = (int)(_ratioX * _dp);
		auto ratioY = (int)(_ratioY * _dp);

		ostringstream ss;
		auto difference = abs(ratioX - ratioY);
		if (difference <= 0.0000001)
			ss << ratioX;
		else
			ss << ratioX << ";" << ratioY;

		auto ratioKey = ss.str();
		shared_lock<shared_mutex> lock(_mutex);
		return _fitnessMap.find(ratioKey);
	}

	void PnnLABGAQuantizer::calculateFitness() {
		auto got = findByRatioKey();
		if (got != _fitnessMap.end()) {
			_objectives = got->second;
			_fitness = -1.0f * (float) accumulate(_objectives.begin(), _objectives.end(), 0);
			return;
		}

		_objectives.resize(4);
		m_pq->setRatio(_ratioX, _ratioY);
		
		auto scalar = m_pq->hasAlpha() ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);
		auto palette = make_shared<Mat>(_nMaxColors, 1, _type, scalar);
		m_pq->pnnquan(*m_pixels, *palette, _nMaxColors);
		m_pq->setPalette(*palette);

		auto& errors = _objectives;
		fill(errors.begin(), errors.end(), 0);

		int threshold = maxRatio < .1 ? -64 : -112;
		int pixelIndex = 0;
		for (int y = 0; y < m_pixels->rows; ++y)
		{
			for (int x = 0; x < m_pixels->cols; ++x, ++pixelIndex)
			{
				if(BlueNoise::RAW_BLUE_NOISE[pixelIndex & 4095] > threshold)
					continue;

				auto c = m_pixels->at<Vec4b>(y, x);
				CIELABConvertor::Lab lab1, lab2;
				m_pq->GetLab(c, lab1);
				auto qPixelIndex = m_pq->nearestColorIndex(*palette, c, pixelIndex);
				Vec4b c2;
				GrabPixel(c2, *palette, qPixelIndex, 0);
				m_pq->GetLab(c2, lab2);

				if (m_pq->hasAlpha()) {
					errors[0] += sqr(lab2.L - lab1.L);
					errors[1] += sqr(lab2.A - lab1.A);
					errors[2] += sqr(lab2.B - lab1.B);
					errors[3] += sqr(lab2.alpha - lab1.alpha) / exp(1.5);
				}
				else {
					errors[0] += abs(lab2.L - lab1.L);
					errors[1] += sqrt(sqr(lab2.A - lab1.A) + sqr(lab2.B - lab1.B));
				}
			}
		}
		
		_fitness = -1.0f * (float) accumulate(_objectives.begin(), _objectives.end(), 0);
		unique_lock<shared_mutex> lock(_mutex);
		_fitnessMap.insert({ ratioKey, _objectives });
	}
	
	Mat PnnLABGAQuantizer::QuantizeImage(vector<uchar>& bytes, bool dither) {
		m_pq->setRatio(_ratioX, _ratioY);
		auto scalar = m_pq->hasAlpha() ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);
		auto palette = make_shared<Mat>(_nMaxColors, 1, _type, scalar);

		m_pq->pnnquan(*m_pixels, *palette, _nMaxColors);
		m_pq->setPalette(*palette);
		return m_pq->QuantizeImage(*m_pixels, *palette, bytes, _nMaxColors, dither);
	}

	PnnLABGAQuantizer::~PnnLABGAQuantizer() {
		unique_lock<shared_mutex> lock(_mutex);
		_fitnessMap.clear();
	}

	double randrange(double min, double max)
	{
		auto f = (double) rand() / RAND_MAX;
		return min + f * (max - min);
	}
	
	void PnnLABGAQuantizer::setRatio(double ratioX, double ratioY)
	{
		_ratioX = min(max(ratioX, minRatio), maxRatio);
		_ratioY = min(max(ratioY, minRatio), maxRatio);
	}

	float PnnLABGAQuantizer::getFitness() {
		return _fitness;
	}

	shared_ptr<PnnLABGAQuantizer> PnnLABGAQuantizer::crossover(const PnnLABGAQuantizer& mother, int numberOfCrossoverPoints, float crossoverProbability)
	{
		auto child = makeNewFromPrototype();
		if ((rand() % 100) <= crossoverProbability)
			return child;
		
		auto ratioX = sqrt(_ratioX * mother._ratioY);
		auto ratioY = sqrt(_ratioY * mother._ratioX);
		child->setRatio(ratioX, ratioY);
		child->calculateFitness();
		return child;
	}

	bool PnnLABGAQuantizer::dominates(const PnnLABGAQuantizer* right) {
		bool better = false;
		for (int f = 0; f < getObjectives().size(); ++f) {
			if (getObjectives()[f] > right->getObjectives()[f])
				return false;

			if (getObjectives()[f] < right->getObjectives()[f])
				better = true;
		}
		return better;
	}

	void PnnLABGAQuantizer::mutation(int mutationSize, float mutationProbability) {
		// check probability of mutation operation
		if ((rand() % 100) > mutationProbability)
			return;

		auto ratioX = _ratioX;
		auto ratioY = _ratioY;
		if(randrange(.0, 1.0) > .5)
			ratioX = .5 * (ratioX + randrange(minRatio, maxRatio));
		else
			ratioY = .5 * (ratioY + randrange(minRatio, maxRatio));

		setRatio(ratioX, ratioY);
		calculateFitness();
	}

	vector<double> PnnLABGAQuantizer::getObjectives() const
	{
		return _objectives;
	}

	vector<double>& PnnLABGAQuantizer::getConvertedObjectives()
	{
		return _convertedObjectives;
	}

	void PnnLABGAQuantizer::resizeConvertedObjectives(int numObj) {
		_convertedObjectives.resize(numObj);
	}

	shared_ptr<PnnLABGAQuantizer> PnnLABGAQuantizer::makeNewFromPrototype() {
		auto child = make_shared<PnnLABGAQuantizer>(*m_pq, m_pixels, _bitmapWidth, _nMaxColors);
		auto minRatio2 = 1.5 * minRatio;
		if(minRatio2 > 1)
			minRatio2 = 0;
		auto ratioX = randrange(minRatio, maxRatio);
		auto ratioY = ratioX < minRatio2 ? randrange(minRatio, maxRatio) : ratioX;
		child->setRatio(ratioX, ratioY);
		child->calculateFitness();
		return child;
	}

	uint PnnLABGAQuantizer::getMaxColors() const {
		return _nMaxColors;
	}

	
	string PnnLABGAQuantizer::getResult() const
	{
		ostringstream ss;
		auto difference = abs(_ratioX - _ratioY);
		if (difference <= 0.0000001)
			ss << std::setprecision(6) << _ratioX;
		else
			ss << std::setprecision(6) << _ratioX << ", " << _ratioY;
		return ss.str();
	}

}
