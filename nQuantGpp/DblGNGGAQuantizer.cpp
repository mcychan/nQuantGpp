/* Distributed Batch Learning Growing Neural Gas algorithm with CIELAB color space
* Copyright (c) 2023 - 2026 Miller Cy Chan */

#include "stdafx.h"
#include "DblGNGGAQuantizer.h"
#include "CIELABConvertor.h"
#include "BlueNoise.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <iomanip>
#include <numeric>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>

namespace GrowingNeuralGas
{
	int _dp = 3, _type = 0;
	uint _nMaxColors = 256;
	double minLR = .0015, maxLR = .008;
	int minSpt = 2, maxSpt = _nMaxColors / 4;

	static unordered_map<string, vector<double> > _fitnessMap;
	static shared_mutex _mutex;

	DblGNGGAQuantizer::DblGNGGAQuantizer(DblGNGQuantizer& dgq, const vector<shared_ptr<Mat> >& srcImgs, uint nMaxColors) {
		// increment value when criteria violation occurs
		_objectives.resize(4);
		srand(srcImgs[0]->cols * srcImgs[0]->rows);
		
		m_dgq = make_unique<DblGNGQuantizer>(dgq);
		if(dgq.IsGA())
			return;

		clear();
		_nMaxColors = nMaxColors;
		maxSpt = _nMaxColors / 4;
		m_pixelsList.clear();

		bool hasSemiTransparency = false;
		for(auto& srcImg : srcImgs) {
			auto pixels = make_shared<Mat4b>(srcImg->rows, srcImg->cols, Scalar(0, 0, 0, UCHAR_MAX));
			m_dgq->grabPixels(*srcImg, *pixels, _nMaxColors, hasSemiTransparency);
			m_pixelsList.emplace_back(pixels);
			_type = srcImg->type();
		}
	}

	DblGNGGAQuantizer::DblGNGGAQuantizer(DblGNGQuantizer& dgq, const vector<shared_ptr<Mat4b> > & pixelsList, const uint& nMaxColors)
	{
		m_dgq = make_unique<DblGNGQuantizer>(dgq);
		// increment value when criteria violation occurs
		_objectives.resize(4);
		m_pixelsList = pixelsList;
		srand(m_pixelsList[0]->rows * m_pixelsList[0]->cols);
		_nMaxColors = nMaxColors;
	}

	string DblGNGGAQuantizer::getParamsKey() const
	{
		ostringstream ss;
		ss << _startingPoints;

		ss << ";" << (int)(_learningRate * _dp);
		return ss.str();
	}

	auto DblGNGGAQuantizer::findByParamsKey(const string& paramsKey) const
	{
		unique_lock<shared_mutex> lock(_mutex);
		auto got = _fitnessMap.find(paramsKey);
		if (got != _fitnessMap.end())
			return got->second;
		return vector<double>();
	}

	void DblGNGGAQuantizer::calculateError(vector<double>& errors) {
		if (m_dgq->hasAlpha()) {
			minSpt = 8;
			maxLR = .04;
		}

		auto fitness = 0.0;
		int length = accumulate(begin(m_pixelsList), end(m_pixelsList), 0, [](int i, const shared_ptr<Mat4b>& pixels) {
			return pixels->rows * pixels->cols + i;
		});
		for (int i = 0; i < errors.size(); ++i)
			errors[i] /= 1.0 * length;

		for (int i = 0; i < errors.size(); ++i) {
			if (i >= 1)
				errors[i] /= 2.55;
			fitness -= errors[i];
		}

		_objectives = errors;
		_fitness = fitness;
	}

	void DblGNGGAQuantizer::calculateFitness() {
		auto paramsKey = getParamsKey();
		auto objectives = findByParamsKey(paramsKey);
		if (!objectives.empty()) {
			_fitness = -1.0 * accumulate(objectives.begin(), objectives.end(), 0.0);
			_objectives = objectives;
			return;
		}

		_objectives.resize(4);
		m_dgq->setParams(_learningRate, _startingPoints);
		
		auto scalar = m_dgq->hasAlpha() ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);
		auto palette = make_shared<Mat>(_nMaxColors, 1, _type, scalar);
		m_dgq->gngquan(*m_pixelsList[0], *palette, _nMaxColors);

		auto errors = _objectives;
		fill(errors.begin(), errors.end(), 0);

		int threshold = (_nMaxColors < 64 || m_dgq->hasAlpha()) ? -64 : -112;

		int pixelIndex = 0;
		for (auto& pixels : m_pixelsList) {
			for (int y = 0; y < pixels->rows; ++y) {
				for (int x = 0; x < pixels->cols; ++x, ++pixelIndex)
				{
					if(BlueNoise::TELL_BLUE_NOISE[pixelIndex & 4095] > threshold)
						continue;

					auto c = pixels->at<Vec4b>(y, x);
					CIELABConvertor::Lab lab1, lab2;
					m_dgq->GetLab(c, lab1);
					auto qPixelIndex = m_dgq->nearestColorIndex(*palette, c, pixelIndex);
					Vec4b c2;
					GrabPixel(c2, *palette, qPixelIndex, 0);
					m_dgq->GetLab(c2, lab2);

					if (m_dgq->hasAlpha()) {
						errors[0] += sqr(lab2.L - lab1.L);
						errors[1] += sqr(lab2.A - lab1.A);
						errors[2] += sqr(lab2.B - lab1.B);
						errors[3] += sqr(lab2.alpha - lab1.alpha) * m_dgq->TRANS_RATE;
					}
					else {
						errors[0] += abs(lab2.L - lab1.L);
						errors[1] += sqrt(sqr(lab2.A - lab1.A) + sqr(lab2.B - lab1.B));
					}
				}
			}
		}
		
		calculateError(errors);
		unique_lock<shared_mutex> lock(_mutex);
		_fitnessMap.insert({ paramsKey, _objectives });
	}
	
	vector<shared_ptr<Mat> > DblGNGGAQuantizer::QuantizeImage(vector<vector<uchar> >& bytesList, bool dither) {
		m_dgq->setParams(_learningRate, _startingPoints);
		auto scalar = m_dgq->hasAlpha() ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);
		auto palette = make_shared<Mat>(_nMaxColors, 1, _type, scalar);
		
		m_dgq->gngquan(*m_pixelsList[0], *palette, _nMaxColors);
		vector<shared_ptr<Mat> > imgList;
		for(auto& pixels : m_pixelsList) {
			vector<uchar> bytes;
			shared_ptr<Mat> pImg;
			if(palette->rows > 256)
				pImg = make_shared<Mat>(m_dgq->QuantizeImage(*pixels, *palette, bytes, _nMaxColors, dither));
			else
				pImg = make_shared<Mat>(m_dgq->QuantizeImageByPal(*pixels, *palette, bytes, _nMaxColors, dither));
			bytesList.emplace_back(bytes);
			if(imgList.empty() || (pImg->rows * pImg->cols) > _nMaxColors)
				imgList.emplace_back(pImg);
		}
		m_dgq->clear();
		return imgList;
	}

	void DblGNGGAQuantizer::clear() {
		unique_lock<shared_mutex> lock(_mutex);
		m_pixelsList.clear();
		_fitnessMap.clear();
	}

	double randrange(double min, double max)
	{
		auto f = (double) rand() / RAND_MAX;
		return min + f * (max - min);
	}
	
	void DblGNGGAQuantizer::setParams(double learningRate, int startingPoints)
	{
		_learningRate = learningRate;
		_startingPoints = startingPoints;
	}

	float DblGNGGAQuantizer::getFitness() {
		return (float) _fitness;
	}

	int rotateSpt(double u, double v) {
		auto theta = M_PI * randrange(minSpt, maxSpt);
		auto result = u * sin(theta) + v * cos(theta);
		if (result <= minSpt || result >= maxSpt) {
			auto range = maxSpt - minSpt;
			result = minSpt + fmod(abs(result) - minSpt, range);
		}
		return (int) round(result);
	}
	
	double rotateLR(double u, double v) {
		auto theta = randrange(0.0, 2 * M_PI);
		auto result = u * cos(theta) - v * sin(theta);
		if (result <= minLR || result >= maxLR) {
			auto range = maxLR - minLR;
			result = minLR + fmod(abs(result) - minLR, range);
		}
		return result;
	}

	shared_ptr<DblGNGGAQuantizer> DblGNGGAQuantizer::crossover(const DblGNGGAQuantizer& mother, int numberOfCrossoverPoints, float crossoverProbability)
	{
		auto child = makeNewFromPrototype();
		if ((rand() % 100) <= crossoverProbability)
			return child;

		auto learningRate = rotateLR(_learningRate, mother._learningRate);
		int startingPoints = rotateSpt(_startingPoints, mother._startingPoints);
		child->setParams(learningRate, startingPoints);
		child->calculateFitness();
		return child;
	}

	const double MUTATION_STDDEV_RATIO = 0.15;
	double boxMuller(double value, double minValue, double maxValue) {
		auto u1 = randrange(0.0, 1.0);
		auto u2 = randrange(0.0, 1.0);

		if (u1 < 1e-9)
			u1 = 1e-9;

		auto pureBoxMuller = sqrt(-2 * log(u1))* cos(2 * M_PI * u2);
		auto stddev = (maxValue - minValue) * MUTATION_STDDEV_RATIO;

		auto result = value + pureBoxMuller * stddev;
		if (result < minValue) {
			result = minValue;
		}
		else if (result > maxValue) {
			result = maxValue;
		}
		return result;
	}

	bool DblGNGGAQuantizer::dominates(const DblGNGGAQuantizer* right) {
		bool better = false;
		for (int f = 0; f < getObjectives().size(); ++f) {
			if (getObjectives()[f] > right->getObjectives()[f])
				return false;

			if (getObjectives()[f] < right->getObjectives()[f])
				better = true;
		}
		return better;
	}

	void DblGNGGAQuantizer::mutation(int mutationSize, float mutationProbability) {
		// check probability of mutation operation
		if ((rand() % 100) > mutationProbability)
			return;

		auto learningRate = _learningRate;
		auto startingPoints = _startingPoints;
		if(randrange(.0, 1.0) > .5)
			learningRate = boxMuller(learningRate, minLR, maxLR);
		else
			startingPoints = boxMuller(startingPoints, minSpt, maxSpt);

		setParams(learningRate, startingPoints);
		calculateFitness();
	}

	vector<double> DblGNGGAQuantizer::getObjectives() const
	{
		return _objectives;
	}

	vector<double>& DblGNGGAQuantizer::getConvertedObjectives()
	{
		return _convertedObjectives;
	}

	void DblGNGGAQuantizer::resizeConvertedObjectives(int numObj) {
		_convertedObjectives.resize(numObj);
	}

	shared_ptr<DblGNGGAQuantizer> DblGNGGAQuantizer::makeNewFromPrototype() {
		auto child = make_shared<DblGNGGAQuantizer>(*m_dgq, m_pixelsList, _nMaxColors);
		auto learningRate = randrange(minLR, maxLR);
		int startingPoints = (int) round(randrange(minSpt, maxSpt));
		child->setParams(learningRate, startingPoints);
		child->calculateFitness();
		return child;
	}

	uint DblGNGGAQuantizer::getMaxColors() const {
		return _nMaxColors;
	}
	
	string DblGNGGAQuantizer::getResult() const
	{
		ostringstream ss;
		ss << setprecision(3) << _learningRate << ", " << _startingPoints;
		return ss.str();
	}

}
