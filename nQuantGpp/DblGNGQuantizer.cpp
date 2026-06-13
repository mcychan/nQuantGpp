#include "stdafx.h"
#include "DblGNGQuantizer.h"
#include "bitmapUtilities.h"
#include "CIELABConvertor.h"
#include "BlueNoise.h"
#include "GilbertCurve.h"
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace GrowingNeuralGas
{
	/* Distributed Batch Learning Growing Neural Gas algorithm with CIELAB color space
	Copyright (c) 2026 Miller Cy Chan
	 * Siow, C. Z., Saputra, A. A., Obo, T., & Kubota, N. (2024).
	 * Distributed batch learning of growing neural gas for quick and efficient clustering.
	 * Mathematics, 12(12), 1909. */

	double PR = 0.299, PG = 0.587, PB = 0.114, PA = .3333;
	uchar alphaThreshold = 0xF;
	const double TRANS_RATE = 1 - (512 + 101) / 768.0;

	int maxNodes = 256;		// number of colours used	
	int epochs = 20, maxAge = 10; // Maximum age in terms of epochs/batches, not pixels
	mt19937 random;

	bool enforcedDither = true, hasSemiTransparency = false;
	int m_transparentPixelIndex = -1;
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);

	static const float coeffs[3][3] = {
		{0.299f, 0.587f, 0.114f},
		{-0.14713f, -0.28886f, 0.436f},
		{0.615f, -0.51499f, -0.10001f}
	};
	unordered_map<ARGB, CIELABConvertor::Lab> pixelMap;
	unordered_map<int, vector<ushort> > closestMap;
	unordered_map<int, ushort> nearestMap;
	vector<float> saliencies;

	struct GNGNode;

	struct SharedPtrHash {
		template <typename T>
		size_t operator()(const shared_ptr<T>& ptr) const {
			return hash<T*>()(ptr.get());
		}
	};

	struct GNGNode {
		vector<double> weight;
		double error = 0.0;

		unordered_map<shared_ptr<GNGNode>, int, SharedPtrHash> neighbors;

		GNGNode(const vector<double>& w) : weight(w), error(0.0) {}

		void addNeighbour(const shared_ptr<GNGNode>& nextNode) {
			neighbors[nextNode] = 0;
		}

		shared_ptr<GNGNode> findNeighborByMaxError() {
			if (neighbors.empty())
				return nullptr;
			auto it = max_element(neighbors.begin(), neighbors.end(),
				[](const auto& a, const auto& b) -> bool {
					return a.first->error < b.first->error;
				}
			);
			return it->first;
		}

		void incrementAge() {
			for (auto& [neighbor, age] : neighbors) {
				age += 1;
			}
		}

		bool noNeighbor() const { return neighbors.empty(); }
		void removeNeighbour(const shared_ptr<GNGNode>& nextNode) { neighbors.erase(nextNode); }

		void removeNeighbourByAge(int maxAge) {
			for (auto it = neighbors.begin(); it != neighbors.end(); ) {
				if (it->second > maxAge) {
					it = neighbors.erase(it);
				}
				else {
					++it;
				}
			}
		}

		double distance(const vector<double>& input) const {
			double d = 0.0;
			for (size_t i = 0; i < weight.size(); ++i) {
				double diff = weight[i] - input[i];
				d += diff * diff;
			}
			return d;
		}
	};

	vector<shared_ptr<GNGNode>> nodes;

	const bool hasAlpha() {
		return m_transparentPixelIndex >= 0;
	}

	int calculateStartingPoints(double learningRate) {
		const auto K = (hasAlpha() || maxNodes > 32) ? 6.5 : 10.0; 
		auto continuousPoints = maxNodes * exp(-K * learningRate);
		auto noOfStartingPoints = static_cast<int>(round(continuousPoints));

		auto minFloor = hasAlpha() ? 8 : 2;
		auto maxCeiling = max(2, maxNodes / 4); 

		return max(minFloor, min(maxCeiling, noOfStartingPoints));
	}

	void insertNewNodeWeighted(unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>& assignments) {
		auto it_q = max_element(nodes.begin(), nodes.end(),
			[&](const auto& a, const auto& b) -> bool {
				auto errA = 0.0;
				auto errB = 0.0;

				auto itA = assignments.find(a);
				if (itA != assignments.end() && !itA->second.empty()) {
					errA = a->error / log1p(itA->second.size());
				}

				auto itB = assignments.find(b);
				if (itB != assignments.end() && !itB->second.empty()) {
					errB = b->error / log1p(itB->second.size());
				}

				return errA < errB;
			}
		);

		if (it_q == nodes.end())
			return;

		auto q = *it_q;

		if (q == nullptr || q->noNeighbor())
			return;

		auto f = q->findNeighborByMaxError();
		if (f == nullptr)
			return;

		auto weightSize = q->weight.size();
		auto newWeight = vector<double>(weightSize);
		for (auto i = 0u; i < weightSize; ++i) {
			newWeight[i] = (q->weight[i] + f->weight[i]) / 2.0;
		}

		auto r = make_shared<GNGNode>(newWeight);

		q->removeNeighbour(f);
		f->removeNeighbour(q);

		q->addNeighbour(r);
		f->addNeighbour(r);
		r->addNeighbour(q);
		r->addNeighbour(f);

		nodes.push_back(r);

		q->error *= 0.5;
		f->error *= 0.5;
		r->error = q->error;
	}

	void updateNodeWeightsAdaptive(
		unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>& assignments,
		double baseLearningRate,
		double progress
	) {
		// Flatten map entries into indexable vector for uniform OpenMP loop parsing
		auto entries = vector<pair<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>>>(assignments.begin(), assignments.end());

		#pragma omp parallel for schedule(dynamic)
		for (auto i = 0; i < static_cast<int>(entries.size()); ++i) {
			auto& node = entries[i].first;
			const auto& cluster = entries[i].second;

			if (cluster.empty()) continue;

			auto mean = vector<double>(node->weight.size(), 0.0);
			for (const auto& sample : cluster) {
				for (auto j = 0u; j < mean.size(); ++j) {
					mean[j] += sample->weight[j];
				}
			}
			for (auto j = 0u; j < mean.size(); ++j) {
				mean[j] /= cluster.size();
			}

			if (progress < 0.4) {
				auto currentLR = baseLearningRate * (1.0 - progress);
				for (auto j = 0u; j < node->weight.size(); ++j) {
					node->weight[j] += currentLR * (mean[j] - node->weight[j]);
				}
			}
			else {
				auto snapFactor = (progress - 0.4) / 0.6;
				for (auto j = 0u; j < node->weight.size(); ++j) {
					node->weight[j] = node->weight[j] + snapFactor * (mean[j] - node->weight[j]);
				}
			}
		}
	}

	void manageGraphTopology(
		unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>& assignments,
		int remainingEpochs
	) {
		const auto MAX_AGE = 20;

		for (auto& node : nodes) {
			node->incrementAge();
		}

		for (const auto& [firstWinner, samplesInCluster] : assignments) {
			if (samplesInCluster.empty()) continue;

			const auto& anchorSample = samplesInCluster[0]->weight;
			auto secondWinner = shared_ptr<GNGNode>(nullptr);
			auto minDistance2 = DBL_MAX;

			for (const auto& potentialSecond : nodes) {
				if (potentialSecond == firstWinner) continue;

				auto dist = potentialSecond->distance(anchorSample);
				if (dist < minDistance2) {
					minDistance2 = dist;
					secondWinner = potentialSecond;
				}
			}

			if (secondWinner != nullptr) {
				firstWinner->addNeighbour(secondWinner);
				secondWinner->addNeighbour(firstWinner);
			}
		}

		for (auto& node : nodes) {
			node->removeNeighbourByAge(MAX_AGE);
		}

		nodes.erase(remove_if(nodes.begin(), nodes.end(), [&](const shared_ptr<GNGNode>& node) -> bool {
			if (node->noNeighbor()) {
				auto it = assignments.find(node);
				return it == assignments.end() || it->second.empty();
			}
			return false;
			}), nodes.end());

		auto missingNodes = maxNodes - static_cast<int>(nodes.size());
		if (missingNodes > 0 && remainingEpochs > 0) {
			auto targetInsertions = static_cast<int>(ceil((double)missingNodes / remainingEpochs));
			for (auto i = 0; i < targetInsertions; ++i) {
				if (static_cast<int>(nodes.size()) < maxNodes) {
					insertNewNodeWeighted(assignments);
				}
				else {
					break;
				}
			}
		}
	}

	void GetLab(const Vec4b& pixel, CIELABConvertor::Lab& lab1)
	{
		auto argb = GetArgb8888(pixel);
		auto got = pixelMap.find(argb);
		if (got == pixelMap.end()) {
			CIELABConvertor::RGB2LAB(pixel, lab1);
			pixelMap[argb] = lab1;
		}
		else
			lab1 = got->second;
	}

	void initializeDistributedNode(const vector<shared_ptr<GNGNode>>& samples, int noOfStartingPoints) {
		if (samples.empty()) {
			throw invalid_argument("Sample list cannot be empty.");
		}

		auto actualStartingPoints = min(noOfStartingPoints, static_cast<int>(samples.size()));
		if (actualStartingPoints < 2) actualStartingPoints = 2;

		maxAge = max(maxAge, actualStartingPoints * (epochs / 10));
		random.seed(samples.size());
		nodes.clear();

		auto initialNodes = vector<shared_ptr<GNGNode>>();
		auto chosenIndices = unordered_set<int>();
		auto dist = uniform_int_distribution<int>(0, samples.size() - 1);

		while (static_cast<int>(initialNodes.size()) < actualStartingPoints) {
			auto randomIndex = dist(random);
			if (chosenIndices.insert(randomIndex).second) {
				initialNodes.push_back(make_shared<GNGNode>(samples[randomIndex]->weight));
			}
		}

		for (auto i = 0u; i < initialNodes.size(); ++i) {
			auto currentNode = initialNodes[i];
			auto nextNode = initialNodes[(i + 1) % initialNodes.size()];
			currentNode->neighbors[nextNode] = 0;
			nextNode->neighbors[currentNode] = 0;
		}

		nodes.insert(nodes.end(), initialNodes.begin(), initialNodes.end());
	}

	double calculateBalancedLearningRate(int stdDevSampleSize) {
		const auto EPSILON_BASE = 0.12; 
		if (stdDevSampleSize <= 0)
			return EPSILON_BASE;

		auto ratio = (double) maxNodes / stdDevSampleSize;
		auto adaptiveLR = EPSILON_BASE * sqrt(ratio);

		return max(0.015, min(0.080, adaptiveLR));
	}

	shared_ptr<GNGNode> findBestWinner(const vector<double>& sample, const vector<shared_ptr<GNGNode>>& snapshot) {
		auto winner = shared_ptr<GNGNode>(nullptr);
		auto minDist = DBL_MAX;
		for (auto i = 0u; i < snapshot.size(); i++) {
			auto d = snapshot[i]->distance(sample);
			if (d < minDist) {
				minDist = d;
				winner = snapshot[i];
			}
		}
		if (winner != nullptr) {
			#pragma omp critical(WinnerErrorAccumulation)
			{
				winner->error += minDist;
			}
		}
		return winner;
	}

	void trainBatch(vector<shared_ptr<GNGNode>>& samples,
		vector<shared_ptr<GNGNode>>& uniqueSamples,
		vector<shared_ptr<GNGNode>>& stdDevSamples,
		int totalEpochs)
	{
		auto balancedLR = calculateBalancedLearningRate(stdDevSamples.size());
		auto startingPoints = calculateStartingPoints(balancedLR);

		initializeDistributedNode(uniqueSamples, startingPoints);
		auto growthEpochs = static_cast<int>(totalEpochs * 0.7);

		// PHASE 2: TOPOLOGY GROWTH
		for (auto epoch = 0; epoch < growthEpochs; ++epoch) {
			for (auto& node : nodes) node->error = 0.0;
			auto currentNodesSnapshot = nodes;

			unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash> assignments;
			for (const auto& sample : stdDevSamples) {
				auto winner = findBestWinner(sample->weight, currentNodesSnapshot);
				if (winner) {
					assignments[winner].push_back(sample);
				}
			}

			auto progress = (double)epoch / growthEpochs;
			updateNodeWeightsAdaptive(assignments, balancedLR, progress);
			manageGraphTopology(assignments, growthEpochs - epoch);
		}

		// PHASE 3: FINAL CENTROID TUNING
		auto tuningEpochs = totalEpochs - growthEpochs;
		for (auto epoch = 0; epoch < tuningEpochs; ++epoch) {
			auto currentNodesSnapshot = nodes;
			auto realAssignments = unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>();

			// Concurrent collection parallelized securely via OpenMP thread-private map reductions
			#pragma omp parallel
			{
				auto localAssignments = unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>();

				#pragma omp for nowait
				for (auto i = 0; i < static_cast<int>(samples.size()); ++i) {
					auto winner = findBestWinner(samples[i]->weight, currentNodesSnapshot);
					if (winner) localAssignments[winner].push_back(samples[i]);
				}

				#pragma omp critical(MergeRealAssignments)
				{
					for (auto& [node, cluster] : localAssignments) {
						auto& globalCluster = realAssignments[node];
						globalCluster.insert(globalCluster.end(), cluster.begin(), cluster.end());
					}
				}
			}

			// Equivalent to entrySet().parallelStream().forEach(...)
			auto realEntries = vector<pair<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>>>(realAssignments.begin(), realAssignments.end());

			#pragma omp parallel for schedule(dynamic)
			for (auto i = 0; i < static_cast<int>(realEntries.size()); ++i) {
				auto& node = realEntries[i].first;
				const auto& cluster = realEntries[i].second;
				if (cluster.empty()) continue;

				auto trueMean = vector<double>(node->weight.size(), 0.0);
				for (const auto& n : cluster) {
					for (auto j = 0u; j < trueMean.size(); ++j) {
						trueMean[j] += n->weight[j];
					}
				}

				for (auto j = 0u; j < node->weight.size(); ++j) {
					node->weight[j] = trueMean[j] / cluster.size();
				}
			}

			// Topology management and aged-link pruning
			for (auto& node : nodes) {
				node->incrementAge();
			}

			for (auto& node : nodes) {
				node->removeNeighbourByAge(maxAge);
			}

			nodes.erase(remove_if(nodes.begin(), nodes.end(), [&](const auto& node) {
				if (node->noNeighbor()) {
					auto it = realAssignments.find(node);
					return it == realAssignments.end() || it->second.empty();
				}
				return false;
				}), nodes.end());

			auto missingNodes = maxNodes - static_cast<int>(nodes.size());
			if (missingNodes > 0 && epochs - epoch > 0) {
				auto targetInsertions = static_cast<int>(ceil((double)missingNodes / (epochs - epoch)));
				for (auto i = 0; i < targetInsertions; ++i) {
					if (static_cast<int>(nodes.size()) >= maxNodes) break;
					insertNewNodeWeighted(realAssignments);
				}
			}
		}
	}

	void Inxbuild(Mat palette) {
		uint nMaxColors = palette.rows;

		uint k = 0;
		for (const auto& n : nodes) {
			const auto& channels = n.get()->weight;

			CIELABConvertor::Lab lab1;
			lab1.L = (float)channels[0], lab1.A = (float)channels[1], lab1.B = (float)channels[2];
			lab1.alpha = (channels.size() > 3) ? (uchar)channels[3] : UCHAR_MAX;
			Vec4b c1;
			CIELABConvertor::LAB2RGB(c1, lab1);
			SetPixel(palette, k++, 0, c1);

			if (k >= nMaxColors)
				break;
		}
	}

	ushort nearestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
	{
		const auto nMaxColors = palette.rows;
		int offset = GetArgbIndex(c0, hasSemiTransparency, hasAlpha());
		auto got = nearestMap.find(offset);
		if (got != nearestMap.end())
			return got->second;

		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			c = m_transparentColor;

		if (nMaxColors > 2 && hasAlpha() && c[3] > alphaThreshold)
			k = 1;

		double mindist = INT_MAX;
		CIELABConvertor::Lab lab1, lab2;
		GetLab(c, lab1);
		
		for (uint i = k; i < nMaxColors; ++i) {
			Vec4b c2;
			GrabPixel(c2, palette, i, 0);
			auto curdist = hasSemiTransparency ? sqr(c2[3] - c[3]) * TRANS_RATE : 0;
			if (curdist > mindist)
				continue;

			GetLab(c2, lab2);
			if (nMaxColors <= 4) {
				curdist += sqr(c2[2] - c[2]);
				if (curdist > mindist)
					continue;

				curdist += sqr(c2[1] - c[1]);
				if (curdist > mindist)
					continue;

				curdist += sqr(c2[0] - c[0]);
				if (hasSemiTransparency) {
					if (curdist > mindist)
						continue;
					curdist += sqr(c2[3] - c[3]);
				}
			}
			else if (hasSemiTransparency || nMaxColors < 16) {
				curdist += sqr(lab2.L - lab1.L);
				if (curdist > mindist)
					continue;

				curdist += sqr(lab2.A - lab1.A);
				if (curdist > mindist)
					continue;

				curdist += sqr(lab2.B - lab1.B);
			}
			else if (nMaxColors > 32) {
				curdist += abs(lab2.L - lab1.L);
				if (curdist > mindist)
					continue;

				curdist += sqrt(sqr(lab2.A - lab1.A) + sqr(lab2.B - lab1.B));
			}
			else {
				auto deltaL_prime_div_k_L_S_L = CIELABConvertor::L_prime_div_k_L_S_L(lab1, lab2);
				curdist += sqr(deltaL_prime_div_k_L_S_L);
				if (curdist > mindist)
					continue;

				double a1Prime, a2Prime, CPrime1, CPrime2;
				auto deltaC_prime_div_k_L_S_L = CIELABConvertor::C_prime_div_k_L_S_L(lab1, lab2, a1Prime, a2Prime, CPrime1, CPrime2);
				curdist += sqr(deltaC_prime_div_k_L_S_L);
				if (curdist > mindist)
					continue;

				double barCPrime, barhPrime;
				auto deltaH_prime_div_k_L_S_L = CIELABConvertor::H_prime_div_k_L_S_L(lab1, lab2, a1Prime, a2Prime, CPrime1, CPrime2, barCPrime, barhPrime);
				curdist += sqr(deltaH_prime_div_k_L_S_L);
				if (curdist > mindist)
					continue;

				curdist += CIELABConvertor::R_T(barCPrime, barhPrime, deltaC_prime_div_k_L_S_L, deltaH_prime_div_k_L_S_L);
			}

			if (curdist > mindist)
				continue;
			mindist = curdist;
			k = i;
		}
		nearestMap[offset] = k;
		return k;
	}
	
	ushort closestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
	{
		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			return nearestColorIndex(palette, c0, pos);

		const auto nMaxColors = (ushort) palette.rows;
		vector<ushort> closest(4);
		int offset = GetArgbIndex(c0, hasSemiTransparency, hasAlpha());
		auto got = closestMap.find(offset);
		if (got == closestMap.end()) {
			closest[2] = closest[3] = USHRT_MAX;

			for (; k < nMaxColors; ++k) {
				Vec4b c2;
				GrabPixel(c2, palette, k, 0);
				
				auto err = PR * sqr(c2[2] - c[2]);
				if (err >= closest[3])
					continue;

				err += PG * sqr(c2[1] - c[1]);
				if (err >= closest[3])
					continue;

				err += PB * sqr(c2[0] - c[0]);
				if (err >= closest[3])
					continue;

				if (hasSemiTransparency)
					err += PA * sqr(c2[3] - c[3]);

				if (err < closest[2]) {
					closest[1] = closest[0];
					closest[3] = closest[2];
					closest[0] = k;
					closest[2] = err;
				}
				else if (err < closest[3]) {
					closest[1] = k;
					closest[3] = err;
				}
			}

			if (closest[3] == USHRT_MAX)
				closest[1] = closest[0];

			closestMap[offset] = closest;
		}
		else
			closest = got->second;

		int idx = 1;
		if (closest[2] == 0 || (rand() % (int)ceil(closest[3] + closest[2])) <= closest[3])
			idx = 0;

		auto MAX_ERR = palette.rows;
		Vec4b c1;
		GrabPixel(c1, palette, closest[idx], 0);
		if (closest[idx + 2] >= MAX_ERR || closest[idx + 2] == 0 || c1[3] < c[3])
			return nearestColorIndex(palette, c, pos);
		return closest[idx];
	}

	void clear() {
		saliencies.clear();
		closestMap.clear();
		nearestMap.clear();
	}

	void gngquan(const Mat4b pixels, Mat palette, uint& nMaxColors)
	{
		maxNodes = nMaxColors;		// number of colours used

		auto GetColorIndex = [&](const Vec4b& c) -> int {
			return GetArgbIndex(c, hasSemiTransparency, hasAlpha());
		};
		auto NearestColorIndex = [nMaxColors](const Mat palette, const Vec4b& c, const uint pos) -> ushort {
			if (nMaxColors <= 4)
				return nearestColorIndex(palette, c, pos);
			return closestColorIndex(palette, c, pos);
		};

		vector<shared_ptr<GNGNode>> samples;
		vector<shared_ptr<GNGNode>> uniqueSamples;
		unordered_map<ARGB, int> histogram;

		// Sequential color extraction loop
		for (int y = 0; y < pixels.rows; ++y) {
			for (int x = 0; x < pixels.cols; ++x) {
				auto c = pixels(y, x);

				Vec4b pixel;
				GetArgb(pixel, c, hasSemiTransparency, hasAlpha());
				auto argb = GetArgb8888(pixel);
				auto isRegistered = pixelMap.find(argb) != pixelMap.end();

				CIELABConvertor::Lab lab1;
				GetLab(pixel, lab1);
				vector<double> currentWeight;

				if (hasAlpha()) {
					currentWeight = { lab1.L, lab1.A, lab1.B, (double) lab1.alpha };
				}
				else {
					currentWeight = { lab1.L, lab1.A, lab1.B };
				}

				// Generate a new GNGNode inside a safe shared heap block
				auto sampleNode = make_shared<GNGNode>(currentWeight);
				samples.push_back(sampleNode);

				if (!isRegistered) {
					uniqueSamples.push_back(make_shared<GNGNode>(currentWeight));
				}

				// Idiomatic count tracking incrementation (Auto-initializes to 0 if key is absent)
				histogram[argb]++;
			}
		}

		if (pixelMap.size() <= nMaxColors) {
			/* Fill palette */
			nMaxColors = pixelMap.size();
			palette = palette.rowRange(0, nMaxColors);
			int k = 0;
			for (const auto& [pixel, lab] : pixelMap) {
				uchar red = pixel & 0xff,
				green = (pixel >> 8) & 0xff,
				blue = (pixel >> 16) & 0xff,
				alpha = (pixel >> 24) & 0xff;
				if (palette.channels() == 4)
					palette.at<Vec4b>(k, 0) = Vec4b(blue, green, red, alpha);
				else
					palette.at<Vec3b>(k, 0) = Vec3b(blue, green, red);

				if (k > 0 && alpha == 0)
					swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(k, 0));
				++k;
			}

			return;
		}

		auto mDivn = min(0.9, nMaxColors * 1.0 / pixelMap.size());
		if (hasSemiTransparency)
			mDivn *= -1;

		vector<shared_ptr<GNGNode>> stdDevSamples;
		for (const auto& [pixel, count] : histogram) {
			auto freq = static_cast<int>(sqrt(count));

			for (auto j = 0; j < freq; ++j) {
				auto lab1 = pixelMap[pixel];

				vector<double> currentWeight;
				if (hasAlpha()) {
					currentWeight = { lab1.L, lab1.A, lab1.B, (double)lab1.alpha };
				}
				else {
					currentWeight = { lab1.L, lab1.A, lab1.B };
				}

				stdDevSamples.push_back(make_shared<GNGNode>(currentWeight));
			}
		}

		if (mDivn < .04 && PG < 1 && PG >= coeffs[0][1] && nMaxColors >= 64)
			enforcedDither = false;
		if (mDivn > .003 && nMaxColors <= 32)
			enforcedDither = false;

		if ((nMaxColors < 32 && mDivn > .015 && mDivn < .032) || (nMaxColors >= 32 && nMaxColors < 64 && mDivn > .03 && mDivn < .06))
			trainBatch(uniqueSamples, samples, stdDevSamples, epochs);
		else
			trainBatch(samples, uniqueSamples, stdDevSamples, epochs);

		if (nodes.size() > static_cast<size_t>(nMaxColors)) {
			cerr << "Truncated no. of clusters from " << nodes.size() << " to " << nMaxColors << "\n";
		} 
		else if (nodes.size() < static_cast<size_t>(nMaxColors)) {
			nMaxColors = nodes.size();
			cerr << "Reduced no. of clusters to " << nMaxColors << "\n";
		}
		Inxbuild(palette);
	}	

	void grabPixels(const Mat srcImg, Mat4b pixels, uint& nMaxColors, bool& hasSemiTransparency)
	{
		int semiTransCount = 0;
		GrabPixels(srcImg, pixels, semiTransCount, m_transparentPixelIndex, m_transparentColor, alphaThreshold, nMaxColors);
		hasSemiTransparency = semiTransCount > 0;
	}

	bool quantize_image(const Mat4b pixels, const Mat palette, const uint nMaxColors, Mat1b qPixels, const bool dither)
	{
		auto width = pixels.cols;
		auto height = pixels.rows;
		if (dither)
			return dither_image(pixels, palette, nearestColorIndex, hasSemiTransparency, m_transparentPixelIndex, nMaxColors, qPixels);

		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				auto& pixel = pixels(j, i);
				qPixels(j, i) = (uchar) nearestColorIndex(palette, pixel, i + j);
			}
		}

		return true;
	}

	Mat DblGNGQuantizer::QuantizeImageByPal(const Mat4b pixels4b, const Mat palette, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		auto mDivn = min(0.9, nMaxColors * 1.0 / pixelMap.size());
		if (hasSemiTransparency)
			mDivn *= -1;

		auto bitmapWidth = pixels4b.cols;
		auto bitmapHeight = pixels4b.rows;

		if (dither && !enforcedDither) {
			Mat1b qPixels(bitmapHeight, bitmapWidth);
			quantize_image(pixels4b, palette, nMaxColors, qPixels, dither);

			pixelMap.clear();
			clear();

			ProcessImagePixels(bytes, palette, qPixels, hasAlpha());
			return palette;
		}

		if (dither) {
			auto length = (size_t) pixels4b.rows * pixels4b.cols;
			saliencies.resize(length);
			auto saliencyBase = .1f;

			for (int y = 0; y < pixels4b.rows; ++y) {
				for (int x = 0; x < pixels4b.cols; ++x) {
					auto c = pixels4b(y, x);

					CIELABConvertor::Lab lab1;
					GetLab(c, lab1);
					int i = x + y * pixels4b.cols;
					saliencies[i] = saliencyBase + (1 - saliencyBase) * lab1.L / 100.0f * lab1.alpha / 255.0f;
				}
			}
		}

		if (enforcedDither)
			enforcedDither = nMaxColors < 32 || nMaxColors > 64;

		auto GetColorIndex = [&](const Vec4b& c) -> int {
			return GetArgbIndex(c, hasSemiTransparency, hasAlpha());
		};
		auto NearestColorIndex = [this, nMaxColors](const Mat palette, const Vec4b& c, const uint pos) -> ushort {
			if (nMaxColors <= 4)
				return nearestColorIndex(palette, c, pos);
			return closestColorIndex(palette, c, pos);
		};

		if (nMaxColors > 256) {
			Mat qPixels(bitmapHeight, bitmapWidth, palette.type());
			Peano::GilbertCurve::dithering(pixels4b, palette, NearestColorIndex, GetColorIndex, qPixels, saliencies.data(), mDivn, dither, enforcedDither);

			pixelMap.clear();
			clear();
			ProcessImagePixels(bytes, qPixels, hasAlpha());
			return qPixels;
		}

		Mat1b qPixels(bitmapHeight, bitmapWidth);
		Peano::GilbertCurve::dithering(pixels4b, palette, NearestColorIndex, GetColorIndex, qPixels, saliencies.data(), mDivn, dither, enforcedDither);

		if (!dither && nMaxColors > 32) {
			const auto delta = sqr(nMaxColors) / pixelMap.size();
			mDivn = delta > 0.023 ? 1.0f : (float)(36.921 * delta + 0.906);
			BlueNoise::dither(pixels4b, palette, NearestColorIndex, GetColorIndex, qPixels, mDivn);
		}

		pixelMap.clear();
		clear();

		ProcessImagePixels(bytes, palette, qPixels, hasAlpha());
		return palette;
	}

	Mat DblGNGQuantizer::QuantizeImage(const Mat4b pixels4b, Mat palette, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		if (hasAlpha() || nMaxColors <= 32)
			PR = PG = PB = PA = 1;
		else {
			PR = coeffs[0][0]; PG = coeffs[0][1]; PB = coeffs[0][2];
		}
		
		if (nMaxColors > 2)
			gngquan(pixels4b, palette, nMaxColors);
		else {
			if (m_transparentPixelIndex >= 0)
				palette.at<Vec4b>(0, 0) = m_transparentColor;
			else
				palette.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);
		}

		if (m_transparentPixelIndex >= 0) {
			auto bitmapWidth = pixels4b.cols;
			auto k = nearestColorIndex(palette, pixels4b.at<Vec4b>(m_transparentPixelIndex / bitmapWidth, m_transparentPixelIndex % bitmapWidth), m_transparentPixelIndex);
			if (nMaxColors > 2)
				palette.at<Vec4b>(k, 0) = m_transparentColor;
			else if (GetArgb8888(palette.at<Vec4b>(k, 0)) != GetArgb8888(m_transparentColor))
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(1, 0));
		}

		return QuantizeImageByPal(pixels4b, palette, bytes, nMaxColors, dither);
	}

	Mat DblGNGQuantizer::QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
		grabPixels(srcImg, pixels4b, nMaxColors, hasSemiTransparency);

		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);
		Mat palette(nMaxColors, 1, srcImg.type(), scalar);
		return this->QuantizeImage(pixels4b, palette, bytes, nMaxColors, dither);
	}

}
