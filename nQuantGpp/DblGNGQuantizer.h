#pragma once
#include "CIELABConvertor.h"
#include <memory>
#include <vector>
using namespace std;

namespace GrowingNeuralGas
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

	class DblGNGQuantizer
	{
		private:
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

			bool isGA = false, hasSemiTransparency = false;
			int m_transparentPixelIndex = -1;
			int startingPoints = 2;
			double learningRate = .002;
			double mDivn = 0.0;
			vector<shared_ptr<GNGNode>> nodes;
			vector<shared_ptr<GNGNode>> samples;
			vector<shared_ptr<GNGNode>> uniqueSamples;
			vector<shared_ptr<GNGNode>> stdDevSamples;
			unordered_map<uint32_t, int> histogram;
			
			unordered_map<ARGB, CIELABConvertor::Lab> pixelMap;
			unordered_map<int, vector<ushort> > closestMap;
			unordered_map<int, ushort> nearestMap;
			vector<float> saliencies;
			
			void insertNewNodeWeighted(unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>& assignments);
			void updateNodeWeightsAdaptive(unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>& assignments,
				double baseLearningRate, double progress);
			void manageGraphTopology(unordered_map<shared_ptr<GNGNode>, vector<shared_ptr<GNGNode>>, SharedPtrHash>& assignments, int remainingEpochs);
			void initializeDistributedNode(const vector<shared_ptr<GNGNode>>& samples, int noOfStartingPoints);
			shared_ptr<GNGNode> findBestWinner(const vector<double>& sample, const vector<shared_ptr<GNGNode>>& snapshot);
			void trainBatch(vector<shared_ptr<GNGNode>>& samples, vector<shared_ptr<GNGNode>>& uniqueSamples,
				vector<shared_ptr<GNGNode>>& stdDevSamples, int totalEpochs);
			void Inxbuild(Mat palette);

			bool quantize_image(const Mat4b pixels, const Mat palette, const uint nMaxColors, Mat1b qPixels, const bool dither);
			ushort closestColorIndex(const Mat palette, const Vec4b& c0, const uint pos);
			
		public:
			const double TRANS_RATE = 1 - (512 + 101) / 768.0;
			DblGNGQuantizer();
			DblGNGQuantizer(const DblGNGQuantizer& quantizer);
			void clear();
			void gngquan(const Mat4b pixels, Mat palette, uint& nMaxColors);
			const bool IsGA() const;
			void GetLab(const Vec4b& pixel, CIELABConvertor::Lab& lab1);
			const bool hasAlpha() const;
			void setParams(double learningRate, int startingPoints);
			void grabPixels(const Mat srcImg, Mat4b pixels, uint& nMaxColors, bool& hasSemiTransparency);

			ushort nearestColorIndex(const Mat palette, const Vec4b& c0, const uint pos);
			Mat QuantizeImageByPal(const Mat4b pixels4b, const Mat palette, vector<uchar>& bytes, uint& nMaxColors, bool dither = true);
			Mat QuantizeImage(const Mat4b pixels, Mat palette, vector<uchar>& bytes, uint& nMaxColors, bool dither = true);
			Mat QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither = true);
	};
}