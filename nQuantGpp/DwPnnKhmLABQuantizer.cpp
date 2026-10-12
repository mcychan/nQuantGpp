/* Density-weighted pairwise nearest neighbor K-harmonic means algorithm with CIELAB color space advanced version
Copyright (c) 2026 Miller Cy Chan
* error measure; time used is proportional to number of bins squared - WJ */

#include "stdafx.h"
#include "DwPnnKhmLABQuantizer.h"
#include "bitmapUtilities.h"
#include <cmath>
#include <vector>
#include <omp.h>

namespace PnnLABQuant
{
	DwPnnKhmLABQuantizer::DwPnnKhmLABQuantizer() {
	}

	/// <summary>
	/// Refines the initial PNN palette using Density-Weighted K-harmonic means 
	/// optimized with OpenCV and OpenMP.
	/// </summary>
	void DwPnnKhmLABQuantizer::RefinePaletteWithDwKhMeans(const vector<CIELABConvertor::Lab>& uniqueLabs, const vector<float>& weights, Mat palette, int maxIterations)
	{
		int k = palette.rows;
		vector<CIELABConvertor::Lab> centroids(k);

		// Initialize centroids from the initial palette
		for (int i = 0; i < k; ++i) {
			Vec4b c;
			GrabPixel(c, palette, i, 0);
			GetLab(c, centroids[i]);
		}

		vector<float> newL(k, 0.0f);
		vector<float> newA(k, 0.0f);
		vector<float> newB(k, 0.0f);
		vector<float> newAlpha(k, 0.0f);
		vector<float> weightSum(k, 0.0f);

		const auto p = hasSemiTransparency ? 2.5f : 3.0f; // K-harmonic means power parameter
		auto numUniqueLabs = uniqueLabs.size();

		for (int iter = 0; iter < maxIterations; ++iter) {
			fill(weightSum.begin(), weightSum.end(), 0.0f);
			fill(newL.begin(), newL.end(), 0.0f);
			fill(newA.begin(), newA.end(), 0.0f);
			fill(newB.begin(), newB.end(), 0.0f);
			fill(newAlpha.begin(), newAlpha.end(), 0.0f);

			// K-harmonic means update phase with density weighting using OpenMP parallel for
			#pragma omp parallel for
			for (int i = 0; i < (int)numUniqueLabs; ++i) {
				auto lab = uniqueLabs[i];
				Vec3f p1(lab.L, lab.A, lab.B);
				vector<float> distances(k);
				auto sumInvDistP = 0.0f;

				for (int j = 0; j < k; ++j) {
					Vec3f p2(centroids[j].L, centroids[j].A, centroids[j].B);
					auto dist = (float)norm(p1, p2);
					if (dist < 1e-5f)
						dist = 1e-5f; // Prevent division by zero

					auto invDistP = (float)pow(dist, -p);
					distances[j] = invDistP;
					sumInvDistP += invDistP;
				}

				// Accumulate weighted contributions across all clusters using KHM soft membership
				auto w = weights[i];
				for (int j = 0; j < k; ++j) {
					Vec3f p2(centroids[j].L, centroids[j].A, centroids[j].B);
					auto dist = (float)norm(p1, p2);
					if (dist < 1e-5f)
						dist = 1e-5f;

					auto khmWeight = w * (float)(pow(dist, -p - 2.0) / pow(sumInvDistP, 2.0));

					// Thread-safe accumulation using OpenMP critical section
					#pragma omp critical
					{
						newL[j] += lab.L * khmWeight;
						newA[j] += lab.A * khmWeight;
						newB[j] += lab.B * khmWeight;
						newAlpha[j] += (float)lab.alpha * khmWeight;
						weightSum[j] += khmWeight;
					}
				}
			}

			bool changed = false;
			for (int j = 0; j < k; ++j) {
				if (weightSum[j] > 1e-5f) {
					CIELABConvertor::Lab updatedLab;
					updatedLab.L = newL[j] / weightSum[j];
					updatedLab.A = newA[j] / weightSum[j];
					updatedLab.B = newB[j] / weightSum[j];
					updatedLab.alpha = (uchar)min(255.0f, max(0.0f, newAlpha[j] / weightSum[j]));

					centroids[j] = updatedLab;
					Vec4b c1;
					CIELABConvertor::LAB2RGB(c1, updatedLab);
					SetPixel(palette, j, 0, c1);
					changed = true;
				}
			}

			if (!changed)
				break;
		}
	}

	void DwPnnKhmLABQuantizer::pnnquan(const Mat4b pixels, Mat palette, uint& nMaxColors)
	{
		int maxbins = 0;
		auto bins = Getbins(pixels, palette, maxbins, nMaxColors);

		// Build synchronized uniqueLabs and weights vectors from non-empty bins
		vector<CIELABConvertor::Lab> uniqueLabsList;
		vector<float> finalWeightsList;

		for (int i = 0; i < maxbins; ++i) {
			CIELABConvertor::Lab lab;
			lab.alpha = (uchar)bins[i].ac;
			lab.L = bins[i].Lc;
			lab.A = bins[i].Ac;
			lab.B = bins[i].Bc;

			uniqueLabsList.emplace_back(lab);
			finalWeightsList.push_back((float)(bins[i].cnt * (1.0f + abs(bins[i].Lc / 100.0f))));
		}

		int maxIterations = nMaxColors <= 16 ? 5 : (nMaxColors <= 64 ? 8 : 10);
		RefinePaletteWithDwKhMeans(uniqueLabsList, finalWeightsList, palette, maxIterations);
	}
}