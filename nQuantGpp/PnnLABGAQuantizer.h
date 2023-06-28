#pragma once
#include "PnnLABQuantizer.h"
#include "ga/Chromosome.h"

namespace PnnLABQuant
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

	class PnnLABGAQuantizer : nQuantGA::Chromosome<PnnLABGAQuantizer>
	{
		private:
			//Asserts floating point compatibility at compile time
			static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

			float _fitness = -numeric_limits<float>::infinity();
			double _ratioX = 0, _ratioY = 0;
			vector<double> _convertedObjectives;
			vector<double> _objectives;
			shared_ptr<Mat4b> m_pixels;
			unique_ptr<PnnLABQuantizer> m_pq;

			void calculateFitness();
			auto findByRatioKey() const;

		public:
			PnnLABGAQuantizer(PnnLABQuantizer& pq, Mat srcImg, uint nMaxColors);
			PnnLABGAQuantizer(PnnLABQuantizer& pq, const shared_ptr<Mat4b> pixels, int bitmapWidth, uint nMaxColors);
			
			float getFitness() override;
			shared_ptr<PnnLABGAQuantizer> crossover(const PnnLABGAQuantizer& mother, int numberOfCrossoverPoints, float crossoverProbability) override;
			bool dominates(const PnnLABGAQuantizer* right) override;
			void mutation(int mutationSize, float mutationProbability) override;
			vector<double> getObjectives() const override;
			vector<double>& getConvertedObjectives() override;
			void resizeConvertedObjectives(int numObj) override;
			shared_ptr<PnnLABGAQuantizer> makeNewFromPrototype() override;

			uint getMaxColors() const;
			string getResult() const;
			void setRatio(double ratioX, double ratioY);
			Mat QuantizeImage(vector<uchar>& bytes, bool dither = true);
			~PnnLABGAQuantizer();
	};
}
