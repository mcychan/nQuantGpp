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

			double _fitness = -numeric_limits<double>::infinity();
			double _ratioX = 0, _ratioY = 0;
			vector<double> _convertedObjectives;
			vector<double> _objectives;
			vector<shared_ptr<Mat4b> > m_pixelsList;
			unique_ptr<PnnLABQuantizer> m_pq;

			void calculateError(vector<double>& errors);
			void calculateFitness();
			string getRatioKey() const;
			auto findByRatioKey(const string& ratioKey) const;
			void clear();

		public:
			PnnLABGAQuantizer(PnnLABQuantizer& pq, const vector<shared_ptr<Mat> >& srcImgs, uint nMaxColors);
			PnnLABGAQuantizer(PnnLABQuantizer& pq, const vector<shared_ptr<Mat4b> >& pixelsList, const uint& nMaxColors);
			
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
			vector<shared_ptr<Mat> > QuantizeImage(vector<vector<uchar> >& bytesList, bool dither = true);
	};
}
