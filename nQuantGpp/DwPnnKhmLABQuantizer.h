#pragma once
#include "PnnLABQuantizer.h"
#include <vector>

namespace PnnLABQuant
{
	class DwPnnKhmLABQuantizer : public PnnLABQuantizer
	{
	private:
		/// <summary>
		/// Refines the initial PNN palette using Density-Weighted K-harmonic means 
		/// optimized with OpenCV and OpenMP.
		/// </summary>
		void RefinePaletteWithDwKhMeans(const std::vector<CIELABConvertor::Lab>& uniqueLabs, const std::vector<float>& weights, Mat palette, int maxIterations = 10);

	public:
		DwPnnKhmLABQuantizer();

		// Overrides the base quantization method
		void pnnquan(const Mat4b pixels, Mat palette, uint& nMaxColors) override;
	};
}