/* Copyright (c) 2006 Derrick Coetzee
Copyright(c) 2015 Hao-Zhi Huang
Copyright (c) 2023-2024 Miller Cy Chan

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "stdafx.h"
#include "EdgeAwareSQuantizer.h"
#include "MedianCut.h"
#include "bitmapUtilities.h"
#include "CIELABConvertor.h"
#include "GilbertCurve.h"

#include <deque>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <random>
#include <math.h>
#include <time.h>
#include <limits>

namespace EdgeAwareSQuant
{
	uchar alphaThreshold = 0xF;
	bool hasSemiTransparency = false;
	int m_transparentPixelIndex = -1;
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);
	unordered_map<ARGB, CIELABConvertor::Lab> pixelMap;
	unordered_map<ARGB, ushort> nearestMap;

	const int DECOMP_SVD = 1;
	const short minLabValues[] = { 0, -128, -128, 0 };
	const short maxLabValues[] = { 100, 127, 127, 255 };

	static bool mycmp(pair<float, int> p1, pair<float, int> p2)
	{
		return p1.first < p2.first;
	}

	template <typename T>
	class array3d
	{
	public:
		array3d(int width, int height, int depth)
		{
			this->width = width;
			this->height = height;
			this->depth = depth;
			data = make_unique<T[]>(width * height * depth);
		}

		array3d(const array3d<T>& rhs)
		{
			width = rhs.get_width();
			height = rhs.height;
			depth = rhs.depth;
			data = make_unique<T[]>(width * height * depth);
			copy(rhs.data.get(), rhs.data.get() + (width * height * depth), data.get());
		}

		inline T& operator()(int col, int row, int layer)
		{
			return data[row * width * depth + col * depth + layer];
		}

		inline const T& operator()(int col, int row, int layer) const
		{
			return data[row * width * depth + col * depth + layer];
		}

		inline int get_width()  const { return width; }
		inline int get_height() const { return height; }
		inline int get_depth() const { return depth; }

	private:
		unique_ptr<T[]> data;
		int width, height, depth;
	};

	void fill_random_icm(Mat1b indexImg8, int palette_size) {
		for (int i = 0; i < indexImg8.rows; ++i) {
			for (int j = 0; j < indexImg8.cols; ++j)
				indexImg8(i, j) = (uchar) (rand() % palette_size);
		}
	}

	void random_permutation(int count, vector<int>& result) {
		result.resize(count);
		iota(result.begin(), result.end(), 0);
		auto rng = default_random_engine{};
		shuffle(result.begin(), result.end(), rng);
	}

	void random_permutation_2d(int width, int height, deque<pair<int, int> >& result) {
		vector<int> perm1d;
		random_permutation(width * height, perm1d);
		while (!perm1d.empty()) {
			int idx = perm1d.back();
			perm1d.pop_back();
			result.push_back(pair<int, int>(idx % width, idx / width));
		}
	}

	void getLab(const Vec4b& pixel, CIELABConvertor::Lab& lab1)
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

	void compute_b_array_ea_saliency(vector<vector<Mat> >& weightMaps, vector<vector<Mat> >& b, int filterRadius, Mat saliencyMap)
	{
		int imgHeight = weightMaps.size();
		int imgWidth = weightMaps[0].size();
		for (int i_y = 0; i_y < imgHeight; ++i_y) {
			for (int i_x = 0; i_x < imgWidth; ++i_x) {
				auto& wmap_i = weightMaps[i_y][i_x];
				int extendedFilterRadius = filterRadius * 2;
				b[i_y][i_x] = Mat::zeros(extendedFilterRadius * 2 + 1, extendedFilterRadius * 2 + 1, CV_32F);
				int j_y_min = i_y - extendedFilterRadius, j_y_max = i_y + extendedFilterRadius;
				int j_x_min = i_x - extendedFilterRadius, j_x_max = i_x + extendedFilterRadius;
				int wmI_y_min = i_y - filterRadius, wmI_y_max = i_y + filterRadius, wmI_x_min = i_x - filterRadius, wmI_x_max = i_x + filterRadius;
				for (int j_y = max(0, j_y_min); j_y < imgHeight && j_y <= j_y_max; ++j_y) {
					for (int j_x = max(0, j_x_min); j_x < imgWidth && j_x <= j_x_max; ++j_x) {
						auto& wmap_j = weightMaps[j_y][j_x];
						int wmJ_y_min = j_y - filterRadius, wmJ_y_max = j_y + filterRadius, wmJ_x_min = j_x - filterRadius, wmJ_x_max = j_x + filterRadius;
						for (int wmJ_y = max(0, wmJ_y_min); wmJ_y < imgHeight && wmJ_y <= wmJ_y_max; ++wmJ_y) {
							for (int wmJ_x = max(0, wmJ_x_min); wmJ_x < imgWidth && wmJ_x <= wmJ_x_max; ++wmJ_x) {
								// if in overlap area
								if (abs(wmJ_y - i_y) <= filterRadius && abs(wmJ_x - i_x) <= filterRadius)
									b[i_y][i_x].at<float>(j_y - j_y_min, j_x - j_x_min) += saliencyMap.at<float>(wmJ_y, wmJ_x) * wmap_i.at<float>(wmJ_y - wmI_y_min, wmJ_x - wmI_x_min) * wmap_j.at<float>(wmJ_y - wmJ_y_min, wmJ_x - wmJ_x_min);
							}
						}

						if (b[i_y][i_x].at<float>(j_y - j_y_min, j_x - j_x_min) == 0)
							b[i_y][i_x].at<float>(j_y - j_y_min, j_x - j_x_min) = 1e-10f;
					}
				}
			}
		}
	}

	float b_value_ea(vector<vector<Mat> >& b, const int i_x, const int i_y, const int j_x, const int j_y)
	{
		int extendedFilterRadius = (b[i_y][i_x].rows - 1) / 2;
		int k_x = j_x - i_x + extendedFilterRadius;
		int k_y = j_y - i_y + extendedFilterRadius;
		if (k_x < 0 || k_y < 0 || k_x >= b[i_y][i_x].cols || k_y >= b[i_y][i_x].rows)
			return 1e-10f;
		return b[i_y][i_x].at<float>(k_y, k_x);
	}

	void compute_a_image_ea(const Mat4b image, vector<vector<Mat> >& b, array2d<vector_fixed<float, 4> >& a, const uint nMaxColors)
	{
		Vec4b lastPixel = m_transparentColor;
		int threshold = 256 / nMaxColors;

		int extendedFilterRadius = (b[0][0].rows - 1) / 2;
		for (int i_y = 0; i_y < a.get_height(); ++i_y) {
			for (int i_x = 0; i_x < a.get_width(); ++i_x) {
				auto& iPixel = image(i_y, i_x);
				for (int j_y = max(0, i_y - extendedFilterRadius); j_y < a.get_height() && j_y <= i_y + extendedFilterRadius; ++j_y) {
					for (int j_x = max(0, i_x - extendedFilterRadius); j_x < a.get_width() && j_x <= i_x + extendedFilterRadius; ++j_x) {
						auto tmpBvalue = b_value_ea(b, i_x, i_y, j_x, j_y);
						if (tmpBvalue == 0)
							continue;

						auto pixelIndex = j_y * a.get_width() + j_x;
						auto jPixel = image(j_y, j_x);
						if (jPixel[3] <= alphaThreshold)
							jPixel = (nMaxColors >= 16 && pixelIndex % threshold == 0) ? lastPixel : iPixel;
						else if (nMaxColors > 64 || pixelIndex % 2 == 0)
							lastPixel = jPixel;

						if (nMaxColors > 2) {
							CIELABConvertor::Lab lab1;
							getLab(jPixel, lab1);

							a(i_x, i_y)[0] += tmpBvalue * lab1.L;
							a(i_x, i_y)[1] += tmpBvalue * lab1.A;
							a(i_x, i_y)[2] += tmpBvalue * lab1.B;
						}
						else {
							a(i_x, i_y)[2] += tmpBvalue * jPixel[2];
							a(i_x, i_y)[1] += tmpBvalue * jPixel[1];
							a(i_x, i_y)[0] += tmpBvalue * jPixel[0];
						}
						a(i_x, i_y)[3] += tmpBvalue * jPixel[3];
					}
				}
				a(i_x, i_y) *= -2.0f;
			}
		}
	}

	void sum_coarsen(const array2d<vector_fixed<float, 4> >& fine, array2d<vector_fixed<float, 4> >& coarse)
	{
		for (int y = 0; y < coarse.get_height(); ++y) {
			for (int x = 0; x < coarse.get_width(); ++x) {
				coarse(x, y) = fine(x * 2, y * 2);
				if (x * 2 + 1 < fine.get_width())
					coarse(x, y) += fine(x * 2 + 1, y * 2);
				if (y * 2 + 1 < fine.get_height())
					coarse(x, y) += fine(x * 2, y * 2 + 1);
				if (x * 2 + 1 < fine.get_width() && y * 2 + 1 < fine.get_height())
					coarse(x, y) += fine(x * 2 + 1, y * 2 + 1);
			}
		}
	}

	void zoom_float_icm(const Mat smallVal, Mat big)
	{
		for (int y = 0; y < big.rows; ++y) {
			int small_y = y / 2;
			if (small_y >= smallVal.rows)
				continue;
			for (int x = 0; x < big.cols; ++x) {
				int small_x = x / 2;
				if (small_x >= smallVal.cols)
					continue;
				big.at<uchar>(y, x) = smallVal.at<uchar>(small_y, small_x);
			}
		}
	}

	void compute_initial_s_ea_icm(array2d<vector_fixed<float, 4> >& s, const Mat indexImg8, vector<vector<Mat> >& b)
	{
		const int length = hasSemiTransparency ? 4 : 3;
		int palette_size = s.get_width();
		int coarse_width = indexImg8.cols;
		int coarse_height = indexImg8.rows;
		int center_x = (b[0][0].rows - 1) / 2, center_y = (b[0][0].rows - 1) / 2;
		int extendedFilterRadius = (b[0][0].rows - 1) / 2;
		vector_fixed<float, 4> zero_vector;
		for (int v = 0; v < palette_size; ++v) {
			for (int alpha = v; alpha < palette_size; ++alpha)
				s(v, alpha) = zero_vector; // alpha > v
		}
		for (int i_y = 0; i_y < coarse_height; ++i_y) {
			for (int i_x = 0; i_x < coarse_width; ++i_x) {
				int j_y_min = i_y - extendedFilterRadius, j_y_max = i_y + extendedFilterRadius;
				int j_x_min = i_x - extendedFilterRadius, j_x_max = i_x + extendedFilterRadius;
				for (int j_y = max(0, j_y_min); j_y < coarse_height && j_y <= j_y_max; ++j_y) {
					for (int j_x = max(0, j_x_min); j_x < coarse_width && j_x <= j_x_max; ++j_x) {
						if (i_x == j_x && i_y == j_y)
							continue;
						auto b_ij = b_value_ea(b, i_x, i_y, j_x, j_y);
						int v = indexImg8.at<uchar>(i_y, i_x);
						int alpha = indexImg8.at<uchar>(j_y, j_x);
						for (int p = 0; p < length; ++p)
							s(v, alpha)[p] += b_ij;
					}
				}
				int v = indexImg8.at<uchar>(i_y, i_x);
				auto b_ii = b_value_ea(b, i_x, i_y, i_x, i_y);
				for (int p = 0; p < length; ++p)
					s(v, v)[p] += b_ii;
			}
		}
	}

	void refine_palette_icm_mat(array2d<vector_fixed<float, 4> >& s, const Mat indexImg8,
		const array2d<vector_fixed<float, 4> >& a, vector<vector_fixed<float, 4> >& palette, int& palatte_changed)
	{
		// We only computed the half of S above the diagonal - reflect it
		for (int v = 0; v < s.get_width(); v++) {
			for (int alpha = 0; alpha < v; alpha++)
				s(v, alpha) = s(alpha, v);
		}

		vector<vector_fixed<float, 4> > r(palette.size());

		for (int i_y = 0; i_y < indexImg8.rows; ++i_y) {
			for (int i_x = 0; i_x < indexImg8.cols; ++i_x) {
				int label = indexImg8.at<uchar>(i_y, i_x);
				r[label] += a(i_x, i_y);
			}
		}

		const int length = hasSemiTransparency ? 4 : 3;
		const auto maxDelta = hasSemiTransparency ? 1.0 / palette.size() : 1.0f / 64.0f;

		for (short k = 0; k < length; ++k) {
			auto j = palette.size() > 2 ? k : 3;

			Mat S_k(s.get_height(), s.get_width(), CV_32FC1);
			for (int y = 0; y < s.get_height(); ++y)
			{
				for (int x = 0; x < s.get_width(); ++x)
					S_k.at<float>(y, x) = s(x, y)[k];
			}

			Mat R_k(r.size(), 1, CV_32FC1);
			for (int y = 0; y < r.size(); ++y)
				R_k.at<float>(y, 0) = r[y][k];

			Mat palette_channel = -1.0 * ((2.0 * S_k).inv(DECOMP_SVD)) * R_k;
			for (uint v = 0; v < palette.size(); ++v) {
				auto val = palette_channel.at<float>(v, 0);
				if (val < minLabValues[j] || isnan(val))
					val = minLabValues[j];
				else if (val > maxLabValues[j])
					val = maxLabValues[j];

				if (k > 2)
					val = max((float) alphaThreshold, val);

				auto palette_delta = abs(palette[v][k] - val);
				if (palette_delta > maxDelta)
					++palatte_changed;
				palette[v][k] = val;
			}
		}
	}

	void spatial_color_quant_ea_icm_saliency(const Mat image, vector<vector<Mat> >& weightMaps, Mat1f saliencyMap,
		Mat1b quantized_image, vector<vector_fixed<float, 4> >& palette, const int filter_radius = 1)
	{
		const auto length = hasSemiTransparency ? 4 : 3;
		const auto bitmapWidth = image.cols;
		const auto bitmapHeight = image.rows;
		auto allNeiLevel = 1;
		auto max_coarse_level = 4;
		auto neiSize = 10;

		auto pIndexImg8 = make_unique<Mat1b>(bitmapHeight >> max_coarse_level, bitmapWidth >> max_coarse_level, CV_8UC1);
		fill_random_icm(*pIndexImg8, palette.size());

		// Compute a_i, b_{ij} according to Puzicha's (12)
		vector<vector<Mat> > b0(bitmapHeight, vector<Mat>(bitmapWidth));
		compute_b_array_ea_saliency(weightMaps, b0, filter_radius, saliencyMap);

		array2d< vector_fixed<float, 4> > a0(bitmapWidth, bitmapHeight);
		compute_a_image_ea(image, b0, a0, palette.size());
		// Compute a_I^l, b_{IJ}^l according to  Puzicha's (18)
		vector< array2d< vector_fixed<float, 4> > > a_vec;
		vector< vector<vector<Mat> > > b_vec;
		a_vec.push_back(a0);
		b_vec.push_back(b0);

		for (int coarse_level = 1; coarse_level <= max_coarse_level; ++coarse_level) {
			array2d< vector_fixed<float, 4> > ai(bitmapWidth >> coarse_level, bitmapHeight >> coarse_level);

			int newExtendedFilterSize = b_vec.back()[0][0].rows - 2;
			newExtendedFilterSize = max(length, newExtendedFilterSize);

			vector<vector<Mat> > bi(ai.get_height(), vector<Mat>(ai.get_width()));
			int newExtendedFilterRadius = (newExtendedFilterSize - 1) / 2;

			for (int I_y = 0; I_y < ai.get_height(); ++I_y) {
				for (int I_x = 0; I_x < ai.get_width(); ++I_x) {
					bi[I_y][I_x] = Mat::zeros(newExtendedFilterSize, newExtendedFilterSize, CV_32F);
					int J_y_min = I_y - newExtendedFilterRadius, J_y_max = I_y + newExtendedFilterRadius;
					int J_x_min = I_x - newExtendedFilterRadius, J_x_max = I_x + newExtendedFilterRadius;

					for (int J_y = max(0, J_y_min); J_y < ai.get_height() && J_y <= J_y_max; ++J_y) {
						for (int J_x = max(0, J_x_min); J_x < ai.get_width() && J_x <= J_x_max; ++J_x) {
							for (int i_y = I_y * 2; i_y < a0.get_height() && i_y < I_y * 2 + 2; ++i_y) {
								for (int i_x = I_x * 2; i_x < a0.get_width() && i_x < I_x * 2 + 2; ++i_x) {
									for (int j_y = J_y * 2; j_y < a0.get_height() && j_y < J_y * 2 + 2; ++j_y) {
										for (int j_x = J_x * 2; j_x < a0.get_width() && j_x < J_x * 2 + 2; ++j_x) {
											bi[I_y][I_x].at<float>(J_y - J_y_min, J_x - J_x_min) += b_value_ea(b_vec.back(), i_x, i_y, j_x, j_y);
										}
									}
								}
							}

							if (bi[I_y][I_x].at<float>(J_y - J_y_min, J_x - J_x_min) == 0)
								bi[I_y][I_x].at<float>(J_y - J_y_min, J_x - J_x_min) = 1e-10f;

						}
					}
				}
			}

			b_vec.push_back(bi);
			sum_coarsen(a0, ai);
			a_vec.push_back(ai);
		}


		// Multiscale ICM
		auto coarse_level = max_coarse_level;
		array2d<vector_fixed<float, 4> > s(palette.size(), palette.size());
		compute_initial_s_ea_icm(s, *pIndexImg8, b_vec[coarse_level]);

		const auto total_pixels = pIndexImg8->cols * pIndexImg8->rows;
		auto paletteSize = palette.size() * 1.0f;
		const auto maxDelta = hasSemiTransparency ? 4.0 : 1.0;
		auto rate = 2.0 / log2(paletteSize);

		while (coarse_level >= 0) {
			// calculate the distance between centroids
			vector<vector<pair<float, int> > > centroidDist(paletteSize, vector<pair<float, int> >(paletteSize, pair<float, int>(0.0f, -1)));
			for (int l1 = 0; l1 < palette.size(); ++l1) {
				CIELABConvertor::Lab lab1;
				lab1.alpha = hasSemiTransparency ? static_cast<uchar>(palette[l1][3]) : UCHAR_MAX;
				lab1.L = palette[l1][0], lab1.A = palette[l1][1], lab1.B = palette[l1][2];

				for (int l2 = l1; l2 < palette.size(); ++l2) {
					CIELABConvertor::Lab lab2;
					lab2.alpha = hasSemiTransparency ? static_cast<uchar>(palette[l2][3]) : UCHAR_MAX;
					lab2.L = palette[l2][0], lab2.A = palette[l2][1], lab2.B = palette[l2][2];

					auto curDist = sqr(lab2.A - lab1.A) + sqr(lab2.B - lab1.B);
					if (hasSemiTransparency)
						curDist += sqr(lab2.L - lab1.L) + sqr(lab2.alpha - lab1.alpha) / exp(1.5);
					else
						curDist = abs(lab2.L - lab1.L) + sqrt(curDist);

					centroidDist[l1][l2] = pair<float, int>(curDist, l2);
					centroidDist[l2][l1] = pair<float, int>(curDist, l1);
				}
			}
			// sort centroidDist row by row
			for (int l1 = 0; l1 < palette.size(); l1++)
				sort(centroidDist[l1].begin(), centroidDist[l1].end(), mycmp);

			auto& a = a_vec[coarse_level];
			auto& b = b_vec[coarse_level];

			int center_x = (b[0][0].cols - 1) / 2, center_y = (b[0][0].rows - 1) / 2;

			int step_counter = 0;
			int repeat_outter = 0;
			int palette_changed = 0;
			while (repeat_outter++ == 0 || palette_changed > palette.size() * 0.1 * rate) {
				palette_changed = 0;
				//----update labeling
				int pixels_changed = 0, pixels_visited = 0;
				int repeat_inner = 0;

				while (repeat_inner++ == 0 || pixels_changed > 0.0001 * rate * total_pixels) {
					rate += 0.001;
					pixels_changed = 0;
					pixels_visited = 0;

					deque<pair<int, int> > visit_queue;
					random_permutation_2d(pIndexImg8->cols, pIndexImg8->rows, visit_queue);

					// Compute 2*sum(j in extended neighborhood of i, j != i) b_ij
					while (!visit_queue.empty()) {
						// pick a pixel every time
						const auto& pos = visit_queue.front();
						int i_x = pos.first, i_y = pos.second;
						visit_queue.pop_front();

						// Compute based on Puzicha's (28)
						vector_fixed<float, 4> p_i;
						for (int j_y = max(0, i_y - center_y); j_y < pIndexImg8->rows && j_y <= i_y + center_y; ++j_y) {
							for (int j_x = max(0, i_x - center_x); j_x < pIndexImg8->cols && j_x <= i_x + center_x; ++j_x) {
								if (i_x == j_x && i_y == j_y)
									continue;

								auto b_ij = b_value_ea(b, i_x, i_y, j_x, j_y);
								auto& pixelIndex = pIndexImg8->at<uchar>(j_y, j_x);
								for (int p = 0; p < length; ++p)
									p_i[p] += b_ij * palette[pixelIndex][p];
							}
						}

						p_i *= 2.0;
						p_i += a(i_x, i_y);

						auto old_max_v = pIndexImg8->at<uchar>(i_y, i_x);

						auto min_meanfield = (numeric_limits<float>::max)();
						auto middle_b = b_value_ea(b, i_x, i_y, i_x, i_y);
						auto bestLabel = old_max_v;

						if (coarse_level >= allNeiLevel) {
							// search for all palette color
							for (uint v = 0; v < palette.size(); ++v) {
								auto mf_val = palette[v].dot_product(p_i + palette[v] * middle_b);
								if (mf_val < min_meanfield) {
									min_meanfield = mf_val;
									bestLabel = v;
								}
							}
						}
						else {
							// just looking for the palette color which is near current palette color
							if (neiSize > palette.size())
								neiSize = palette.size();
							for (int v = 0; v < neiSize; ++v) {
								auto tryLabel = centroidDist[old_max_v][v].second;
								auto mf_val = palette[tryLabel].dot_product(p_i + palette[tryLabel] * middle_b);
								if (mf_val < min_meanfield) {
									min_meanfield = mf_val;
									bestLabel = tryLabel;
								}
							}
						}

						if (length > 3)
							palette[bestLabel][3] = max(alphaThreshold + 1.0f, palette[bestLabel][3]);
						pIndexImg8->at<uchar>(i_y, i_x) = bestLabel;
						if ((palette[bestLabel] - palette[old_max_v]).norm_squared() >= maxDelta)
							++pixels_changed;

						++pixels_visited;
					}
				}

				//----update palette----
				compute_initial_s_ea_icm(s, *pIndexImg8, b_vec[coarse_level]);
				refine_palette_icm_mat(s, *pIndexImg8, a, palette, palette_changed);
			}

			if (--coarse_level < 0)
				break;
			auto pOldIndexImg8 = make_unique<Mat1b>(bitmapHeight >> coarse_level, bitmapWidth >> coarse_level, CV_8UC1);
			swap(pOldIndexImg8, pIndexImg8);
			zoom_float_icm(*pOldIndexImg8, *pIndexImg8);
		}

		for (int i_y = 0; i_y < bitmapHeight; ++i_y) {
			for (int i_x = 0; i_x < bitmapWidth; ++i_x)
				quantized_image(i_y, i_x) = pIndexImg8->at<uchar>(i_y, i_x);
		}
	}

	void filter_bila(const Mat img, vector<vector<Mat> >& weightMaps, const float sigma_s = 1.0f, const float sigma_r = 2.0f)
	{
		// pixel-wise filter
		const int radius = 1;
		auto wMin = 100.0f;
		auto colorDivisor = 2 * sigma_r * sigma_r;
		auto spacerDivisor = 2 * sigma_s * sigma_s;

		for (int y = 0; y < img.rows; ++y) {
			for (int x = 0; x < img.cols; ++x) {
				Vec4b pixelXY;
				GrabPixel(pixelXY, img, y, x);
				CIELABConvertor::Lab lab1;
				getLab(pixelXY, lab1);

				auto weightSum = 0.0f;
				int yyMin = y - radius, yyMax = y + radius, xxMin = x - radius, xxMax = x + radius;
				weightMaps[y][x] = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);
				for (int yy = max(0, yyMin); yy < img.rows && yy <= yyMax; ++yy) {
					for (int xx = max(0, xxMin); xx < img.cols && xx <= xxMax; ++xx) {
						auto spaceD = sqr(y - yy) + sqr(x - xx);

						Vec4b pixelXXYY;
						GrabPixel(pixelXXYY, img, yy, xx);
						CIELABConvertor::Lab lab2;
						getLab(pixelXXYY, lab2);

						auto colorD = sqr(lab2.A - lab1.A) + sqr(lab2.B - lab2.B);
						if (hasSemiTransparency)
							colorD += sqr(lab2.L - lab1.L) + sqr(pixelXY[3] - pixelXXYY[3]) / exp(1.5);
						else
							colorD = abs(lab2.L - lab1.L) + sqrt(colorD);

						auto tmpW = exp(-spaceD / spacerDivisor - colorD / colorDivisor);

						weightSum += tmpW;

						weightMaps[y][x].at<float>(xx - xxMin, yy - yyMin) = tmpW;

						if (tmpW < wMin)
							wMin = tmpW;
					}
				}

				weightMaps[y][x] /= weightSum;
			}
		}
	}

	ushort nearestColorIndex(const Mat palette, const Vec4b& c0, const uint pos)
	{
		auto argb = GetArgb8888(c0);
		auto got = nearestMap.find(argb);
		if (got != nearestMap.end())
			return got->second;

		ushort k = 0;
		auto c = c0;
		if (c[3] <= alphaThreshold)
			c = m_transparentColor;

		double mindist = SHRT_MAX;
		CIELABConvertor::Lab lab1, lab2;
		getLab(c, lab1);

		const auto nMaxColors = palette.rows;
		for (uint i = 0; i < nMaxColors; ++i) {
			Vec4b c2;
			GrabPixel(c2, palette, i, 0);
			auto curdist = sqr(c2[3] - c[3]) / exp(1.5);
			if (curdist > mindist)
				continue;

			getLab(c2, lab2);
			curdist += sqr(lab2.L - lab1.L);
			if (curdist > mindist)
				continue;

			curdist += sqr(lab2.A - lab1.A);
			if (curdist > mindist)
				continue;

			curdist += sqr(lab2.B - lab1.B);
			if (curdist > mindist)
				continue;

			mindist = curdist;
			k = i;
		}
		nearestMap[argb] = k;
		return k;
	}

	inline auto GetColorIndex(const Vec4b& c)
	{
		return GetArgbIndex(c, hasSemiTransparency, m_transparentPixelIndex >= 0);
	}

	void arrange2Colors(const Mat4b pixels, Mat palette, Mat1b qPixels)
	{
		auto bitmapWidth = pixels.cols;
		auto bitmapHeight = pixels.rows;
		
		if (m_transparentPixelIndex >= 0) {
			auto k = qPixels(m_transparentPixelIndex / bitmapWidth, m_transparentPixelIndex % bitmapWidth);
			palette.at<Vec4b>(k, 0) = m_transparentColor;

			if (k > 0) {
				swap(palette.at<Vec4b>(0, 0), palette.at<Vec4b>(k, 0));
				for (int y = 0; y < pixels.rows; ++y)
				{
					for (int x = 0; x < pixels.cols; ++x)
					{
						auto& c = pixels(y, x);
						auto& qPixel = qPixels(y, x);
						if (qPixel == k || c[3] <= alphaThreshold)
							qPixel = 0;
						else if (qPixel == 0 && c[3] > alphaThreshold)
							qPixel = k;
					}
				}
			}
			else {
				for (int y = 0; y < pixels.rows; ++y)
				{
					for (int x = 0; x < pixels.cols; ++x)
					{
						auto& c = pixels(y, x);
						if (c[3] <= alphaThreshold)
							qPixels(y, x) = 0;
					}
				}
			}
		}
	}

	Mat EdgeAwareSQuantizer::QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);

		m_transparentPixelIndex = -1;
		Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
		GrabPixels(srcImg, pixels4b, hasSemiTransparency, m_transparentPixelIndex, m_transparentColor, alphaThreshold, nMaxColors);
		vector<Vec4b> pixels(pixels4b.begin(), pixels4b.end());

		// see equation (7) in the paper
		Mat1f saliencyMap(bitmapHeight, bitmapWidth, CV_32F);
		auto saliencyBase = 0.1f;
		for (int y = 0; y < saliencyMap.rows; ++y) {
			for (int x = 0; x < saliencyMap.cols; ++x) {
				auto& c = pixels4b(y, x);

				CIELABConvertor::Lab lab1;
				getLab(c, lab1);
				saliencyMap(y, x) = saliencyBase + (1 - saliencyBase) * lab1.L / 100.0f;
			}
		}

		if (nMaxColors > 256)
			nMaxColors = 256;

		Mat pal(nMaxColors, 1, srcImg.type(), scalar);

		MedianCutQuant::MedianCut mmcQuantizer;
		mmcQuantizer.quantizeImg(pixels, bitmapWidth, saliencyMap, pal, nMaxColors);

		const auto divisor = hasSemiTransparency ? 255.0f : 1.0f;
		vector<vector_fixed<float, 4> > palette(nMaxColors);
		for (uint k = 0; k < nMaxColors; ++k) {
			Vec4b c;
			GrabPixel(c, pal, k, 0);
			palette[k][0] = c[2] / 255.0f;
			palette[k][1] = c[1] / 255.0f;
			palette[k][2] = c[0] / 255.0f;
			palette[k][3] = c[3] / divisor;
		}

		vector<vector<Mat> > weightMaps(bitmapHeight, vector<Mat>(bitmapWidth));
		filter_bila(pixels4b, weightMaps);
		Mat1b qPixels(bitmapHeight, bitmapWidth);
		spatial_color_quant_ea_icm_saliency(pixels4b, weightMaps, saliencyMap, qPixels, palette);

		/* Fill palette */
		for (uint k = 0; k < nMaxColors; ++k) {
			Vec4b c1;
			if (nMaxColors > 2) {
				CIELABConvertor::Lab lab1;
				lab1.alpha = (m_transparentPixelIndex >= 0) ? static_cast<uchar>(palette[k][3]) : UCHAR_MAX;
				lab1.L = palette[k][0], lab1.A = palette[k][1], lab1.B = palette[k][2];
				CIELABConvertor::LAB2RGB(c1, lab1);
				SetPixel(pal, k, 0, c1);
			}
			else {
				c1 = Vec4b(clamp(static_cast<int>(palette[k][2]), 0, UCHAR_MAX), clamp(static_cast<int>(palette[k][1]), 0, UCHAR_MAX),
					clamp(static_cast<int>(palette[k][0]), 0, UCHAR_MAX), clamp(static_cast<int>(palette[k][3]), 0, UCHAR_MAX));
				SetPixel(pal, k, 0, c1);
			}
		}

		if (nMaxColors > 2)
			arrange2Colors(pixels4b, pal, qPixels);
		else {
			if (m_transparentPixelIndex >= 0) {
				pal.at<Vec4b>(1, 0) = m_transparentColor;
			}
			else {
				if (GetRgb888(pal.at<Vec3b>(1, 0)) < GetRgb888(pal.at<Vec3b>(0, 0))) {
					pal.at<Vec3b>(1, 0) = Vec3b(0, 0, 0);
					pal.at<Vec3b>(0, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);
				}
				else {
					pal.at<Vec3b>(1, 0) = Vec3b(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX);
					pal.at<Vec3b>(0, 0) = Vec3b(0, 0, 0);
				}
			}
		}

		if (!dither && nMaxColors > 2) {
			Peano::GilbertCurve::dither(pixels4b, pal, nearestColorIndex, GetColorIndex, qPixels, nullptr);
			nearestMap.clear();
		}

		pixelMap.clear();
		ProcessImagePixels(bytes, pal, qPixels, m_transparentPixelIndex >= 0);
		return pal;
	}

}