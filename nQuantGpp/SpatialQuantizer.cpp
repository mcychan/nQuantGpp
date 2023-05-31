﻿/* Copyright (c) 2006 Derrick Coetzee
Copyright (c) 2023 Miller Cy Chan

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
#include "SpatialQuantizer.h"
#include "bitmapUtilities.h"
#include "CIELABConvertor.h"
#include "BlueNoise.h"

#include <deque>
#include <algorithm>
#include <numeric>
#include <random>
#include <math.h>
#include <time.h>
#include <limits>
#include <unordered_map>

namespace SpatialQuant
{
	uchar alphaThreshold = 0xF;
	bool hasSemiTransparency = false;
	int m_transparentPixelIndex = -1;
	Vec4b m_transparentColor(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 0);
	unordered_map<ARGB, CIELABConvertor::Lab> pixelMap;
	unordered_map<ARGB, ushort> nearestMap;

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

	template <typename T, int length>
	class vector_fixed
	{
	public:
		vector_fixed()
		{
		}

		vector_fixed(const vector_fixed<T, length>& rhs)
		{
			copy(rhs.data, rhs.data + length, data);
		}

		vector_fixed(const vector<T>& rhs)
		{
			copy(rhs.data, rhs.data + length, data);
		}

		inline T& operator[](int index)
		{
			return data[index];
		}

		inline const T& operator[](int index) const
		{
			return data[index];
		}

		inline int get_length() const { return length; }

		T norm_squared() {
			T result = 0;
			for (int i = 0; i < length; ++i)
				result += data[i] * data[i];

			return result;
		}

		vector_fixed<T, length>& operator=(const vector_fixed<T, length>& rhs)
		{
			copy(rhs.data, rhs.data + length, data);
			return *this;
		}

		vector_fixed<T, length> direct_product(const vector_fixed<T, length>& rhs) {
			vector_fixed<T, length> result;
			for (int i = 0; i < length; ++i)
				result[i] = data[i] * rhs.data[i];

			return result;
		}

		T dot_product(const vector_fixed<T, length>& rhs) {
			T result = 0;
			for (int i = 0; i < length; ++i)
				result += data[i] * rhs.data[i];

			return result;
		}

		vector_fixed<T, length>& operator+=(const vector_fixed<T, length>& rhs) {
			for (int i = 0; i < length; ++i)
				data[i] += rhs.data[i];

			return *this;
		}

		vector_fixed<T, length> operator+(const vector_fixed<T, length>& rhs) {
			vector_fixed<T, length> result(*this);
			result += rhs;
			return result;
		}

		vector_fixed<T, length>& operator-=(const vector_fixed<T, length>& rhs) {
			for (int i = 0; i < length; ++i)
				data[i] -= rhs.data[i];

			return *this;
		}

		vector_fixed<T, length> operator-(const vector_fixed<T, length>& rhs) {
			vector_fixed<T, length> result(*this);
			result -= rhs;
			return result;
		}

		vector_fixed<T, length>& operator*=(const T scalar) {
			for (int i = 0; i < length; ++i)
				data[i] *= scalar;

			return *this;
		}

		vector_fixed<T, length> operator*(const T scalar) {
			vector_fixed<T, length> result(*this);
			result *= scalar;
			return result;
		}

	private:
		T data[length] = { 0 };
	};

	template <typename T, int length>
	vector_fixed<T, length> operator*(const T scalar, vector_fixed<T, length> vec) {
		return vec * scalar;
	}


	template <typename T>
	class array2d
	{
	public:
		array2d(int width, int height)
		{
			this->width = width;
			this->height = height;
			data = make_unique<T[]>(width * height);
		}

		array2d(const array2d<T>& rhs)
		{
			width = rhs.width;
			height = rhs.height;
			data = make_unique<T[]>(width * height);
			copy(rhs.data.get(), rhs.data.get() + (width * height), data.get());
		}

		inline T& operator()(int col, int row)
		{
			return data[row * width + col];
		}

		inline const T& operator()(int col, int row) const
		{
			return data[row * width + col];
		}

		inline const T& operator[](int index) const
		{
			return data[index];
		}

		inline int get_width() const { return width; }
		inline int get_height() const { return height; }

		array2d<T>& operator*=(const T scalar) {
			for (int i = 0; i < (width * height); ++i)
				data[i] *= scalar;
			return *this;
		}

		array2d<T> operator*(const T scalar) {
			array2d<T> result(*this);
			result *= scalar;
			return result;
		}

		vector<T> operator*(const vector<T>& vec) {
			vector<T> result(get_height());
			for (int row = 0; row < get_height(); ++row) {
				T sum = 0;
				for (int col = 0; col < get_width(); ++col)
					sum += (*this)(row, col) * vec[col];

				result[row] = sum;
			}
			return result;
		}

		array2d<T>& multiply_row_scalar(int row, T mult) {
			for (int i = 0; i < get_width(); ++i)
				(*this)(i, row) *= mult;

			return *this;
		}

		array2d<T>& add_row_multiple(int from_row, int to_row, T mult) {
			if (mult != 0) {
				for (int i = 0; i < get_width(); ++i)
					(*this)(i, to_row) += mult * (*this)(i, from_row);
			}

			return *this;
		}

		// We use simple Gaussian elimination - perf doesn't matter since
		// the matrices will be K x K, where K = number of palette entries.
		array2d<T> matrix_inverse() {
			array2d<T> result(get_width(), get_height());
			auto& a = *this;

			// Set result to identity matrix
			for (int i = 0; i < get_width(); ++i)
				result(i, i) = 1;

			// Reduce to echelon form, mirroring in result
			for (int i = 0; i < get_width(); ++i) {
				result.multiply_row_scalar(i, 1 / a(i, i));
				multiply_row_scalar(i, 1 / a(i, i));
				for (int j = i + 1; j < get_height(); ++j) {
					result.add_row_multiple(i, j, -a(i, j));
					add_row_multiple(i, j, -a(i, j));
				}
			}
			// Back substitute, mirroring in result
			for (int i = get_width() - 1; i >= 0; --i) {
				for (int j = i - 1; j >= 0; --j) {
					result.add_row_multiple(i, j, -a(i, j));
					add_row_multiple(i, j, -a(i, j));
				}
			}
			// result is now the inverse
			return result;
		}

	private:
		unique_ptr<T[]> data;
		int width, height;
	};

	template <typename T>
	array2d<T> operator*(const T scalar, array2d<T> a) {
		return a * scalar;
	}

	const short minLabValues[] = { 0, -128, -128, 0 };
	const short maxLabValues[] = { 100, 127, 127, 255 };

	short getRandom(uchar k) {
		return rand() % maxLabValues[k] + minLabValues[k];
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
			width = rhs.width;
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

		void fill_random(int length) {
			const int volume = width * height * depth;
			for (int i = 0; i < volume; ++i)
				data[i] = getRandom(i % length);
		}

		inline int get_width()  const { return width; }
		inline int get_height() const { return height; }
		inline int get_depth() const { return depth; }

	private:
		unique_ptr<T[]> data;
		int width, height, depth;
	};

	int compute_max_coarse_level(int width, int height) {
		// We want the coarsest layer to have at most MAX_PIXELS pixels
		const int MAX_PIXELS = 4000;
		int result = 0;
		while (width * height > MAX_PIXELS) {
			width >>= 1;
			height >>= 1;
			++result;
		}
		return result;
	}

	void random_permutation(vector<int>& result) {
		iota(result.begin(), result.end(), 0);
		auto rng = default_random_engine{};
		shuffle(result.begin(), result.end(), rng);
	}

	void random_permutation_2d(int width, int height, deque<pair<int, int> >& result) {
		vector<int> perm1d(width * height);
		random_permutation(perm1d);
		for (const auto& val : perm1d)
			result.emplace_front(val % width, val / width);
	}

	void compute_b_array(array2d<vector_fixed<double, 4> >& filter_weights,
		array2d<vector_fixed<double, 4> >& b)
	{
		// Assume that the pixel i is always located at the center of b,
		// and vary pixel j's location through each location in b.
		int radius_width = (filter_weights.get_width() - 1) / 2,
			radius_height = (filter_weights.get_height() - 1) / 2;
		int offset_x = (b.get_width() - 1) / 2 - radius_width;
		int offset_y = (b.get_height() - 1) / 2 - radius_height;
		for (int j_y = 0; j_y < b.get_height(); ++j_y) {
			for (int j_x = 0; j_x < b.get_width(); ++j_x) {
				for (int k_y = 0; k_y < filter_weights.get_height(); ++k_y) {
					if (k_y + offset_y < j_y - radius_width)
						continue;
					if (k_y + offset_y > j_y + radius_width)
						break;

					for (int k_x = 0; k_x < filter_weights.get_width(); ++k_x) {
						if (k_x + offset_x < j_x - radius_width)
							continue;
						if (k_x + offset_x > j_x + radius_width)
							break;
						b(j_x, j_y) += filter_weights(k_x, k_y).direct_product(filter_weights(k_x + offset_x - j_x + radius_width, k_y + offset_y - j_y + radius_height));
					}
				}
			}
		}
	}

	vector_fixed<double, 4> b_value(array2d<vector_fixed<double, 4> >& b, const int i_x, const int i_y, const int j_x, const int j_y)
	{
		int radius_width = (b.get_width() - 1) / 2, radius_height = (b.get_height() - 1) / 2;
		int k_x = j_x - i_x + radius_width;
		int k_y = j_y - i_y + radius_height;
		if (k_x < 0 || k_y < 0 || k_x >= b.get_width() || k_y >= b.get_height())
			return vector_fixed<double, 4>();
		return b(k_x, k_y);
	}

	void compute_a_image(const Mat4b image, array2d<vector_fixed<double, 4> >& b, array2d<vector_fixed<double, 4> >& a, const uint nMaxColors)
	{
		Vec4b lastPixel = m_transparentColor;
		int threshold = 256 / nMaxColors;

		const int a_width = a.get_width(), a_height = a.get_height();
		const int radius_width = (b.get_width() - 1) / 2, radius_height = (b.get_height() - 1) / 2;
		for (int i_y = 0; i_y < a_height; ++i_y) {
			for (int i_x = 0; i_x < a_width; ++i_x) {
				auto& iPixel = image(i_y, i_x);
				for (int j_y = max(0, i_y - radius_height); j_y < a_height && j_y <= i_y + radius_height; ++j_y) {
					for (int j_x = max(0, i_x - radius_width); j_x < a_width && j_x <= i_x + radius_width; ++j_x) {
						auto pixelIndex = j_y * a.get_width() + j_x;
						auto jPixel = image(j_y, j_x);
						if (jPixel[3] <= alphaThreshold)
							jPixel = (nMaxColors >= 16 && pixelIndex % threshold == 0) ? lastPixel : iPixel;
						else if (nMaxColors > 64 || pixelIndex % 2 == 0)
							lastPixel = jPixel;

						vector_fixed<double, 4> pixel;
						if (nMaxColors > 2) {
							CIELABConvertor::Lab lab1;
							getLab(jPixel, lab1);
							
							pixel[0] = lab1.L;
							pixel[1] = lab1.A;
							pixel[2] = lab1.B;
						}
						else {
							pixel[2] = jPixel[2];
							pixel[1] = jPixel[1];
							pixel[0] = jPixel[0];
						}
						pixel[3] = jPixel[3];

						a(i_x, i_y) += b_value(b, i_x, i_y, j_x, j_y).direct_product(pixel);
					}
				}
				a(i_x, i_y) *= -2.0;
			}
		}
	}

	void sum_coarsen(const array2d<vector_fixed<double, 4> >& fine, array2d<vector_fixed<double, 4> >& coarse)
	{
		for (int y = 0; y < coarse.get_height(); ++y) {
			for (int x = 0; x < coarse.get_width(); ++x) {
				coarse(x, y) = fine(x * 2, y * 2);
				if (y * 2 + 1 < fine.get_height())
					coarse(x, y) += fine(x * 2, y * 2 + 1);

				if (x * 2 + 1 < fine.get_width()) {
					coarse(x, y) += fine(x * 2 + 1, y * 2);
					if (y * 2 + 1 < fine.get_height())
						coarse(x, y) += fine(x * 2 + 1, y * 2 + 1);
				}
			}
		}
	}

	template <typename T, int length>
	array2d<T> extract_vector_layer_2d(const array2d<vector_fixed<T, length> >& s, short k)
	{
		array2d<T> result(s.get_width(), s.get_height());
		for (int i = 0; i < s.get_width(); ++i) {
			for (int j = 0; j < s.get_height(); ++j)
				result(i, j) = s(i, j)[k];
		}
		return result;
	}

	template <typename T, int length>
	vector<T> extract_vector_layer_1d(const vector<vector_fixed<T, length> >& s, short k)
	{
		vector<T> result(s.size());
		for (uint i = 0; i < s.size(); ++i)
			result[i] = s[i][k];

		return result;
	}

	uint best_match_color(const array3d<double>& vars, const int i_x, const int i_y, const uint nMaxColor)
	{
		uint max_v = 0;

		auto max_weight = vars(i_x, i_y, max_v);
		for (uint v = 1; v < nMaxColor; ++v) {
			const auto& weight = vars(i_x, i_y, v);
			if (weight > max_weight) {
				max_v = v;
				max_weight = weight;
			}
		}

		return max_v;
	}

	void zoom_double(const array3d<double>& smallVal, array3d<double>& big)
	{
		const int coarse_width = big.get_width(), coarse_height = big.get_height();
		// Simple scaling of the weights array based on mixing the four
		// pixels falling under each fine pixel, weighted by area.
		// To mix the pixels a little, we assume each fine pixel
		// is 1.2 fine pixels wide and high.
		for (int y = 0; y < coarse_height; ++y) {
			const auto top = max(0.0, (y - 0.1) / 2.0), bottom = min(smallVal.get_height() - 0.001, (y + 1.1) / 2.0);
			const auto y_top = (int)floor(top), y_bottom = (int)floor(bottom);
			for (int x = 0; x < coarse_width; ++x) {
				const auto left = max(0.0, (x - 0.1) / 2.0), right = min(smallVal.get_width() - 0.001, (x + 1.1) / 2.0);
				const auto x_left = (int)floor(left), x_right = (int)floor(right);
				const auto area = (right - left) * (bottom - top);
				const auto top_left_weight = (ceil(left) - left) * (ceil(top) - top) / area;
				const auto top_right_weight = (right - floor(right)) * (ceil(top) - top) / area;
				const auto bottom_left_weight = (ceil(left) - left) * (bottom - floor(bottom)) / area;
				const auto bottom_right_weight = (right - floor(right)) * (bottom - floor(bottom)) / area;
				const auto top_weight = (right - left) * (ceil(top) - top) / area;
				const auto bottom_weight = (right - left) * (bottom - floor(bottom)) / area;
				const auto left_weight = (bottom - top) * (ceil(left) - left) / area;
				const auto right_weight = (bottom - top) * (right - floor(right)) / area;
				for (int z = 0; z < big.get_depth(); ++z) {
					if (x_left == x_right && y_top == y_bottom)
						big(x, y, z) = smallVal(x_left, y_top, z);
					else if (x_left == x_right)
						big(x, y, z) = top_weight * smallVal(x_left, y_top, z) + bottom_weight * smallVal(x_left, y_bottom, z);
					else if (y_top == y_bottom)
						big(x, y, z) = left_weight * smallVal(x_left, y_top, z) + right_weight * smallVal(x_right, y_top, z);
					else {
						big(x, y, z) = top_left_weight * smallVal(x_left, y_top, z) + top_right_weight * smallVal(x_right, y_top, z) +
							bottom_left_weight * smallVal(x_left, y_bottom, z) + bottom_right_weight * smallVal(x_right, y_bottom, z);
					}
				}
			}
		}
	}

	void compute_initial_s(array2d<vector_fixed<double, 4> >& s, const array3d<double>& coarse_variables, array2d<vector_fixed<double, 4> >& b)
	{
		const int length = hasSemiTransparency ? 4 : 3;
		const int palette_size = s.get_width();
		const int coarse_width = coarse_variables.get_width(), coarse_height = coarse_variables.get_height();
		const int center_x = (b.get_width() - 1) / 2, center_y = (b.get_height() - 1) / 2;
		const auto& center_b = b_value(b, 0, 0, 0, 0);
		vector_fixed<double, 4> zero_vector;
		for (int v = 0; v < palette_size; ++v) {
			for (int alpha = v; alpha < palette_size; ++alpha)
				s(v, alpha) = zero_vector;
		}
		for (int i_y = 0; i_y < coarse_height; ++i_y) {
			int max_j_y = min(coarse_height, i_y - center_y + b.get_height());
			for (int i_x = 0; i_x < coarse_width; ++i_x) {
				int max_j_x = min(coarse_width, i_x - center_x + b.get_width());
				for (int j_y = max(0, i_y - center_y); j_y < max_j_y; ++j_y) {
					for (int j_x = max(0, i_x - center_x); j_x < max_j_x; ++j_x) {
						if (i_x == j_x && i_y == j_y)
							continue;

						const auto& b_ij = b_value(b, i_x, i_y, j_x, j_y);
						for (int v = 0; v < palette_size; ++v) {
							auto v1 = coarse_variables(i_x, i_y, v);
							for (int alpha = v; alpha < palette_size; ++alpha) {
								auto mult = v1 * coarse_variables(j_x, j_y, alpha);
								for (int p = 0; p < length; ++p)
									s(v, alpha)[p] += mult * b_ij[p];
							}
						}
					}
				}
				for (int v = 0; v < palette_size; ++v)
					s(v, v) += coarse_variables(i_x, i_y, v) * center_b;
			}
		}
	}

	void update_s(array2d<vector_fixed<double, 4> >& s, const array3d<double>& coarse_variables, array2d<vector_fixed<double, 4> >& b,
		const int j_x, const int j_y, const int alpha, const double delta)
	{
		const int length = hasSemiTransparency ? 4 : 3;
		const int palette_size = s.get_width();
		const int coarse_width = coarse_variables.get_width(), coarse_height = coarse_variables.get_height();
		const int center_x = (b.get_width() - 1) / 2, center_y = (b.get_height() - 1) / 2;
		const int min_i_x = max(0, j_x - center_x), min_i_y = max(0, j_y - center_y);
		const int max_i_x = min(coarse_width, j_x + center_x + 1), max_i_y = min(coarse_height, j_y + center_y + 1);
		for (int i_y = min_i_y; i_y < max_i_y; ++i_y) {
			for (int i_x = min_i_x; i_x < max_i_x; ++i_x) {
				auto delta_b_ij = delta * b_value(b, i_x, i_y, j_x, j_y);
				if (i_x == j_x && i_y == j_y)
					continue;
				for (int v = 0; v <= alpha; ++v) {
					auto mult = coarse_variables(i_x, i_y, v);
					for (int p = 0; p < length; ++p)
						s(v, alpha)[p] += mult * delta_b_ij[p];
				}
				for (int v = alpha; v < palette_size; ++v) {
					auto mult = coarse_variables(i_x, i_y, v);
					for (int p = 0; p < length; ++p)
						s(alpha, v)[p] += mult * delta_b_ij[p];
				}
			}
		}
		s(alpha, alpha) += delta * b_value(b, 0, 0, 0, 0);
	}

	void refine_palette(array2d<vector_fixed<double, 4> >& s, const array3d<double>& coarse_variables,
		const array2d<vector_fixed<double, 4> >& a, vector<vector_fixed<double, 4> >& palette)
	{
		// We only computed the half of S above the diagonal - reflect it
		for (int v = 0; v < s.get_width(); ++v) {
			for (int alpha = 0; alpha < v; ++alpha)
				s(v, alpha) = s(alpha, v);
		}

		const int coarse_width = coarse_variables.get_width(), coarse_height = coarse_variables.get_height();
		const auto nMaxColor = palette.size();
		vector<vector_fixed<double, 4> > r(nMaxColor);
		for (int i_y = 0; i_y < coarse_height; ++i_y) {
			for (int i_x = 0; i_x < coarse_width; ++i_x) {
				const auto& ai = a(i_x, i_y);
				for (int v = 0; v < nMaxColor; ++v)
					r[v] += coarse_variables(i_x, i_y, v) * ai;
			}
		}

		const short length = hasSemiTransparency ? 4 : 3;
		for (int k = 0; k < length; ++k) {
			const auto& S_k = extract_vector_layer_2d(s, k);
			const auto& R_k = extract_vector_layer_1d(r, k);
			const auto& palette_channel = (-2.0 * S_k).matrix_inverse() * R_k;
			for (uint v = 0; v < nMaxColor; ++v) {
				auto val = palette_channel[v];
				auto j = palette.size() > 2 ? k : 3;
				if (val < minLabValues[j] || isnan(val))
					val = minLabValues[j];
				else if (val > maxLabValues[j])
					val = maxLabValues[j];

				if (k > 2)
					val = max((double) alphaThreshold, val);
				palette[v][k] = val;
			}
		}
	}

	void compute_initial_j_palette_sum(array2d<vector_fixed<double, 4> >& j_palette_sum, const array3d<double>& coarse_variables, const vector<vector_fixed<double, 4> >& palette)
	{
		const int coarse_width = coarse_variables.get_width(), coarse_height = coarse_variables.get_height();
		const uint nMaxColor = palette.size();
		for (int j_y = 0; j_y < coarse_height; ++j_y) {
			for (int j_x = 0; j_x < coarse_width; ++j_x) {
				vector_fixed<double, 4> palette_sum;
				for (uint alpha = 0; alpha < nMaxColor; ++alpha)
					palette_sum += coarse_variables(j_x, j_y, alpha) * palette[alpha];
				j_palette_sum(j_x, j_y) = palette_sum;
			}
		}
	}

	bool spatial_color_quant(const Mat4b image, array2d<vector_fixed<double, 4> >& filter_weights,
		Mat1b quantized_image, vector<vector_fixed<double, 4> >& palette,
		const double initial_temperature = 1.0, const double final_temperature = 0.001, const int temps_per_level = 3, const int repeats_per_temp = 1)
	{
		const int length = hasSemiTransparency ? 4 : 3;
		auto bitmapWidth = image.cols;
		auto bitmapHeight = image.rows;

		const auto nMaxColor = palette.size();
		int max_coarse_level = compute_max_coarse_level(bitmapWidth, bitmapHeight);
		auto p_coarse_variables = make_unique<array3d<double> >(
			bitmapWidth >> max_coarse_level,
			bitmapHeight >> max_coarse_level,
			nMaxColor);

		p_coarse_variables->fill_random(length);

		auto temperature = initial_temperature;

		// Compute a_i, b_{ij} according to (11)
		int extended_neighborhood_width = filter_weights.get_width() * 2 - 1;
		int extended_neighborhood_height = filter_weights.get_height() * 2 - 1;
		array2d<vector_fixed<double, 4> > b0(extended_neighborhood_width, extended_neighborhood_height);
		compute_b_array(filter_weights, b0);

		array2d<vector_fixed<double, 4> > a0(bitmapWidth, bitmapHeight);
		compute_a_image(image, b0, a0, nMaxColor);

		// Compute a_I^l, b_{IJ}^l according to (18)
		vector<array2d<vector_fixed<double, 4> > > a_vec, b_vec;
		a_vec.reserve(max_coarse_level + 1), b_vec.reserve(max_coarse_level + 1);
		a_vec.emplace_back(a0), b_vec.emplace_back(b0);

		const int diameter_width = (filter_weights.get_width() - 1), diameter_height = (filter_weights.get_height() - 1);
		for (int coarse_level = 1; coarse_level <= max_coarse_level; ++coarse_level) {
			array2d<vector_fixed<double, 4> > bi(max(length, b_vec.back().get_width() - 2), max(length, b_vec.back().get_height() - 2));
			const int bi_width = bi.get_width(), bi_height = bi.get_height();

			for (int J_y = 0; J_y < bi_height; ++J_y) {
				const int max_Jy = J_y * 2 + 2;
				for (int J_x = 0; J_x < bi_width; ++J_x) {
					const int max_Jx = J_x * 2 + 2;
					for (int i_y = diameter_height; i_y < diameter_height + 2; ++i_y) {
						for (int i_x = diameter_width; i_x < diameter_width + 2; ++i_x) {
							for (int j_y = J_y * 2; j_y < max_Jy; ++j_y) {
								for (int j_x = J_x * 2; j_x < max_Jx; ++j_x)
									bi(J_x, J_y) += b_value(b_vec.back(), i_x, i_y, j_x, j_y);
							}
						}
					}
				}
			}
			b_vec.emplace_back(bi);

			array2d<vector_fixed<double, 4> > ai(bitmapWidth >> coarse_level, bitmapHeight >> coarse_level);
			sum_coarsen(a_vec.back(), ai);
			a_vec.emplace_back(ai);
		}

		// Multiscale annealing
		int coarse_level = max_coarse_level;
		const int iters_per_level = temps_per_level;
		auto temperature_multiplier = pow(final_temperature / initial_temperature, 1.0 / (max(length, max_coarse_level * iters_per_level)));

		int iters_at_current_level = 0;
		bool skip_palette_maintenance = false;
		array2d<vector_fixed<double, 4> > s(nMaxColor, nMaxColor);
		compute_initial_s(s, *p_coarse_variables, b_vec[coarse_level]);
		auto p_palette_sum = make_unique<array2d< vector_fixed<double, 4> > >(p_coarse_variables->get_width(), p_coarse_variables->get_height());
		compute_initial_j_palette_sum(*p_palette_sum, *p_coarse_variables, palette);

		while (coarse_level >= 0 || temperature > final_temperature) {
			// Need to reseat this reference in case we changed p_coarse_variables
			auto& coarse_variables = *p_coarse_variables;
			const int coarse_width = coarse_variables.get_width(), coarse_height = coarse_variables.get_height();

			auto& a = a_vec[coarse_level];
			auto& b = b_vec[coarse_level];
			const int b_width = b.get_width(), b_height = b.get_height();
			auto middle_b = b_value(b, 0, 0, 0, 0);

			const int center_x = (b_width - 1) / 2, center_y = (b_height - 1) / 2;
			const int min_x = min(1, center_x - 1), min_y = min(1, center_y - 1);
			const int max_x = max(b_width - 1, center_x + 1), max_y = max(b_height - 1, center_y + 1);

			int step_counter = 0;
			for (int repeat = 0; repeat < repeats_per_temp; ++repeat) {
				int pixels_changed = 0, pixels_visited = 0;
				deque<pair<int, int> > visit_queue;
				random_permutation_2d(coarse_width, coarse_height, visit_queue);

				// Compute 2*sum(j in extended neighborhood of i, j != i) b_ij

				while (!visit_queue.empty()) {
					// If we get to 10% above initial size, just revisit them all
					if ((int)visit_queue.size() > coarse_width * coarse_height * 1.1) {
						visit_queue.clear();
						random_permutation_2d(coarse_width, coarse_height, visit_queue);
					}

					const auto& pos = visit_queue.front();
					int i_x = pos.first, i_y = pos.second;
					visit_queue.pop_front();

					// Compute (25)
					vector_fixed<double, 4> p_i;
					for (int y = 0; y < b_height; ++y) {
						int j_y = y - center_y + i_y;
						if (j_y < 0 || j_y >= coarse_height)
							continue;
						for (int x = 0; x < b_width; ++x) {
							int j_x = x - center_x + i_x;
							if (i_x == j_x && i_y == j_y)
								continue;
							if (j_x < 0 || j_x >= coarse_width)
								continue;
							const auto& b_ij = b_value(b, i_x, i_y, j_x, j_y);
							auto& j_pal = (*p_palette_sum)(j_x, j_y);
							for (int p = 0; p < length; ++p)
								p_i[p] += b_ij[p] * j_pal[p];
						}
					}
					p_i *= 2.0;
					p_i += a(i_x, i_y);

					auto max_meanfield_log = -numeric_limits<double>::infinity();
					auto meanfield_sum = 0.0;
					auto meanfield_logs = make_unique<double[]>(nMaxColor);

					for (uint v = 0; v < nMaxColor; ++v) {
						// Update m_{pi(i)v}^I according to (23)
						// We can subtract an arbitrary factor to prevent overflow,
						// since only the weight relative to the sum matters, so we
						// will choose a value that makes the maximum e^100.
						meanfield_logs[v] = -(palette[v].dot_product(p_i + middle_b.direct_product(palette[v]))) / temperature;
						if (meanfield_logs[v] > max_meanfield_log)
							max_meanfield_log = meanfield_logs[v];
					}

					auto meanfields = make_unique<double[]>(nMaxColor);
					for (uint v = 0; v < nMaxColor; ++v) {
						meanfields[v] = exp(meanfield_logs[v] - max_meanfield_log + 100);
						meanfield_sum += meanfields[v];
					}
					meanfield_logs.reset();

					if (meanfield_sum == 0)
						return false;

					auto old_max_v = best_match_color(coarse_variables, i_x, i_y, nMaxColor);
					auto& j_pal = (*p_palette_sum)(i_x, i_y);
					for (uint v = 0; v < nMaxColor; ++v) {
						auto new_val = meanfields[v] / meanfield_sum;
						// Prevent the matrix S from becoming singular
						if (new_val <= 0)
							new_val = 1e-10;
						if (new_val >= 1)
							new_val = 1 - 1e-10;

						auto delta_m_iv = new_val - coarse_variables(i_x, i_y, v);

						coarse_variables(i_x, i_y, v) = new_val;
						for (int p = 0; p < length; ++p)
							j_pal[p] += delta_m_iv * palette[v][p];

						if (abs(delta_m_iv) > 0.001 && !skip_palette_maintenance)
							update_s(s, coarse_variables, b, i_x, i_y, v, delta_m_iv);
					}
					meanfields.reset();

					auto max_v = best_match_color(coarse_variables, i_x, i_y, nMaxColor);
					if (length > 3)
						palette[max_v][3] = max(alphaThreshold + 1.0, palette[max_v][3]);

					// Only consider it a change if the colors are different enough
					if ((palette[max_v] - palette[old_max_v]).norm_squared() >= length) {
						++pixels_changed;
						// We don't add the outer layer of pixels , because
						// there isn't much weight there, and if it does need
						// to be visited, it'll probably be added when we visit
						// neighboring pixels.
						// The commented out loops are faster but cause a little bit of distortion
						//for (int y=center_y-1; y<center_y+1; y++) {
						//   for (int x=center_x-1; x<center_x+1; x++) {
						for (int y = min_y; y < max_y; ++y) {
							int j_y = y - center_y + i_y;
							if (j_y < 0 || j_y >= coarse_height)
								continue;
							for (int x = min_x; x < max_x; ++x) {
								int j_x = x - center_x + i_x;
								if (j_x < 0 || j_x >= coarse_width)
									continue;
								visit_queue.emplace_front(j_x, j_y);
							}
						}
					}
					++pixels_visited;

					// Show progress with dots - in a graphical interface,
					// we'd show progressive refinements of the image instead,
					// and maybe a palette preview.
					++step_counter;
				}
				if (skip_palette_maintenance)
					compute_initial_s(s, *p_coarse_variables, b_vec[coarse_level]);

				refine_palette(s, coarse_variables, a, palette);
				compute_initial_j_palette_sum(*p_palette_sum, coarse_variables, palette);
			}

			++iters_at_current_level;
			skip_palette_maintenance = false;
			if ((temperature <= final_temperature || coarse_level > 0) && iters_at_current_level >= iters_per_level) {
				if (--coarse_level < 0)
					break;

				auto p_old_coarse_variables = make_unique<array3d<double> >(bitmapWidth >> coarse_level, bitmapHeight >> coarse_level, nMaxColor);
				swap(p_old_coarse_variables, p_coarse_variables);
				zoom_double(*p_old_coarse_variables, *p_coarse_variables);
				iters_at_current_level = 0;
				p_palette_sum = make_unique<array2d<vector_fixed<double, 4> > >(p_coarse_variables->get_width(), p_coarse_variables->get_height());
				compute_initial_j_palette_sum(*p_palette_sum, *p_coarse_variables, palette);
				skip_palette_maintenance = true;
			}
			if (temperature > final_temperature)
				temperature *= temperature_multiplier;
		}

		for (int i_y = 0; i_y < bitmapHeight; ++i_y) {
			for (int i_x = 0; i_x < bitmapWidth; ++i_x)
				quantized_image(i_y, i_x) = (uchar) best_match_color(*p_coarse_variables, i_x, i_y, nMaxColor);
		}

		return true;
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

		double mindist = INT_MAX;
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

	Mat SpatialQuantizer::QuantizeImage(const Mat srcImg, vector<uchar>& bytes, uint& nMaxColors, bool dither)
	{
		auto bitmapWidth = srcImg.cols;
		auto bitmapHeight = srcImg.rows;
		auto scalar = srcImg.channels() == 4 ? Scalar(0, 0, 0, UCHAR_MAX) : Scalar(0, 0, 0);

		hasSemiTransparency = false;
		m_transparentPixelIndex = -1;
		Mat4b pixels4b(bitmapHeight, bitmapWidth, Scalar(0, 0, 0, UCHAR_MAX));
		GrabPixels(srcImg, pixels4b, hasSemiTransparency, m_transparentPixelIndex, m_transparentColor, alphaThreshold, nMaxColors);

		const int length = hasSemiTransparency ? 4 : 3;
		double dithering_level = 1.0;
		array2d<vector_fixed<double, 4> > filter3_weights(3, 3);
		double sum = 0.0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				double value = exp(-sqrt((double)((i - 2) * (i - 2) + (j - 2) * (j - 2))) / (dithering_level * dithering_level));
				for (short k = 0; k < length; ++k)
					sum += filter3_weights(i, j)[k] = value;
			}
		}
		sum /= 3.0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				for (short k = 0; k < length; ++k)
					filter3_weights(i, j)[k] /= sum;
			}
		}

		if (nMaxColors > 256)
			nMaxColors = 256;

		vector<vector_fixed<double, 4> > palette(nMaxColors);
		for (uint i = 0; i < nMaxColors; ++i) {
			for (int p = 0; p < length; ++p)
				palette[i][p] = getRandom(p);
		}

		Mat pal(nMaxColors, 1, srcImg.type(), scalar);

		Mat1b qPixels(bitmapHeight, bitmapWidth);
		if (!spatial_color_quant(pixels4b, filter3_weights, qPixels, palette)) {
			pixelMap.clear();
			return Mat{};
		}		

		nMaxColors = palette.size();
		pal = pal.rowRange(0, nMaxColors);
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
			BlueNoise::dither(pixels4b, pal, nearestColorIndex, GetColorIndex, qPixels, 0.5f);
			nearestMap.clear();
		}
		pixelMap.clear();

		ProcessImagePixels(bytes, pal, qPixels, m_transparentPixelIndex >= 0);
		return Mat{};
	}

}