#include "stdafx.h"
#include "ApngWriter.h"

#include <fstream>
#include <iostream>
#include <map>

namespace PngEncode
{
	ApngWriter::ApngWriter(const long fps, const bool loop)
	{
		_fps = fps;
		_loop = loop;
	}

	// ---------------------------------------------------------------------------------------------------------
	// NOTE copy from: **OFFICIAL** png documentation has a link to "sample crc code"
	// https://www.w3.org/TR/PNG-CRCAppendix.html
	// ---------------------------------------------------------------------------------------------------------

	/* Table of CRCs of all 8-bit messages. */
	unsigned long crc_table[256];

	/* Flag: has the table been computed? Initially false. */
	int crc_table_computed = 0;

	/* Make the table for a fast CRC. */
	static void make_crc_table() {
		unsigned long c;
		int n, k;

		for (n = 0; n < 256; ++n) {
			c = (unsigned long)n;
			for (k = 0; k < 8; k++) {
				if (c & 1)
					c = 0xedb88320L ^ (c >> 1);
				else
					c >>= 1;
			}
			crc_table[n] = c;
		}
		crc_table_computed = 1;
	}

	/* Update a running CRC with the bytes buf[0..len-1]--the CRC
		should be initialized to all 1's, and the transmitted value
		is the 1's complement of the final running CRC (see the
		crc() routine below)). */

	static unsigned long update_crc(unsigned long crc, unsigned char *buf, int len) {
		unsigned long c = crc;
		int n;

		if (!crc_table_computed)
			make_crc_table();
		for (n = 0; n < len; n++) {
			c = crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
		}
		return c;
	}

	/* Return the CRC of the bytes buf[0..len-1]. */
	static unsigned long crc(unsigned char* buf, int len) {
		return update_crc(0xffffffffL, buf, len) ^ 0xffffffffL;
	}

	static const uchar PNG_SIGNATURE[] = {137, 80, 78, 71, 13, 10, 26, 10};

	// NOTE png needs big endian https://stackoverflow.com/questions/2384111/png-file-format-endianness
	static uint32_t to_uint32_big_endian(const uchar* a) {
		return a[3] + (((uint32_t) a[2]) << 8) + (((uint32_t) a[1]) << 16) + (((uint32_t) a[0]) << 24);
	}

	static void from_uint16_big_endian(uchar* a, uint16_t n) {
		a[0] = (n >> 8) & 0xFF;
		a[1] = n & 0xFF;
	}

	static void from_uint32_big_endian(uchar *a, uint32_t n) {
		a[0] = (n >> 24) & 0xFF;
		a[1] = (n >> 16) & 0xFF;
		a[2] = (n >> 8) & 0xFF;
		a[3] = n & 0xFF;
	}

	static int png_iterator_first_chunk(const vector<uchar>& bytes) {
		uint n = sizeof(PNG_SIGNATURE) / sizeof(PNG_SIGNATURE[0]);
		for (int j = 0; j < n; ++j) {
			CV_Assert(bytes[j] == PNG_SIGNATURE[j]);
		}
		return n;
	}

	static int png_iterator_next_chunk(const vector<uchar>& bytes, int chunk_begin_loc, string& chunk_type) {
		int idx = chunk_begin_loc;

		uint32_t length = to_uint32_big_endian(&bytes[idx]);
		idx += 4;

		string type(bytes.begin() + idx, bytes.begin() + idx + 4);
		chunk_type = type;
		return chunk_begin_loc + (int) length + 12;
	}

	static void create_acTL_chunk(vector<uchar>& bytes, const int frameSize, const int loop) {
		uint32_t length = 8;
		int idx = 0;
		bytes.resize(length + 12);

		from_uint32_big_endian(&bytes[idx], length);
		idx += 4;

		bytes[idx++] = 'a';
		bytes[idx++] = 'c';
		bytes[idx++] = 'T';
		bytes[idx++] = 'L';

		from_uint32_big_endian(&bytes[idx], frameSize);
		idx += 4;

		from_uint32_big_endian(&bytes[idx], min(0, loop));
		idx += 4;
		
		auto crc_val = crc(&bytes[4], (int) length + 4);
		from_uint32_big_endian(&bytes[idx], crc_val);
		idx += 4;

		CV_Assert(idx == bytes.size());
	}

	static void create_fcTL_chunk(vector<uchar>& bytes, const int seq, const uint32_t width, const uint32_t height, const uint32_t fps) {
		uint32_t length = 26;
		int idx = 0;
		bytes.resize(length + 12);

		from_uint32_big_endian(&bytes[idx], length);
		idx += 4;

		bytes[idx++] = 'f';
		bytes[idx++] = 'c';
		bytes[idx++] = 'T';
		bytes[idx++] = 'L';

		from_uint32_big_endian(&bytes[idx], seq);
		idx += 4;

		from_uint32_big_endian(&bytes[idx], width);
		idx += 4;

		from_uint32_big_endian(&bytes[idx], height);
		idx += 4;

		from_uint32_big_endian(&bytes[idx], 0);
		idx += 4;
		from_uint32_big_endian(&bytes[idx], 0);
		idx += 4;
		from_uint16_big_endian(&bytes[idx], 1);
		idx += 2;
		from_uint16_big_endian(&bytes[idx], fps);
		idx += 2;

		bytes[idx++] = 0;
		bytes[idx++] = 0;
		
		auto crc_val = crc(&bytes[4], (int) length + 4);
		from_uint32_big_endian(&bytes[idx], crc_val);
		idx += 4;

		CV_Assert(idx == bytes.size());
	}

	static void create_PLTE_chunk(vector<uchar>& bytes, const Mat palette) {
		CV_Assert(palette.cols == 1);
		CV_Assert(palette.rows <= 256); // since only 8-bit depth!

		// *3: since 3 bytes per channel (color image)
		uint32_t length = palette.rows * 3;
		if(palette.channels() < 4)
			CV_Assert(palette.type() == CV_8UC3);
		else
			CV_Assert(palette.type() == CV_8UC4);

		// +12: by definition of chunk "actual" length vs "recorded" length
		bytes.resize(length + 12);

		int idx = 0;

		from_uint32_big_endian(&bytes[idx], length);
		idx += 4;

		bytes[idx++] = 'P';
		bytes[idx++] = 'L';
		bytes[idx++] = 'T';
		bytes[idx++] = 'E';

		// I know not that fast, but palette is pretty small so readability is also important
		for (int j = 0; j < palette.rows; ++j) {
			for (int ch = 0; ch < 3; ++ch) {
				if(palette.channels() < 4)
					bytes[idx + ch] = palette.at<Vec3b>(j, 0)[2 - ch];
				else
					bytes[idx + ch] = palette.at<Vec4b>(j, 0)[2 - ch];
			}
			idx += 3;
		}

		auto crc_val = crc(&bytes[4], (int) length + 4);
		from_uint32_big_endian(&bytes[idx], crc_val);
		idx += 4;

		CV_Assert(idx == bytes.size());
	}

	static bool create_tRNS_chunk(vector<uchar>& bytes, const Mat palette) {
		if(palette.channels() < 4)
			return false;
		
		CV_Assert(palette.cols == 1);
		CV_Assert(palette.rows <= 256); // since only 8-bit depth!

		uint32_t length = palette.rows;
		CV_Assert(palette.type() == CV_8UC4);

		// +12: by definition of chunk "actual" length vs "recorded" length
		bytes.resize(length + 12);

		int idx = 0;

		from_uint32_big_endian(&bytes[idx], length);
		idx += 4;

		bytes[idx++] = 't';
		bytes[idx++] = 'R';
		bytes[idx++] = 'N';
		bytes[idx++] = 'S';

		for (int j = 0; j < palette.rows; ++j)
			bytes[idx++] = palette.at<Vec4b>(j, 0)[3];

		auto crc_val = crc(&bytes[4], (int) length + 4);
		from_uint32_big_endian(&bytes[idx], crc_val);
		idx += 4;

		CV_Assert(idx == bytes.size());
		return true;
	}

	static void change_IHDR_colortype_and_crc(vector<uchar>& bytes, int ihdr_start_loc, int ihdr_end_loc, const int nMaxColors) {
		const int ihdr_data_loc = ihdr_start_loc + 4 + 4;
		const int ihdr_bitdepth_loc = ihdr_data_loc + 8;
		const int ihdr_colortype_loc = ihdr_data_loc + 9;
		const int crc_loc = ihdr_end_loc - 4;

		if (nMaxColors < 3)
			CV_Assert(bytes[ihdr_bitdepth_loc] == 1);
		else
			CV_Assert(bytes[ihdr_bitdepth_loc] == 8);

		// a. change colortype to "indexed color"
		bytes[ihdr_colortype_loc] = 3;

		// b. re-calculate the CRC value
		int chunk_length = ihdr_end_loc - ihdr_start_loc - 12;
		auto crc_val = crc(&bytes[ihdr_start_loc + 4], chunk_length + 4);
		from_uint32_big_endian(&bytes[crc_loc], crc_val);
	}

	static pair<int, int> SeekChunk(const vector<uchar>& bytes, const string& chunkType, const int& idx)
	{
		int next_chunk_start_idx = idx;
		for (;;) {
			string chunk_type;
			int next_chunk_end_idx = png_iterator_next_chunk(bytes, next_chunk_start_idx, chunk_type);
			if (next_chunk_end_idx >= bytes.size())
				return { idx, idx };
			
			if (chunk_type == chunkType)
				return { next_chunk_start_idx, next_chunk_end_idx };
			next_chunk_start_idx = next_chunk_end_idx;
		}
		return { idx, idx };
	}

	static map<int, int> SeekChunks(const vector<uchar>& bytes, const string& chunkType, const int& idx)
	{
		int next_chunk_start_idx = idx;
		map<int, int> result;
		for (;;) {
			string chunk_type;
			int next_chunk_end_idx = png_iterator_next_chunk(bytes, next_chunk_start_idx, chunk_type);
			if (next_chunk_end_idx >= bytes.size())
				return result;

			if (chunk_type == chunkType)
				result.emplace(next_chunk_start_idx, next_chunk_end_idx);
			next_chunk_start_idx = next_chunk_end_idx;
		}
		return result;
	}

	void AddImage(vector<uchar>& bytes, const Mat qPixels, const bool& hasTransparent)
	{
		imencode(".png", qPixels, bytes,
			// use higher compression, thus smaller
			{ IMWRITE_PNG_COMPRESSION, 9});
	}

	void AddImage(vector<uchar>& bytes, const Mat palette, const Mat1b qPixels, const bool& hasTransparent)
	{
		CV_Assert(qPixels.type() == CV_8UC1);
		CV_Assert(palette.cols == 1);
		if(hasTransparent)
			CV_Assert(palette.type() == CV_8UC4);
		else
			CV_Assert(palette.type() == CV_8UC3);

		imencode(".png", qPixels, bytes,
			// use higher compression, thus smaller
			 {IMWRITE_PNG_COMPRESSION, 9, IMWRITE_PNG_BILEVEL, palette.rows < 3});

		int idx = png_iterator_first_chunk(bytes);
		int next_chunk_start_idx = SeekChunk(bytes, "IHDR", idx).second;
		if (next_chunk_start_idx < bytes.size()) {
			// change 1: change IHDR "color" flag
			change_IHDR_colortype_and_crc(bytes, idx, next_chunk_start_idx, palette.rows);

			// change 2: insert the PLTE chunk **after** IHDR chunk
			vector<uchar> plte_chunk_bytes;
			create_PLTE_chunk(plte_chunk_bytes, palette);
			bytes.insert(bytes.begin() + next_chunk_start_idx, plte_chunk_bytes.begin(), plte_chunk_bytes.end());

			// no need to manipulate the data after that. let's stop!
			if (palette.channels() > 3) {
				vector<uchar> trns_chunk_bytes;
				if (create_tRNS_chunk(trns_chunk_bytes, palette))
					bytes.insert(bytes.begin() + next_chunk_start_idx, trns_chunk_bytes.begin(), trns_chunk_bytes.end());
			}
		}
	}

	static void Dump(const vector<uchar>& bytes)
	{
		int next_chunk_start_idx = png_iterator_first_chunk(bytes);
		for (;;) {
			string chunk_type;
			int next_chunk_end_idx = png_iterator_next_chunk(bytes, next_chunk_start_idx, chunk_type);
			auto fcd = chunk_type == "fcTL" || chunk_type == "fdAT";
			if (next_chunk_end_idx >= bytes.size()) {
				if (fcd) {
					auto seq = to_uint32_big_endian(&bytes[next_chunk_start_idx + 8]);
					cout << chunk_type << ":\t" << next_chunk_start_idx << "\t" << next_chunk_end_idx << "\t" << seq << endl;
				}
				else
					cout << chunk_type << ":\t" << next_chunk_start_idx << "\t" << next_chunk_end_idx << endl;
				return;
			}

			if (fcd) {
				auto seq = to_uint32_big_endian(&bytes[next_chunk_start_idx + 8]);
				cout << chunk_type << ":\t" << next_chunk_start_idx << "\t" << next_chunk_end_idx << "\t" << seq << endl;
			}
			else
				cout << chunk_type << ":\t" << next_chunk_start_idx << "\t" << next_chunk_end_idx << endl;
			next_chunk_start_idx = next_chunk_end_idx;
		}
	}

	bool ApngWriter::AddImages(vector<vector<uchar> >& pngList)
	{
		if (pngList.size() < 1)
			return false;

		_data = pngList[0];
		if (pngList.size() < 2)
			return false;

		int idx = png_iterator_first_chunk(_data);
		auto next_chunk_range = SeekChunk(_data, "IHDR", idx);
		int next_chunk_start_idx = next_chunk_range.second;
		if (next_chunk_start_idx < _data.size()) {
			// insert the acTL chunk **after** IHDR chunk
			vector<uchar> acTL_chunk_bytes;
			create_acTL_chunk(acTL_chunk_bytes, pngList.size(), _loop ? 0 : 1);
			_data.insert(_data.begin() + next_chunk_start_idx, acTL_chunk_bytes.begin(), acTL_chunk_bytes.end());
			next_chunk_start_idx = SeekChunk(_data, "IDAT", next_chunk_start_idx).first;
			
			uint32_t seq = 0;
			for(int i = 0; i < pngList.size(); ++i) {
				ostringstream ss;
				ss << "\r" << i << " of " << pngList.size() << " completed." << showpoint;
				cout << ss.str().c_str();
				
				if (i > 0) {
					auto width = to_uint32_big_endian(&pngList[i][idx + 8]);
					auto height = to_uint32_big_endian(&pngList[i][idx + 12]);

					// insert the fcTL chunk **after** last IDAT chunk
					vector<uchar> fcTL_chunk_bytes;
					create_fcTL_chunk(fcTL_chunk_bytes, seq++, width, height, _fps);
					auto fcTL_chunk_start_pos = _data.end() - 12;
					_data.insert(fcTL_chunk_start_pos, fcTL_chunk_bytes.begin(), fcTL_chunk_bytes.end());

					// insert the fdAT chunk **after** fcTL chunk
					auto idat_chunk_map = SeekChunks(pngList[i], "IDAT", idx);
					for (const auto& idat_chunk : idat_chunk_map) {
						vector<uchar> fdAT_chunk_bytes(pngList[i].begin() + idat_chunk.first, pngList[i].begin() + idat_chunk.second);

						vector<uchar> fdAT_len_bytes(4);
						auto fdAT_len = to_uint32_big_endian(&fdAT_chunk_bytes[0]);
						from_uint32_big_endian(fdAT_len_bytes.data(), fdAT_len + 4);
						for (int j = 0; j < fdAT_len_bytes.size(); ++j)
							fdAT_chunk_bytes[j] = fdAT_len_bytes[j];

						fdAT_chunk_bytes[4] = 'f';
						fdAT_chunk_bytes[5] = 'd';

						vector<uchar> fdAT_index_bytes(4);
						from_uint32_big_endian(fdAT_index_bytes.data(), seq);
						fdAT_chunk_bytes.insert(fdAT_chunk_bytes.begin() + 8, fdAT_index_bytes.begin(), fdAT_index_bytes.end());

						int idx = (int)fdAT_chunk_bytes.size() - 4;
						auto crc_val = crc(&fdAT_chunk_bytes[4], fdAT_chunk_bytes.size() - 8);
						from_uint32_big_endian(&fdAT_chunk_bytes[idx], crc_val);
						CV_Assert((idat_chunk.second - idat_chunk.first + 4) == fdAT_chunk_bytes.size());
						_data.insert(_data.end() - 12, fdAT_chunk_bytes.begin(), fdAT_chunk_bytes.end());
					}
					++seq;
				}
				else {
					auto width = to_uint32_big_endian(&_data[idx + 8]);
					auto height = to_uint32_big_endian(&_data[idx + 12]);

					// insert the fcTL chunk **before** first IDAT chunk
					vector<uchar> fcTL_chunk_bytes;
					create_fcTL_chunk(fcTL_chunk_bytes, seq, width, height, _fps);
					auto fcTL_chunk_start_pos = _data.begin() + next_chunk_start_idx;
					_data.insert(fcTL_chunk_start_pos, fcTL_chunk_bytes.begin(), fcTL_chunk_bytes.end());
				}
			}
			cout << "\rWell done!!!                             " << endl;
			//Dump(_data);
			return true;
		}
		return false;
	}

	void ApngWriter::Save(const string& destPath)
	{
		ofstream outfile(destPath, ios::binary);
		outfile.write(reinterpret_cast<const char*>(_data.data()), _data.size());
		outfile.close();
	}

}
