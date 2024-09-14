#pragma once

namespace PngEncode
{
	class ApngWriter
	{
		private:
			bool _loop;
			int _width, _height;
			long _fps;
			vector<uchar> _data;
		
			public:
				ApngWriter(const int width, const int height, const long fps = 30, const bool loop = true);
				bool AddImages(vector<vector<uchar> >& pngList);
				void Save(const string& destPath);
	};

	void AddImage(vector<uchar>& bytes, const Mat palette, const Mat1b qPixels, const bool& hasTransparent);
}
