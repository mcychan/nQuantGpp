// nQuantGpp.cpp
//

#include "stdafx.h"
#include <algorithm>
#include <clocale>
#include <iostream>
#include <fcntl.h>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;


#include "PnnQuantizer.h"
#include "NeuQuantizer.h"
#include "WuQuantizer.h"
#include "PnnLABQuantizer.h"
#include "EdgeAwareSQuantizer.h"
#include "SpatialQuantizer.h"
#include "DivQuantizer.h"
#include "Dl3Quantizer.h"
#include "MedianCut.h"
#include "Otsu.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

ostream& tcout = cout;

string algs[] = { "PNN", "PNNLAB", "NEU", "WU", "EAS", "SPA", "DIV", "DL3", "MMC", "OTSU" };

void PrintUsage()
{
    tcout << endl;
    tcout << "usage: nQuantGpp <input image path> [options]" << endl;
    tcout << endl;
    tcout << "Valid options:" << endl;
	tcout << "  /a : Algorithm used - Choose one of them, otherwise give you the defaults from [";
	int i = 0;	
	for(; i<sizeof(algs)/sizeof(string) - 1; ++i)
		tcout << algs[i] << ", ";
	tcout << algs[i] << "] ." << endl;
    tcout << "  /m : Max Colors (pixel-depth) - Maximum number of colors for the output format to support. The default is 256 (8-bit)." << endl;
	tcout << "  /d : Dithering or not? y or n." << endl;
    tcout << "  /o : Output image file dir. The default is <source image path directory>" << endl;
}

bool isdigit(const char* string) {
	const int string_len = strlen(string);
	for (int i = 0; i < string_len; ++i) {
		if (!isdigit(string[i]))
			return false;
	}
	return true;
}

bool isAlgo(const string& alg) {
	for (const auto& algo : algs) {
		if (alg == algo)
			return true;
	}		
	return false;
}

bool ProcessArgs(int argc, string& algo, uint& nMaxColors, bool& dither, string& targetPath, char** argv)
{
	for (int index = 1; index < argc; ++index) {
		string currentArg(argv[index]);
		transform(currentArg.begin(), currentArg.end(), currentArg.begin(), ::toupper);

		auto currentCmd = currentArg[0];
		if (currentArg.length() > 1 && 
			(currentCmd == '-' || currentCmd == '/')) {
			if (index >= argc - 1) {
				PrintUsage();
				return false;
			}

			if (currentArg[1] == 'A') {
				string strAlgo = argv[index + 1];
				transform(strAlgo.begin(), strAlgo.end(), strAlgo.begin(), ::toupper);
				if (!isAlgo(strAlgo)) {
					PrintUsage();
					return false;
				}
				algo = strAlgo;
			}
			else if (currentArg[1] == 'M') {
				if (!isdigit(argv[index + 1])) {
					PrintUsage();
					return false;
				}
				nMaxColors = atoi(argv[index + 1]);
				if (nMaxColors < 2)
					nMaxColors = 2;
				else if (nMaxColors > 65536)
					nMaxColors = 65536;
			}
			else if (currentArg[1] == 'D') {
				string strDither = argv[index + 1];
				transform(strDither.begin(), strDither.end(), strDither.begin(), ::toupper);
				if (!(strDither == "Y" || strDither == "N")) {
					PrintUsage();
					return false;
				}
				dither = strDither == "Y";
			}
			else if (currentArg[1] == 'O') {
				string tmpPath(argv[index + 1], argv[index + 1] + strlen(argv[index + 1]));
				targetPath = tmpPath;
			}
			else {
				PrintUsage();
				return false;
			}
		}
	}
	return true;
}

inline bool fileExists(const string& path)
{
	return fs::exists(fs::path(path));
}

vector<uchar> QuantizeImage(const string& algorithm, const string& sourceFile, string& targetDir, const Mat source, uint nMaxColors, bool dither)
{
	Mat dest;
	vector<uchar> bytes;
	if (algorithm == "PNN") {
		PnnQuant::PnnQuantizer pnnQuantizer;
		dest = pnnQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "PNNLAB") {
		PnnLABQuant::PnnLABQuantizer pnnLABQuantizer;
		dest = pnnLABQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "NEU") {
		NeuralNet::NeuQuantizer neuQuantizer;
		dest = neuQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "WU") {
		nQuant::WuQuantizer wuQuantizer;
		dest = wuQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "EAS") {
		EdgeAwareSQuant::EdgeAwareSQuantizer easQuantizer;
		dest = easQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "SPA") {
		SpatialQuant::SpatialQuantizer spaQuantizer;
		dest = spaQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "DIV") {
		DivQuant::DivQuantizer divQuantizer;
		dest = divQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "DL3") {
		Dl3Quant::Dl3Quantizer dl3Quantizer;
		dest = dl3Quantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "MMC") {
		MedianCutQuant::MedianCut mmcQuantizer;
		dest = mmcQuantizer.QuantizeImage(source, bytes, nMaxColors, dither);
	}
	else if (algorithm == "OTSU") {
		nMaxColors = 2;
		OtsuThreshold::Otsu otsu;
		otsu.ConvertGrayScaleToBinary(source, bytes);
	}

	if(dest.empty() && bytes.empty())
		return bytes;

	auto sourcePath = fs::canonical(fs::path(sourceFile));
	auto fileName = sourcePath.filename().string();
	fileName = fileName.substr(0, fileName.find_last_of('.'));

	targetDir = fileExists(targetDir) ? fs::canonical(fs::path(targetDir)).string() : fs::current_path().string();
	auto destPath = targetDir + "/" + fileName + "-";
	string algo(algorithm.begin(), algorithm.end());
	destPath += algo + "quant";

	auto targetExtension = ".png";
	destPath += std::to_string(nMaxColors) + targetExtension;

	try {
		if(nMaxColors > 256)
			imwrite(destPath, dest);
		else {
			ofstream outfile(destPath, ios::binary);
			outfile.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
			outfile.close();
		}
		tcout << "Converted image: " << destPath << endl;
	}
	catch (...) {
		tcout << "Failed to save image in '" << destPath << "' file" << endl;
	}

	return bytes;
}

int main(int argc, char** argv)
{
	ios::sync_with_stdio(false); // Linux gcc
	tcout.imbue(locale(""));
	setlocale(LC_CTYPE, "");
	if (argc <= 1) {
#ifndef _DEBUG
		PrintUsage();
		return 0;
#endif
	}
	
	auto szDir = fs::current_path().string();
	
	bool dither = true;
	uint nMaxColors = 256;
	string algo = "";
	string targetDir = "";
#ifdef _DEBUG
	algo = "PNN";
	string sourceFile = szDir + "/../sailing_2020.jpg";	
#else
	if (!ProcessArgs(argc, algo, nMaxColors, dither, targetDir, argv))
		return 0;

	string sourceFile(argv[1], argv[1] + strlen(argv[1]));
	if (!fileExists(sourceFile) && sourceFile.find_first_of("\\/") != string::npos)
		sourceFile = szDir + "/" + sourceFile;
#endif	
	
	if(!fileExists(sourceFile)) {
		tcout << "The source file you specified does not exist." << endl;
		return 0;
	}

#ifdef _DEBUG
	auto sourcePath = fs::canonical(fs::path(sourceFile));
	sourceFile = sourcePath.string();
#endif

	sourceFile = (sourceFile[sourceFile.length() - 1] != '/' && sourceFile[sourceFile.length() - 1] != '\\') ? sourceFile : sourceFile.substr(0, sourceFile.find_last_of("\\/"));
	auto fileExtension = sourceFile.substr(sourceFile.find_last_of('.'));
	Mat source;
	if(fileExtension == ".gif") {
		int position = 0;
		auto cap = VideoCapture(sourceFile);
		cap.set(CAP_PROP_POS_FRAMES, position);
		cap.read(source);
		cap.release();
	}
	else
		source = imread(sourceFile, IMREAD_UNCHANGED);

	if (!source.empty()) {
		if (!fileExists(targetDir))
			targetDir = fs::path(sourceFile).parent_path().string();
		
		if (algo == "") {
			//QuantizeImage("MMC", sourceFile, targetDir, source, nMaxColors, dither);
			QuantizeImage("DIV", sourceFile, targetDir, source, nMaxColors, dither);
			if (nMaxColors > 32) {
				QuantizeImage("PNN", sourceFile, targetDir, source, nMaxColors, dither);
				QuantizeImage("WU", sourceFile, targetDir, source, nMaxColors, dither);
				QuantizeImage("NEU", sourceFile, targetDir, source, nMaxColors, dither);
			}
			else {
				QuantizeImage("PNNLAB", sourceFile, targetDir, source, nMaxColors, dither);
				QuantizeImage("EAS", sourceFile, targetDir, source, nMaxColors, dither);
				QuantizeImage("SPA", sourceFile, targetDir, source, nMaxColors, dither);
			}
		}
		else
			QuantizeImage(algo, sourceFile, targetDir, source, nMaxColors, dither);
	}
	else
		tcout << "Failed to read image in '" << sourceFile.c_str() << "' file";

	return 0;
}
