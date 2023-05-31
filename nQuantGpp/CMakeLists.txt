cmake_minimum_required (VERSION 3.10)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable (nQuantGpp "nQuantGpp.cpp" "nQuantGpp.rc" "bitmapUtilities.cpp" "bitmapUtilities.h" "BlueNoise.cpp" "BlueNoise.h" "CIELABConvertor.cpp" "CIELABConvertor.h" "DivQuantizer.cpp" "DivQuantizer.h"
 "Dl3Quantizer.cpp" "Dl3Quantizer.h" "EdgeAwareSQuantizer.cpp" "EdgeAwareSQuantizer.h" "GilbertCurve.cpp" "GilbertCurve.h" "MedianCut.cpp" "MedianCut.h" "Otsu.cpp" "Otsu.h"
 "NeuQuantizer.cpp" "NeuQuantizer.h" "PnnLABQuantizer.cpp" "PnnLABQuantizer.h" "PnnQuantizer.cpp" "PnnQuantizer.h"
 "SpatialQuantizer.cpp" "SpatialQuantizer.h" "stdafx.h" "WuQuantizer.cpp" "WuQuantizer.h")
target_link_libraries(nQuantGpp ${OpenCV_LIBS})
