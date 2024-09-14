cmake_minimum_required (VERSION 3.10)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(nQuantGpp)
target_sources(nQuantGpp PRIVATE "nQuantGpp.cpp" "nQuantGpp.rc" "ApngWriter.cpp" "ApngWriter.h" "bitmapUtilities.cpp" "bitmapUtilities.h" "BlueNoise.cpp" "BlueNoise.h" "CIELABConvertor.cpp" "CIELABConvertor.h" "DivQuantizer.cpp" "DivQuantizer.h"
 "Dl3Quantizer.cpp" "Dl3Quantizer.h" "EdgeAwareSQuantizer.cpp" "EdgeAwareSQuantizer.h" "GilbertCurve.cpp" "GilbertCurve.h" "MedianCut.cpp" "MedianCut.h" "Otsu.cpp" "Otsu.h"
 "NeuQuantizer.cpp" "NeuQuantizer.h" "PnnLABQuantizer.cpp" "PnnLABQuantizer.h" "PnnLABGAQuantizer.cpp" "PnnLABGAQuantizer.h" "PnnQuantizer.cpp" "PnnQuantizer.h"
 "SpatialQuantizer.cpp" "SpatialQuantizer.h" "stdafx.h" "WuQuantizer.cpp" "WuQuantizer.h"
 "ga/Chromosome.h" "ga/NsgaIII.cpp" "ga/NsgaIII.h" "ga/APNsgaIII.cpp" "ga/APNsgaIII.h")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(nQuantGpp PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
else()
  target_link_libraries(nQuantGpp ${OpenCV_LIBS})
endif()
