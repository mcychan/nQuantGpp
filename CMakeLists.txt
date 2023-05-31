cmake_minimum_required (VERSION 3.10)

project ("nQuantGpp")
add_definitions(-DUNICODE -D_UNICODE)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE
        "MinSizeRel"
        CACHE STRING "Build type: Debug, Release, RelWithDebInfo or MinSizeRel"
              FORCE)

add_subdirectory ("nQuantGpp")
