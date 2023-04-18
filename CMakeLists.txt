cmake_minimum_required(VERSION 3.13)
project(convert  VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)


find_package( OpenCV REQUIRED )

# exe and lib
add_executable(convert convert.cpp)
include_directories(convert ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(convert ${OpenCV_LIBS} ${OpenCV_LIBRARIES})

