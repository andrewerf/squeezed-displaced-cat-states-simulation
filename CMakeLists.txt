cmake_minimum_required(VERSION 3.27)
project(CatStatesGaussianTransformation)

set(CMAKE_CXX_STANDARD 23)

find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(main main.cpp)
target_link_libraries(main TBB::tbb Boost::program_options)
target_compile_options(main PRIVATE -Ofast)