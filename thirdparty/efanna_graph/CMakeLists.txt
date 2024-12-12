cmake_minimum_required(VERSION 3.17)

project(efanna2e_knn)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

#OpenBLAS
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)
find_package(OpenBLAS REQUIRED)
include_directories(${OpenBLAS_INCLUDE_DIR})
set(BLAS_LIB ${OpenBLAS_LIB})

#external libraries like faiss
include_directories(${PROJECT_SOURCE_DIR}/extern_libraries)
link_directories(${PROJECT_SOURCE_DIR}/extern_libraries/faiss/lib)

#add library for current project
file(GLOB_RECURSE LIB_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(efanna2e_knn_lib ${LIB_SOURCES})

# Include the headers so the target can be used.
target_include_directories(efanna2e_knn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(efanna2e_knn_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON # Make the library position independent
    )
add_definitions (-std=c++17 -O3 -lboost -march=native -Wall -DINFO)