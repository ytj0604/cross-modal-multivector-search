cmake_minimum_required(VERSION 3.17)

project(efanna2e_nsg)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")


#Create library target for the project
file(GLOB_RECURSE LIB_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(efanna2e_nsg_lib ${LIB_SOURCES})

# Include the headers for library target
target_include_directories(efanna2e_nsg_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set C++ Standard and Position Independent Code
set_target_properties(efanna2e_nsg_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON # Make the library position independent
    )
add_definitions (-std=c++17 -O3 -lboost -march=native -Wall -DINFO)