cmake_minimum_required(VERSION 3.17)

# Locate required packages
find_package(GTest REQUIRED)
find_package(BLAS REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(${GTEST_INCLUDE_DIRS})

# Target libraries to link
set(TEST_LIBRARIES
    ${GTEST_LIBRARIES}
    pthread
    ${PROJECT_NAME}
    Boost::program_options
    ${BLAS_LIBRARIES}
    efanna2e_knn_lib 
    efanna2e_nsg_lib
)

# Automatically detect all .cpp files in the scripts directory
file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Create executables for each detected test file
foreach(source_file ${TEST_SOURCES})
    get_filename_component(test_name ${source_file} NAME_WE) # Extract filename without extension
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} ${TEST_LIBRARIES})
    target_include_directories(${test_name} PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/nsg/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/efanna_graph/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include)
endforeach()

set_target_properties(efanna2e_knn_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    )