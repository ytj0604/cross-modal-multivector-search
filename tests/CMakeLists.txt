set(CMAKE_CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME}
	gtest
	gtest_main
	# mockcpp
	pthread
	)

	# Locate GTest
find_package(GTest REQUIRED)
find_package(BLAS REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable

add_executable(test_build_roargraph test_build_roargraph.cpp)
target_link_libraries(test_build_roargraph ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)


add_executable(test_search_roargraph test_search_roargraph.cpp)
target_link_libraries(test_search_roargraph ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)

add_executable(test_search_multivector_roargraph test_search_multivector_roargraph.cpp)
target_link_libraries(test_search_multivector_roargraph ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)

add_executable(test_search_multivector_rerank test_search_multivector_rerank.cpp)
target_link_libraries(test_search_multivector_rerank ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options ${BLAS_LIBRARIES})

add_executable(test_search_multivector_rerank_temp test_search_multivector_rerank_temp.cpp)
target_link_libraries(test_search_multivector_rerank_temp ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options ${BLAS_LIBRARIES})

add_executable(test_search_multivector_fullscan_rerank test_search_multivector_fullscan_rerank.cpp)
target_link_libraries(test_search_multivector_fullscan_rerank ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options ${BLAS_LIBRARIES})

add_executable(test_KNN_significance test_KNN_significance.cpp)
target_link_libraries(test_KNN_significance ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options ${BLAS_LIBRARIES})

add_executable(test_rand_vector test_rand_vector.cpp)
target_link_libraries(test_rand_vector ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options ${BLAS_LIBRARIES})

add_executable(test_only_build_bipartite test_only_build_bipartite.cpp)
target_link_libraries(test_only_build_bipartite ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)

add_executable(test_search_bipartite test_search_bipartite.cpp)
target_link_libraries(test_search_bipartite ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)