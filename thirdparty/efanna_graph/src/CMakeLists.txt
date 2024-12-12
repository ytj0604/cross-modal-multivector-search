set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(${PROJECT_NAME}_s PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

#install()