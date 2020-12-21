cmake_minimum_required(VERSION 3.10)
project(imagenet)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fexceptions")


include_directories(${CMAKE_SOURCE_DIR}/src/
                    ${CMAKE_SOURCE_DIR}/3third/mnn/include
  		    ${CMAKE_SOURCE_DIR}/3third/imagehelp)
		    							  
link_directories(${CMAKE_SOURCE_DIR}/3third/mnn/libs/linux)

add_executable(${PROJECT_NAME} src/imagemain.cpp src/imagenet.cpp)
target_link_libraries(${PROJECT_NAME} MNN)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME imagenet)
install(TARGETS ${PROJECT_NAME} DESTINATION ${LIBRARY_OUTPUT_PATH}) 


