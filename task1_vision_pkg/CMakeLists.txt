cmake_minimum_required(VERSION 2.8.3)
project(task1_vision_pkg)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
      roscpp cv_bridge image_transport tf message_filters nav_msgs sensor_msgs jsk_mbzirc_msgs)
find_package(OpenCV REQUIRED)

add_service_files(
    FILES
  NonMaximumSuppression.srv
  Skeletonization.srv)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  jsk_mbzirc_msgs)

#add gpu support
set(ENABLE_CUDA ON)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


###################################
## catkin specific configuration ##
###################################
catkin_package(
    CATKIN_DEPENDS
)
set(PCL_INCLUDE_DIRS "/usr/include/pcl-1.7/")
set(EIGEN_INCLUDE_DIRS "/usr/include/eigen3/")

include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include(FindCUDA)
    set(CUDA_NVCC_FLAGS 
      --generate-code arch=compute_50,code=sm_50
      --generate-code arch=compute_52,code=sm_52
      )
    
    ## chens projection node
    cuda_add_executable(uav_img2pointcloud src/img2pointcloud/uav_img2pointcloud.cpp)
    target_link_libraries(uav_img2pointcloud ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
    
    ## gpu skeletonization node
    cuda_add_executable(gpu_skeletonization_node 
      src/skeletonization/gpu_skeletonization_node.cpp
      src/skeletonization/skeletonization_kernel.cu
      )
    target_link_libraries(gpu_skeletonization_node
      ${catkin_LIBRARIES}
      )

else()
  add_executable(uav_img2pointcloud src/img2pointcloud/uav_img2pointcloud.cpp)
  target_link_libraries(uav_img2pointcloud ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
endif(ENABLE_CUDA)


#build for landing site detection
  add_executable(uav_detect_landing_region_node
    src/uav_detect_landing_region_node.cpp
    src/histogram_of_oriented_gradients.cpp
    src/uav_detect_landing_region_trainer.cpp
    )
  target_link_libraries(uav_detect_landing_region_node ${catkin_LIBRARIES} ${OpenCV_LIBS} )

