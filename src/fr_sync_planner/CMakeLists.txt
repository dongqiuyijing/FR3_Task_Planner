cmake_minimum_required(VERSION 3.22)
project(fr_sync_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(fairino_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(fr_sync_planner_trajectory
  src/winner_trajectory_io.cpp
  src/frozen_executor.cpp
)
target_include_directories(fr_sync_planner_trajectory PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(fr_sync_planner_trajectory yaml-cpp)
ament_target_dependencies(fr_sync_planner_trajectory
  rclcpp geometry_msgs moveit_core moveit_msgs geometric_shapes
)

add_executable(dual7_sync_task_executor src/dual7_real_task_executor.cpp)
target_include_directories(dual7_sync_task_executor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(dual7_sync_task_executor fr_sync_planner_trajectory yaml-cpp)
ament_target_dependencies(dual7_sync_task_executor
  rclcpp rclcpp_action sensor_msgs trajectory_msgs control_msgs
  moveit_msgs fairino_msgs std_srvs
)

install(TARGETS fr_sync_planner_trajectory dual7_sync_task_executor
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
