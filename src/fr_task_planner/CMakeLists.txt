cmake_minimum_required(VERSION 3.22)
project(fr_task_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_task_constructor_msgs REQUIRED)

add_executable(fr3_mtc_smoke_test src/mtc_fr3_smoke_test.cpp)
ament_target_dependencies(fr3_mtc_smoke_test
  rclcpp
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
)

add_executable(fr3_mtc_pregrasp_test src/mtc_fr3_pregrasp_test.cpp)
ament_target_dependencies(fr3_mtc_pregrasp_test
  rclcpp
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)

add_executable(fr3_mtc_pilz_smoke_test src/mtc_fr3_pilz_smoke_test.cpp)
ament_target_dependencies(fr3_mtc_pilz_smoke_test
  rclcpp
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)

add_executable(fr3_mtc_grasp_chain_test src/mtc_fr3_grasp_chain_test.cpp)
ament_target_dependencies(fr3_mtc_grasp_chain_test
  rclcpp
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)

add_executable(fr3_mtc_attach_lift_test src/mtc_fr3_attach_lift_test.cpp)
ament_target_dependencies(fr3_mtc_attach_lift_test
  rclcpp
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)

install(TARGETS
  fr3_mtc_smoke_test
  fr3_mtc_pregrasp_test
  fr3_mtc_pilz_smoke_test
  fr3_mtc_grasp_chain_test
  fr3_mtc_attach_lift_test
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
