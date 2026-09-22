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
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(fairino_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# DUAL-1: read-only dual-arm model test
add_executable(
  dual_model_smoke
  src/dual_model_smoke.cpp
)

ament_target_dependencies(
  dual_model_smoke
  rclcpp
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
)

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

add_executable(fr3_mtc_inspection_view_test src/mtc_fr3_inspection_view_test.cpp)
ament_target_dependencies(fr3_mtc_inspection_view_test
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

add_executable(fr3_mtc_view_transition_test src/mtc_fr3_view_transition_test.cpp)
ament_target_dependencies(fr3_mtc_view_transition_test
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

add_executable(fr3_mtc_full_order_test src/mtc_fr3_full_order_test.cpp)
ament_target_dependencies(fr3_mtc_full_order_test
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

add_library(fr_task_planner_endpoints
  src/inspection_endpoint_candidates.cpp
  src/inspection_visibility.cpp
  src/pregrasp_ik_candidates.cpp
)
target_include_directories(fr_task_planner_endpoints PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(fr_task_planner_endpoints
  rclcpp
  geometry_msgs
  moveit_core
  moveit_msgs
  geometric_shapes
)

add_executable(fr3_mtc_roll_candidate_test src/mtc_fr3_roll_candidate_test.cpp)
ament_target_dependencies(fr3_mtc_roll_candidate_test
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

add_executable(fr3_mtc_endpoint_branch_test src/mtc_fr3_endpoint_branch_test.cpp)
target_include_directories(fr3_mtc_endpoint_branch_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fr3_mtc_endpoint_branch_test fr_task_planner_endpoints)
ament_target_dependencies(fr3_mtc_endpoint_branch_test
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

add_executable(fr3_mtc_collision_diag src/mtc_fr3_collision_diag.cpp)
target_include_directories(fr3_mtc_collision_diag PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fr3_mtc_collision_diag fr_task_planner_endpoints)
ament_target_dependencies(fr3_mtc_collision_diag
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

add_executable(fr3_mtc_sim_visualizer src/mtc_fr3_sim_visualizer.cpp)
target_include_directories(fr3_mtc_sim_visualizer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fr3_mtc_sim_visualizer fr_task_planner_endpoints)
ament_target_dependencies(fr3_mtc_sim_visualizer
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  trajectory_msgs
  visualization_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)

add_executable(fr3_mtc_dense_top_circle_search src/mtc_fr3_dense_top_circle_search.cpp)
target_include_directories(fr3_mtc_dense_top_circle_search PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fr3_mtc_dense_top_circle_search fr_task_planner_endpoints)
ament_target_dependencies(fr3_mtc_dense_top_circle_search
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

add_executable(fr3_mtc_mount_layout_search src/mtc_fr3_mount_layout_search.cpp)
target_include_directories(fr3_mtc_mount_layout_search PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fr3_mtc_mount_layout_search fr_task_planner_endpoints)
ament_target_dependencies(fr3_mtc_mount_layout_search
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)

add_library(fr_task_planner_trajectory
  src/winner_trajectory_io.cpp
  src/frozen_executor.cpp
)
target_include_directories(fr_task_planner_trajectory PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(fr_task_planner_trajectory yaml-cpp)
ament_target_dependencies(fr_task_planner_trajectory
  rclcpp
  geometry_msgs
  moveit_core
  moveit_msgs
  geometric_shapes
)

add_executable(fr3_mtc_complete_abc_search src/mtc_fr3_complete_abc_search.cpp)
target_include_directories(fr3_mtc_complete_abc_search PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(fr3_mtc_complete_abc_search
  fr_task_planner_endpoints
  fr_task_planner_trajectory
  yaml-cpp
)
ament_target_dependencies(fr3_mtc_complete_abc_search
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)

add_executable(fr3_real_frozen_trajectory_executor
  src/fr3_real_frozen_trajectory_executor.cpp
)
target_include_directories(fr3_real_frozen_trajectory_executor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fr3_real_frozen_trajectory_executor
  fr_task_planner_trajectory
  yaml-cpp
)
ament_target_dependencies(fr3_real_frozen_trajectory_executor
  rclcpp
  rclcpp_action
  sensor_msgs
  trajectory_msgs
  control_msgs
  moveit_msgs
  fairino_msgs
)

add_executable(test_fr3_frozen_executor test/test_fr3_frozen_executor.cpp)
target_include_directories(test_fr3_frozen_executor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_fr3_frozen_executor
  fr_task_planner_trajectory
  yaml-cpp
)
ament_target_dependencies(test_fr3_frozen_executor
  rclcpp
  geometry_msgs
  moveit_core
  moveit_msgs
  geometric_shapes
)
target_compile_definitions(test_fr3_frozen_executor PRIVATE
  FR_TASK_PLANNER_TRAJ_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step12c_tilted_camera_winner_trajectory.yaml"
)

add_executable(test_step14_grasp_prefix test/test_step14_grasp_prefix.cpp)
target_include_directories(test_step14_grasp_prefix PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_step14_grasp_prefix
  fr_task_planner_endpoints
  fr_task_planner_trajectory
  yaml-cpp
)
ament_target_dependencies(test_step14_grasp_prefix
  rclcpp
  geometry_msgs
  moveit_core
  moveit_msgs
  geometric_shapes
)
target_compile_definitions(test_step14_grasp_prefix PRIVATE
  FR_TASK_PLANNER_TRAJ_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step12c_tilted_camera_winner_trajectory.yaml"
  FR_TASK_PLANNER_WINNER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step12c_tilted_camera_winner.yaml"
  FR_TASK_PLANNER_STEP14_TRAJ_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step14_optimized_grasp_trajectory.yaml"
  FR_TASK_PLANNER_STEP14_WINNER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step14_optimized_grasp_winner.yaml"
  FR_TASK_PLANNER_STEP14_DIAG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step14_grasp_optimization_diagnostics.yaml"
)

add_executable(test_step15_lift_to_a test/test_step15_lift_to_a.cpp)
target_include_directories(test_step15_lift_to_a PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_step15_lift_to_a
  fr_task_planner_endpoints
  fr_task_planner_trajectory
  yaml-cpp
)
ament_target_dependencies(test_step15_lift_to_a
  rclcpp
  geometry_msgs
  moveit_core
  moveit_msgs
  geometric_shapes
)
target_compile_definitions(test_step15_lift_to_a PRIVATE
  FR_TASK_PLANNER_STEP12C_TRAJ_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step12c_tilted_camera_winner_trajectory.yaml"
  FR_TASK_PLANNER_STEP12C_WINNER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step12c_tilted_camera_winner.yaml"
  FR_TASK_PLANNER_STEP14_TRAJ_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step14_optimized_grasp_trajectory.yaml"
  FR_TASK_PLANNER_STEP14_WINNER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step14_optimized_grasp_winner.yaml"
  FR_TASK_PLANNER_STEP14_DIAG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step14_grasp_optimization_diagnostics.yaml"
  FR_TASK_PLANNER_STEP15_TRAJ_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step15_optimized_lift_to_a_trajectory.yaml"
  FR_TASK_PLANNER_STEP15_WINNER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/step15_optimized_lift_to_a_winner.yaml"
)

add_executable(
  dual_arm_a_prefix_plan
  src/dual_arm_a_prefix_plan.cpp
)

target_link_libraries(
  dual_arm_a_prefix_plan
  yaml-cpp
)

ament_target_dependencies(
  dual_arm_a_prefix_plan
  rclcpp
  moveit_core
  moveit_ros_planning_interface
)

add_executable(
  dual_arm_b_plan
  src/dual_arm_b_plan.cpp
)

ament_target_dependencies(
  dual_arm_b_plan
  rclcpp
  moveit_core
  moveit_ros_planning_interface
)

# DUAL-4B-1: handover IK-only test. No execute.
add_executable(
  dual_handover_ik_test
  src/dual_handover_ik_test.cpp
)

target_link_libraries(
  dual_handover_ik_test
  yaml-cpp
)

ament_target_dependencies(
  dual_handover_ik_test
  rclcpp
  sensor_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-4B-2: handover static collision. Local PlanningScene only. No execute.
add_executable(
  dual_handover_static_collision_test
  src/dual_handover_static_collision_test.cpp
)

# DUAL-4B-2A: collision-source isolation. Local PlanningScene only. No execute.
add_executable(
  dual_collision_source_diagnostic
  src/dual_collision_source_diagnostic.cpp
)

target_link_libraries(
  dual_collision_source_diagnostic
  yaml-cpp
)

ament_target_dependencies(
  dual_collision_source_diagnostic
  rclcpp
  sensor_msgs
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  geometric_shapes
)

target_link_libraries(
  dual_handover_static_collision_test
  yaml-cpp
)

ament_target_dependencies(
  dual_handover_static_collision_test
  rclcpp
  sensor_msgs
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-4C: Arm B Cartesian straight-line approach. Local PlanningScene only. No execute.
add_executable(
  dual_handover_linear_approach_test
  src/dual_handover_linear_approach_test.cpp
)

target_link_libraries(
  dual_handover_linear_approach_test
  yaml-cpp
)

ament_target_dependencies(
  dual_handover_linear_approach_test
  rclcpp
  sensor_msgs
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-4D: gripper close/open sampling and local attachment transfer. No execute.
add_executable(
  dual_handover_transfer_test
  src/dual_handover_transfer_test.cpp
)

target_link_libraries(
  dual_handover_transfer_test
  yaml-cpp
)

ament_target_dependencies(
  dual_handover_transfer_test
  rclcpp
  sensor_msgs
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-4E: Arm A return Home while Arm B holds the part. PLAN ONLY.
add_executable(
  dual_arm_a_return_home_test
  src/dual_arm_a_return_home_test.cpp
)

target_link_libraries(
  dual_arm_a_return_home_test
  yaml-cpp
)

ament_target_dependencies(
  dual_arm_a_return_home_test
  rclcpp
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-6: Arm B remaining inspection sequence time-first optimization. PLAN ONLY.
add_executable(
  dual_arm_b_time_optimization_test
  src/dual_arm_b_time_optimization_test.cpp
)

target_link_libraries(
  dual_arm_b_time_optimization_test
  yaml-cpp
)

ament_target_dependencies(
  dual_arm_b_time_optimization_test
  rclcpp
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-6B: Arm B remaining inspection sequence joint-motion geometric optimization. PLAN ONLY.
add_executable(
  dual_arm_b_joint_motion_optimization_test
  src/dual_arm_b_joint_motion_optimization_test.cpp
)

target_link_libraries(
  dual_arm_b_joint_motion_optimization_test
  yaml-cpp
)

ament_target_dependencies(
  dual_arm_b_joint_motion_optimization_test
  rclcpp
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-5: Arm B remaining inspection sequence while holding small_part. PLAN ONLY.
add_executable(
  dual_arm_b_inspection_sequence_test
  src/dual_arm_b_inspection_sequence_test.cpp
)

target_link_libraries(
  dual_arm_b_inspection_sequence_test
  yaml-cpp
)

ament_target_dependencies(
  dual_arm_b_inspection_sequence_test
  rclcpp
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-7: complete dual-arm six-face model demo. PLAN ONLY.
add_executable(
  dual_complete_six_face_task_test
  src/dual_complete_six_face_task_test.cpp
)
target_link_libraries(
  dual_complete_six_face_task_test
  yaml-cpp
)
ament_target_dependencies(
  dual_complete_six_face_task_test
  rclcpp
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# KEYPOSE V1: multi-IK keypose candidates. No execute, no path planning.
add_executable(
  keypose_candidate_search
  src/keypose_candidate_search.cpp
)
target_link_libraries(
  keypose_candidate_search
  yaml-cpp
)
ament_target_dependencies(
  keypose_candidate_search
  rclcpp
  geometry_msgs
  sensor_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  geometric_shapes
)

# KEYPOSE V1: B_FACE4/5 root-cause diagnosis only. No candidate rewrite.
add_executable(
  keypose_b_face_diag
  src/keypose_b_face_diag.cpp
)
target_link_libraries(
  keypose_b_face_diag
  yaml-cpp
)
ament_target_dependencies(
  keypose_b_face_diag
  rclcpp
  geometry_msgs
  sensor_msgs
  visualization_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  geometric_shapes
)

# KEYPOSE V1: B_FACE4/5 correct-mirror IK only. No path, no other keyposes.
add_executable(
  keypose_b_face_mirror_search
  src/keypose_b_face_mirror_search.cpp
)
target_link_libraries(
  keypose_b_face_mirror_search
  yaml-cpp
)
ament_target_dependencies(
  keypose_b_face_mirror_search
  rclcpp
  geometry_msgs
  sensor_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  geometric_shapes
)

# KEYPOSE V1: B_FACE4 only, A_FACE1 YZ-mirror + grasp-offset TCP.
add_executable(
  keypose_b_face4_solve
  src/keypose_b_face4_solve.cpp
)
target_link_libraries(
  keypose_b_face4_solve
  yaml-cpp
)
add_executable(
  keypose_v1_connect
  src/keypose_v1_connect.cpp
)
target_link_libraries(
  keypose_v1_connect
  yaml-cpp
)
ament_target_dependencies(
  keypose_v1_connect
  rclcpp
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)
add_executable(
  keypose_v1_sequence_check
  src/keypose_v1_sequence_check.cpp
)
target_include_directories(
  keypose_v1_sequence_check
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(
  keypose_v1_sequence_check
  yaml-cpp
)
add_executable(
  keypose_v1_b_home
  src/keypose_v1_b_home.cpp
)
target_link_libraries(
  keypose_v1_b_home
  yaml-cpp
)
ament_target_dependencies(
  keypose_v1_b_home
  rclcpp
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  geometric_shapes
)
ament_target_dependencies(
  keypose_b_face4_solve
  rclcpp
  geometry_msgs
  sensor_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  geometric_shapes
)

# DUAL-8-HOME: live Current→Home planner. PLAN ONLY.
add_executable(
  dual8_current_to_home_test
  src/dual8_current_to_home_test.cpp
)
target_link_libraries(
  dual8_current_to_home_test
  yaml-cpp
)
ament_target_dependencies(
  dual8_current_to_home_test
  rclcpp
  sensor_msgs
  trajectory_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
)

# DUAL-7: dual-arm real executor. Reuses STEP13 frozen-executor helpers.
# Default DRY RUN. Does not rewrite fr3_real_frozen_trajectory_executor.
add_executable(dual7_real_task_executor
  src/dual7_real_task_executor.cpp
)
target_include_directories(dual7_real_task_executor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(dual7_real_task_executor
  fr_task_planner_trajectory
  yaml-cpp
)
ament_target_dependencies(dual7_real_task_executor
  rclcpp
  rclcpp_action
  sensor_msgs
  trajectory_msgs
  control_msgs
  moveit_msgs
  fairino_msgs
  std_srvs
)

install(TARGETS
  fr3_mtc_smoke_test
  fr3_mtc_pregrasp_test
  fr3_mtc_pilz_smoke_test
  fr3_mtc_grasp_chain_test
  fr3_mtc_attach_lift_test
  fr3_mtc_inspection_view_test
  fr3_mtc_view_transition_test
  fr3_mtc_full_order_test
  fr3_mtc_roll_candidate_test
  fr3_mtc_endpoint_branch_test
  fr3_mtc_collision_diag
  fr3_mtc_sim_visualizer
  fr3_mtc_dense_top_circle_search
  fr3_mtc_mount_layout_search
  fr3_mtc_complete_abc_search
  fr3_real_frozen_trajectory_executor
  test_fr3_frozen_executor
  test_step14_grasp_prefix
  test_step15_lift_to_a
  dual_model_smoke
  dual_arm_a_prefix_plan
  dual_arm_b_plan
  dual_handover_ik_test
  dual_handover_static_collision_test
  dual_collision_source_diagnostic
  dual_handover_linear_approach_test
  dual_handover_transfer_test
  dual_arm_a_return_home_test
  dual_arm_b_inspection_sequence_test
  dual_arm_b_time_optimization_test
  dual_arm_b_joint_motion_optimization_test
  dual_complete_six_face_task_test
  dual7_real_task_executor
  dual8_current_to_home_test
  keypose_candidate_search
  keypose_b_face_diag
  keypose_b_face_mirror_search
  keypose_b_face4_solve
  keypose_v1_connect
  keypose_v1_b_home
  keypose_v1_sequence_check
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/step15_sim_playback.py
  scripts/dual_5t_rviz_preview.py
  scripts/dual_complete_six_face_rviz_preview.py
  scripts/keypose_candidate_rviz_preview.py
  scripts/keypose_b_face_diag_preview.py
  scripts/keypose_b_face_mirror_preview.py
  scripts/keypose_b_face4_preview.py
  scripts/keypose_v1_tour_preview.py
  scripts/keypose_v1_a_return_preview.py
  scripts/keypose_v1_b_home_preview.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
