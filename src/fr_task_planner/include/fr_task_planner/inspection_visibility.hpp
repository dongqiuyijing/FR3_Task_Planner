#pragma once

#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>

namespace fr_task_planner
{

// Links that remain opaque to the camera even when ACM allows contact.
inline const std::vector<std::string>& visibilityOccluderLinks()
{
  static const std::vector<std::string> kLinks = {
      "gripper_base_link", "rail_155",     "slider_l",      "slider_r",
      "finger_l",          "finger_r",     "gripper_gap_link", "wrist1_link",
      "wrist2_link",       "wrist3_link",  "forearm_link",  "upperarm_link",
      "shoulder_link"};
  return kLinks;
}

struct CameraModel
{
  std::string frame;
  bool has_extrinsic = false;
  bool has_optical_axis = false;
  bool has_intrinsics = false;
  bool has_resolution = false;
  bool has_fov = false;
  bool has_xy_line_constraint = false;
  double line_x = 0.0;
  double line_y = 0.0;
  Eigen::Isometry3d t_world_camera = Eigen::Isometry3d::Identity();
  Eigen::Vector3d optical_axis_world = Eigen::Vector3d::Zero();
};

inline std::string cameraModelClass(const CameraModel& model)
{
  if (model.has_extrinsic && model.has_optical_axis && model.has_intrinsics &&
      model.has_resolution)
  {
    return "CAMERA_MODEL_COMPLETE";
  }
  return "CAMERA_MODEL_INCOMPLETE";
}

struct DesignCamera
{
  // STEP12C tilted D405 design pose. Optical +Z = forward.
  Eigen::Vector3d optical_center_world = Eigen::Vector3d(0.0, 0.0, 1.40);
  Eigen::Vector3d optical_forward_world = Eigen::Vector3d(0.0, 0.70710678, -0.70710678);
  Eigen::Vector3d image_up_world = Eigen::Vector3d(0.0, 0.70710678, 0.70710678);
};

inline DesignCamera fixedInspectionDesignCamera()
{
  return DesignCamera{};
}

struct SurfaceRoiDef
{
  std::string view;
  std::string physical;
  bool bounded_roi_defined = false;
  std::string gap;
  double radius_m = 0.0;
};

struct VisibilityAudit
{
  std::string view;
  bool orientation_valid = false;
  std::string camera_facing = "NOT AVAILABLE — CAMERA EXTRINSIC REQUIRED";
  int roi_samples = 0;
  int visible_samples = 0;
  std::string visible_fraction = "NOT AVAILABLE";
  std::string fov_pass = "NOT AVAILABLE — CAMERA INTRINSICS REQUIRED";
  std::string occluding_links = "NOT AVAILABLE — CAMERA EXTRINSIC REQUIRED";
  std::string inspection_valid = "FALSE — CAMERA_MODEL_INCOMPLETE";
};

struct GeometricVisibilityResult
{
  std::string view;
  bool orientation_valid = false;
  bool geometric_face_visible = false;
  int roi_samples = 0;
  int visible_samples = 0;
  int blocked_samples = 0;
  double visible_fraction = 0.0;
  double center_err = 0.0;
  double normal_dot = 0.0;
  bool center_ray_clear = false;
  std::vector<std::string> occluders;
  std::string classification = "VISIBILITY_BLOCKED";
  std::string fail_reason = "ok";
  std::vector<Eigen::Vector3d> roi_world;
  std::vector<bool> ray_visible;
};

CameraModel loadCameraModelFromWorkspace();
SurfaceRoiDef topCircleRoiDef();
SurfaceRoiDef bottomCircleRoiDef();
SurfaceRoiDef sideViewRoiDef(const std::string& view_name, const std::string& local_normal);
bool isCircularCapView(const std::string& view_name);
std::vector<Eigen::Vector3d> sampleTopCircleDisk(const Eigen::Vector3d& center,
                                                 const Eigen::Vector3d& normal, double radius);
std::vector<Eigen::Vector3d> sampleCircularCapInObject(const Eigen::Isometry3d& t_world_object,
                                                       double z_local, double radius);
std::vector<Eigen::Vector3d> sampleSideHalfCylinder(const Eigen::Isometry3d& t_world_object,
                                                    const Eigen::Vector3d& local_normal,
                                                    double radius, double height);
VisibilityAudit incompleteVisibility(const std::string& view, bool orientation_valid,
                                     int roi_samples);

GeometricVisibilityResult checkGeometricVisibility(
    const planning_scene::PlanningScene& scene, const moveit::core::RobotState& state,
    const ViewGeom& view, const Eigen::Isometry3d& t_world_object, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& d1, const DesignCamera& camera, const Eigen::Isometry3d& t_world_base,
    double object_radius, double object_height);

}  // namespace fr_task_planner
