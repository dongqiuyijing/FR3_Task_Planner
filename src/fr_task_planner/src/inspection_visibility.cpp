#include "fr_task_planner/inspection_visibility.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>

#include <geometric_shapes/bodies.h>
#include <geometric_shapes/body_operations.h>
#include <geometric_shapes/shapes.h>
#include <moveit/robot_model/link_model.h>

namespace fr_task_planner
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRayEps = 0.002;
constexpr double kSideMinVisible = 0.85;
constexpr double kDiskMinVisible = 0.90;

Eigen::Isometry3d modelFromWorld(const moveit::core::RobotState& state,
                                 const Eigen::Isometry3d& t_world_base)
{
  const Eigen::Isometry3d t_model_base = state.getGlobalLinkTransform("base_link");
  return t_model_base * t_world_base.inverse();
}

bool rayHitsBody(const bodies::Body& body, const Eigen::Vector3d& origin,
                 const Eigen::Vector3d& dir_unit, double max_dist)
{
  EigenSTL::vector_Vector3d hits;
  if (!body.intersectsRay(origin, dir_unit, &hits, 4))
  {
    return false;
  }
  for (const auto& hit : hits)
  {
    const double t = (hit - origin).dot(dir_unit);
    if (t > kRayEps && t < max_dist - kRayEps)
    {
      return true;
    }
  }
  return false;
}
}  // namespace

CameraModel loadCameraModelFromWorkspace()
{
  CameraModel model;
  const auto cam = fixedInspectionDesignCamera();
  model.frame = "world";
  model.has_extrinsic = true;
  model.has_optical_axis = true;
  model.has_intrinsics = false;
  model.has_resolution = false;
  model.has_fov = false;
  model.has_xy_line_constraint = true;
  model.line_x = cam.optical_center_world.x();
  model.line_y = cam.optical_center_world.y();
  model.t_world_camera = Eigen::Isometry3d::Identity();
  model.t_world_camera.translation() = cam.optical_center_world;
  model.t_world_camera.linear().col(2) = cam.optical_forward_world;
  model.t_world_camera.linear().col(1) = cam.image_up_world;
  model.t_world_camera.linear().col(0) =
      cam.optical_forward_world.cross(cam.image_up_world).normalized();
  model.optical_axis_world = cam.optical_forward_world;
  return model;
}

SurfaceRoiDef topCircleRoiDef()
{
  SurfaceRoiDef d;
  d.view = "top_circle";
  d.physical = "object +Z original upward top circular face, reserved for ARM2";
  d.bounded_roi_defined = true;
  d.radius_m = 0.0075;
  return d;
}

SurfaceRoiDef bottomCircleRoiDef()
{
  SurfaceRoiDef d;
  d.view = "bottom_circle";
  d.physical =
      "object -Z original table-contact bottom circular face, inspected by ARM1, "
      "z_local = -H/2";
  d.bounded_roi_defined = true;
  d.radius_m = 0.0075;
  return d;
}

bool isCircularCapView(const std::string& view_name)
{
  return view_name == "top_circle" || view_name == "bottom_circle";
}

SurfaceRoiDef sideViewRoiDef(const std::string& view_name, const std::string& local_normal)
{
  SurfaceRoiDef d;
  d.view = view_name;
  d.physical = std::string("camera-facing half-cylinder, local normal ") + local_normal +
               ", diameter 15 mm x height 35 mm";
  d.bounded_roi_defined = true;
  d.radius_m = 0.0075;
  return d;
}

std::vector<Eigen::Vector3d> sampleTopCircleDisk(const Eigen::Vector3d& center,
                                                 const Eigen::Vector3d& normal, double radius)
{
  Eigen::Vector3d n = normal.normalized();
  Eigen::Vector3d tmp = (std::abs(n.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  Eigen::Vector3d u = n.cross(tmp).normalized();
  Eigen::Vector3d v = n.cross(u).normalized();
  std::vector<Eigen::Vector3d> pts;
  pts.push_back(center);
  const double rings[] = {0.25, 0.50, 0.75, 0.95};
  const int n_ang[] = {8, 8, 12, 16};
  for (int r = 0; r < 4; ++r)
  {
    const double rad = rings[r] * radius;
    for (int k = 0; k < n_ang[r]; ++k)
    {
      const double ang = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(n_ang[r]);
      pts.push_back(center + rad * (std::cos(ang) * u + std::sin(ang) * v));
    }
  }
  return pts;
}

std::vector<Eigen::Vector3d> sampleCircularCapInObject(const Eigen::Isometry3d& t_world_object,
                                                       double z_local, double radius)
{
  std::vector<Eigen::Vector3d> pts;
  pts.push_back(t_world_object * Eigen::Vector3d(0.0, 0.0, z_local));
  const double rings[] = {0.25, 0.50, 0.75, 0.95};
  const int n_ang[] = {8, 8, 12, 16};
  for (int r = 0; r < 4; ++r)
  {
    const double rad = rings[r] * radius;
    for (int k = 0; k < n_ang[r]; ++k)
    {
      const double ang = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(n_ang[r]);
      const Eigen::Vector3d local(rad * std::cos(ang), rad * std::sin(ang), z_local);
      pts.push_back(t_world_object * local);
    }
  }
  return pts;
}

std::vector<Eigen::Vector3d> sampleSideHalfCylinder(const Eigen::Isometry3d& t_world_object,
                                                    const Eigen::Vector3d& local_normal,
                                                    double radius, double height)
{
  const Eigen::Vector3d n = local_normal.normalized();
  const int n_h = 5;
  const int n_az = 7;
  std::vector<Eigen::Vector3d> pts;
  for (int ih = 0; ih < n_h; ++ih)
  {
    const double z = -0.5 * height + height * static_cast<double>(ih) / static_cast<double>(n_h - 1);
    for (int ia = 0; ia < n_az; ++ia)
    {
      const double az =
          -0.5 * kPi + kPi * static_cast<double>(ia) / static_cast<double>(n_az - 1);
      Eigen::Vector3d radial = Eigen::AngleAxisd(az, Eigen::Vector3d::UnitZ()) * n;
      if (radial.head<2>().squaredNorm() < 1e-12)
      {
        radial = Eigen::Vector3d::UnitY();
      }
      radial.z() = 0.0;
      radial.normalize();
      const Eigen::Vector3d local(radius * radial.x(), radius * radial.y(), z);
      pts.push_back(t_world_object * local);
    }
  }
  return pts;
}

VisibilityAudit incompleteVisibility(const std::string& view, bool orientation_valid,
                                     int roi_samples)
{
  VisibilityAudit a;
  a.view = view;
  a.orientation_valid = orientation_valid;
  a.roi_samples = roi_samples;
  a.visible_samples = 0;
  a.camera_facing = "NOT AVAILABLE — CAMERA INTRINSICS REQUIRED FOR FOV";
  a.visible_fraction = "NOT AVAILABLE";
  a.fov_pass = "NOT AVAILABLE — CAMERA INTRINSICS REQUIRED";
  a.occluding_links = "see geometric occlusion checker";
  a.inspection_valid = "GEOMETRIC_ONLY — CAMERA_MODEL_INCOMPLETE FOR FOV";
  return a;
}

GeometricVisibilityResult checkGeometricVisibility(
    const planning_scene::PlanningScene& scene, const moveit::core::RobotState& state,
    const ViewGeom& view, const Eigen::Isometry3d& t_world_object, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& d1, const DesignCamera& camera, const Eigen::Isometry3d& t_world_base,
    double object_radius, double object_height)
{
  GeometricVisibilityResult out;
  out.view = view.name;
  const Eigen::Vector3d actual_center =
      t_world_object.translation() + t_world_object.linear() * view.center_in_object;
  const Eigen::Vector3d actual_normal =
      (t_world_object.linear() * view.normal_in_object).normalized();
  out.center_err = (actual_center - p1).norm();
  out.normal_dot = actual_normal.dot(d1.normalized());
  out.orientation_valid = out.normal_dot > 0.9986 && out.center_err < 0.005;
  if (isCircularCapView(view.name))
  {
    out.roi_world =
        sampleCircularCapInObject(t_world_object, view.center_in_object.z(), object_radius);
  }
  else
  {
    out.roi_world =
        sampleSideHalfCylinder(t_world_object, view.normal_in_object, object_radius, object_height);
  }
  out.roi_samples = static_cast<int>(out.roi_world.size());
  out.ray_visible.assign(out.roi_world.size(), true);

  const Eigen::Isometry3d t_model_world = modelFromWorld(state, t_world_base);
  const Eigen::Vector3d cam_model = t_model_world * camera.optical_center_world;
  std::set<std::string> hit_links;

  struct OccluderBody
  {
    std::string link;
    std::unique_ptr<bodies::Body> body;
  };
  std::vector<OccluderBody> occluders;
  const auto model = scene.getRobotModel();
  for (const auto& link_name : visibilityOccluderLinks())
  {
    if (!model->hasLinkModel(link_name) || !state.knowsFrameTransform(link_name))
    {
      continue;
    }
    const auto* link = model->getLinkModel(link_name);
    const auto& shapes = link->getShapes();
    const auto& origins = link->getCollisionOriginTransforms();
    const Eigen::Isometry3d t_model_link = state.getGlobalLinkTransform(link_name);
    for (size_t i = 0; i < shapes.size(); ++i)
    {
      if (!shapes[i])
      {
        continue;
      }
      std::unique_ptr<bodies::Body> body(bodies::createBodyFromShape(shapes[i].get()));
      if (!body)
      {
        continue;
      }
      const Eigen::Isometry3d pose = t_model_link * origins[i];
      body->setPose(pose);
      OccluderBody item;
      item.link = link_name;
      item.body = std::move(body);
      occluders.push_back(std::move(item));
    }
  }

  for (size_t i = 0; i < out.roi_world.size(); ++i)
  {
    const Eigen::Vector3d p_model = t_model_world * out.roi_world[i];
    Eigen::Vector3d delta = p_model - cam_model;
    const double dist = delta.norm();
    if (dist < 1e-6)
    {
      out.ray_visible[i] = false;
      continue;
    }
    const Eigen::Vector3d dir = delta / dist;
    bool blocked = false;
    for (const auto& occ : occluders)
    {
      if (rayHitsBody(*occ.body, cam_model, dir, dist))
      {
        blocked = true;
        hit_links.insert(occ.link);
      }
    }
    out.ray_visible[i] = !blocked;
    if (blocked)
    {
      ++out.blocked_samples;
    }
    else
    {
      ++out.visible_samples;
    }
  }

  out.occluders.assign(hit_links.begin(), hit_links.end());
  out.visible_fraction =
      out.roi_samples > 0 ? static_cast<double>(out.visible_samples) / out.roi_samples : 0.0;
  const bool center_ok = !out.ray_visible.empty() && out.ray_visible.front();
  out.center_ray_clear = center_ok;
  const double min_frac = isCircularCapView(view.name) ? kDiskMinVisible : kSideMinVisible;
  // Occlusion only. Pose geometry is already gated by the IK validator.
  out.geometric_face_visible = center_ok && out.visible_fraction >= min_frac;
  if (!center_ok)
  {
    out.fail_reason = "center_ray_blocked";
  }
  else if (out.visible_fraction < min_frac)
  {
    out.fail_reason = "roi_fraction_blocked";
  }
  out.classification = out.geometric_face_visible ? "GEOMETRIC_FACE_VISIBLE" : "VISIBILITY_BLOCKED";
  return out;
}

}  // namespace fr_task_planner
