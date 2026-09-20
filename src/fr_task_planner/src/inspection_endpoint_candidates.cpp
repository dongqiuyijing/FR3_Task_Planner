#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/robot_state/attached_body.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>

namespace fr_task_planner
{
namespace
{
struct ObjectPairContact
{
  std::string other;
  double depth = 0.0;
};

bool isTouchLink(const std::string& link, const std::vector<std::string>& touch_links)
{
  return std::find(touch_links.begin(), touch_links.end(), link) != touch_links.end();
}

std::vector<ObjectPairContact> rawObjectContacts(const planning_scene::PlanningScene& scene,
                                                 const std::string& object_id,
                                                 const std::string& table_name,
                                                 const std::vector<std::string>& touch_links)
{
  const auto diag = scene.diff();
  auto& acm = diag->getAllowedCollisionMatrixNonConst();
  if (!touch_links.empty())
  {
    acm.setEntry(object_id, touch_links, false);
  }
  acm.setEntry(object_id, table_name, true);
  collision_detection::CollisionRequest req;
  req.contacts = true;
  req.max_contacts = 50;
  req.max_contacts_per_pair = 3;
  collision_detection::CollisionResult res;
  diag->checkCollision(req, res);

  std::vector<ObjectPairContact> out;
  for (const auto& item : res.contacts)
  {
    const std::string& a = item.first.first;
    const std::string& b = item.first.second;
    if (a != object_id && b != object_id)
    {
      continue;
    }
    const std::string other = (a == object_id) ? b : a;
    if (other == table_name)
    {
      continue;
    }
    ObjectPairContact contact;
    contact.other = other;
    if (!item.second.empty())
    {
      contact.depth = item.second.front().depth;
    }
    out.push_back(contact);
  }
  return out;
}

void classifyObjectContacts(const std::vector<ObjectPairContact>& contacts,
                            const std::vector<std::string>& touch_links,
                            std::vector<ObjectPairContact>& allowed,
                            std::vector<ObjectPairContact>& illegal)
{
  allowed.clear();
  illegal.clear();
  for (const auto& contact : contacts)
  {
    if (isTouchLink(contact.other, touch_links))
    {
      allowed.push_back(contact);
    }
    else
    {
      illegal.push_back(contact);
    }
  }
}

void validateOne(EndpointCandidate& cand, const planning_scene::PlanningScene& lift_scene,
                 const EndpointGenerationConfig& cfg, const ViewGeom& view,
                 const std::map<std::string, double>& q_lift)
{
  const auto robot_model = lift_scene.getRobotModel();
  auto* jmg = robot_model->getJointModelGroup(cfg.group);
  moveit::core::RobotState fk(robot_model);
  fk.setToDefaultValues();
  applyJoints(fk, cand.joints);
  cand.bounds_ok = static_cast<bool>(jmg) && fk.satisfiesBounds(jmg);
  const Eigen::Isometry3d actual_tcp_base = tcpInBase(fk, cfg.ee_link);
  const Eigen::Isometry3d actual_tcp_world = cfg.t_model_base * actual_tcp_base;
  const Eigen::Isometry3d actual_object_world = actual_tcp_world * cfg.t_tcp_object;
  const Eigen::Vector3d actual_center =
      actual_object_world.translation() + actual_object_world.linear() * view.center_in_object;
  const Eigen::Vector3d actual_normal =
      (actual_object_world.linear() * view.normal_in_object).normalized();
  const Eigen::Vector3d actual_up = (actual_object_world.linear() * view.up_in_object).normalized();
  cand.view_center_error = (actual_center - cfg.p1).norm();
  cand.normal_error =
      std::acos(std::min(1.0, std::max(-1.0, actual_normal.dot(cfg.d1.normalized())))) * 180.0 /
      M_PI;
  cand.up_error_deg =
      std::acos(std::min(1.0, std::max(-1.0, actual_up.dot(cfg.preferred_up.normalized())))) *
      180.0 / M_PI;
  double tcp_pos = 0.0;
  double tcp_ori = 0.0;
  poseError(poseToIso(cand.tcp_target.pose), actual_tcp_base, tcp_pos, tcp_ori);
  cand.tcp_object_error = tcp_pos;
  cand.joint_distance_from_lift = jointL2(cand.joints, q_lift);

  const auto diag = lift_scene.diff();
  applyJoints(diag->getCurrentStateNonConst(), cand.joints);
  cand.attached_object_present = diag->getCurrentState().hasAttachedBody(cfg.object_id);
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 40;
  diag->checkCollision(req, res);
  std::vector<ObjectPairContact> allowed;
  std::vector<ObjectPairContact> illegal;
  classifyObjectContacts(rawObjectContacts(*diag, cfg.object_id, cfg.table_name, cfg.touch_links),
                         cfg.touch_links, allowed, illegal);
  cand.collision_free = !res.collision && illegal.empty();

  if (!cand.bounds_ok)
  {
    cand.failure_reason = "joint bounds";
  }
  else if (!cand.attached_object_present)
  {
    cand.failure_reason = "attached object missing";
  }
  else if (!cand.collision_free)
  {
    cand.failure_reason = res.collision ? "collision" : "illegal attached contact";
  }
  else if (cand.view_center_error > cfg.pos_tol || cand.normal_error > cfg.ori_tol_deg)
  {
    cand.failure_reason = "fk geometry";
  }
  else
  {
    cand.valid = true;
    cand.failure_reason = "ok";
  }
  cand.candidate_id = makeCandidateId(cand.view_name, cand.roll_deg, cand.ik_index);
  cand.digest = makeCandidateDigest(cand.view_name, cand.roll_deg, cand.joints);
}
}  // namespace

Eigen::Isometry3d poseToIso(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  transform.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y,
                                          pose.orientation.z)
                           .normalized()
                           .toRotationMatrix();
  return transform;
}

geometry_msgs::msg::Pose isoToPose(const Eigen::Isometry3d& transform)
{
  geometry_msgs::msg::Pose pose;
  const Eigen::Vector3d p = transform.translation();
  const Eigen::Quaterniond q(transform.rotation());
  pose.position.x = p.x();
  pose.position.y = p.y();
  pose.position.z = p.z();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

void poseError(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double& position_m,
               double& orientation_deg)
{
  position_m = (a.translation() - b.translation()).norm();
  const Eigen::Quaterniond qa(a.rotation());
  const Eigen::Quaterniond qb(b.rotation());
  double qdot = std::min(1.0, std::abs(qa.normalized().dot(qb.normalized())));
  orientation_deg = 2.0 * std::acos(qdot) * 180.0 / M_PI;
}

double jointL2(const std::map<std::string, double>& a, const std::map<std::string, double>& b)
{
  double acc = 0.0;
  for (const auto& name : kArmJoints)
  {
    const double dq = a.at(name) - b.at(name);
    acc += dq * dq;
  }
  return std::sqrt(acc);
}

double jointL1(const std::map<std::string, double>& a, const std::map<std::string, double>& b)
{
  double acc = 0.0;
  for (const auto& name : kArmJoints)
  {
    acc += std::abs(a.at(name) - b.at(name));
  }
  return acc;
}

double maxJointError(const std::map<std::string, double>& a, const std::map<std::string, double>& b)
{
  double max_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_error = std::max(max_error, std::abs(a.at(name) - b.at(name)));
  }
  return max_error;
}

std::map<std::string, double> jointsFromState(const moveit::core::RobotState& state)
{
  std::map<std::string, double> joints;
  for (const auto& name : kArmJoints)
  {
    joints[name] = state.getVariablePosition(name);
  }
  return joints;
}

void applyJoints(moveit::core::RobotState& state, const std::map<std::string, double>& joints)
{
  for (const auto& item : joints)
  {
    if (state.getRobotModel()->hasJointModel(item.first))
    {
      state.setVariablePosition(item.first, item.second);
    }
  }
  state.update();
}

Eigen::Isometry3d tcpInBase(const moveit::core::RobotState& state, const std::string& ee_link)
{
  return state.getGlobalLinkTransform("base_link").inverse() * state.getGlobalLinkTransform(ee_link);
}

std::string formatRollDeg(double roll_deg)
{
  if (std::abs(roll_deg) < 1e-9)
  {
    return "0";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(0) << roll_deg;
  return oss.str();
}

std::string makeCandidateId(const std::string& view, double roll_deg, int ik_index)
{
  return view + "_r" + formatRollDeg(roll_deg) + "_ik" + std::to_string(ik_index);
}

std::string makeCandidateDigest(const std::string& view, double roll_deg,
                                const std::map<std::string, double>& joints)
{
  std::ostringstream key;
  key << std::fixed << std::setprecision(6) << view << '|' << roll_deg;
  for (const auto& name : kArmJoints)
  {
    key << '|' << joints.at(name);
  }
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char byte : key.str())
  {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 1099511628211ull;
  }
  std::ostringstream hex;
  hex << std::hex << std::setw(16) << std::setfill('0') << hash;
  return hex.str();
}

std::string makeStageName(const EndpointCandidate& cand)
{
  return "endpoint_" + cand.candidate_id;
}

void sortEndpointCandidates(std::vector<EndpointCandidate>& cands)
{
  std::sort(cands.begin(), cands.end(), [](const EndpointCandidate& a, const EndpointCandidate& b) {
    if (a.pose_index != b.pose_index)
    {
      return a.pose_index < b.pose_index;
    }
    return a.ik_index < b.ik_index;
  });
}

std::vector<RollPose> filterRollsByView(const std::vector<RollPose>& rolls, const std::string& view)
{
  std::vector<RollPose> out;
  for (const auto& roll : rolls)
  {
    if (roll.view == view)
    {
      out.push_back(roll);
    }
  }
  return out;
}

std::vector<EndpointCandidate> capCandidates(const std::vector<EndpointCandidate>& valid,
                                             int max_endpoint_candidates)
{
  if (max_endpoint_candidates <= 0 ||
      static_cast<int>(valid.size()) <= max_endpoint_candidates)
  {
    return valid;
  }
  return std::vector<EndpointCandidate>(valid.begin(),
                                        valid.begin() + max_endpoint_candidates);
}

EndpointGenerationResult generateValidEndpointCandidates(
    const planning_scene::PlanningScene& lift_scene, const std::vector<RollPose>& rolls,
    const ViewGeom& view, const EndpointGenerationConfig& cfg, const rclcpp::Logger& logger)
{
  EndpointGenerationResult result;
  const auto robot_model = lift_scene.getRobotModel();
  auto* jmg = robot_model->getJointModelGroup(cfg.group);
  if (!jmg)
  {
    return result;
  }
  const auto q_lift = jointsFromState(lift_scene.getCurrentState());
  moveit::core::RobotState seed(lift_scene.getCurrentState());
  for (const auto& roll : rolls)
  {
    if (roll.view != view.name)
    {
      continue;
    }
    const Eigen::Isometry3d t_model_tcp = cfg.t_model_base * poseToIso(roll.tcp.pose);
    std::vector<std::map<std::string, double>> found;
    const uint32_t attempts = std::max<uint32_t>(cfg.max_ik_solutions_per_pose * 4, 8);
    for (uint32_t attempt = 0; attempt < attempts && found.size() < cfg.max_ik_solutions_per_pose;
         ++attempt)
    {
      moveit::core::RobotState ik_state(seed);
      if (attempt > 0)
      {
        ik_state.setToRandomPositionsNearBy(jmg, seed, 2.0);
      }
      const double timeout = (attempt == 0) ? 0.25 : 0.05;
      if (!ik_state.setFromIK(jmg, t_model_tcp, cfg.ee_link, timeout))
      {
        if (attempt == 0 && std::abs(roll.roll_deg) < 1e-9)
        {
          RCLCPP_WARN(logger,
                      "canonical setFromIK miss %s model_tcp xyz=(%.4f, %.4f, %.4f)",
                      roll.view.c_str(), t_model_tcp.translation().x(),
                      t_model_tcp.translation().y(), t_model_tcp.translation().z());
        }
        continue;
      }
      const auto q = jointsFromState(ik_state);
      bool duplicate = false;
      for (const auto& prev : found)
      {
        if (jointL2(prev, q) < cfg.min_ik_solution_distance)
        {
          duplicate = true;
          break;
        }
      }
      if (duplicate)
      {
        continue;
      }
      found.push_back(q);
      EndpointCandidate cand;
      cand.view_name = roll.view;
      cand.roll_deg = roll.roll_deg;
      cand.pose_index = roll.pose_index;
      cand.ik_index = static_cast<int>(found.size() - 1);
      cand.object_target = roll.object;
      cand.tcp_target = roll.tcp;
      cand.joints = q;
      validateOne(cand, lift_scene, cfg, view, q_lift);
      result.raw.push_back(cand);
    }
  }

  std::vector<EndpointCandidate> unique;
  for (const auto& cand : result.raw)
  {
    if (!cand.valid)
    {
      continue;
    }
    bool dup = false;
    for (const auto& prev : unique)
    {
      if (jointL2(prev.joints, cand.joints) < cfg.min_ik_solution_distance)
      {
        dup = true;
        break;
      }
    }
    if (!dup)
    {
      unique.push_back(cand);
    }
  }
  sortEndpointCandidates(unique);
  for (size_t i = 0; i < unique.size(); ++i)
  {
    unique[i].ik_index = unique[i].ik_index;
    unique[i].candidate_id =
        makeCandidateId(unique[i].view_name, unique[i].roll_deg, unique[i].ik_index);
  }
  result.valid = unique;
  return result;
}

std::string collisionCategoryName(CollisionCategory category)
{
  switch (category)
  {
    case CollisionCategory::ROBOT_SELF:
      return "ROBOT_SELF";
    case CollisionCategory::ROBOT_TABLE:
      return "ROBOT_TABLE";
    case CollisionCategory::ROBOT_COLUMN:
      return "ROBOT_COLUMN";
    case CollisionCategory::PART_TABLE:
      return "PART_TABLE";
    case CollisionCategory::PART_COLUMN:
      return "PART_COLUMN";
    case CollisionCategory::PART_NON_TOUCH_ROBOT:
      return "PART_NON_TOUCH_ROBOT";
    case CollisionCategory::PART_TOUCH_ROBOT:
      return "PART_TOUCH_ROBOT";
    case CollisionCategory::OTHER:
    default:
      return "OTHER";
  }
}

std::string normalizePairKey(const std::string& a, const std::string& b)
{
  return (a <= b) ? (a + " <-> " + b) : (b + " <-> " + a);
}

CollisionCategory classifyContactPair(const std::string& a, const std::string& b,
                                      const CollisionDiagConfig& cfg,
                                      const moveit::core::RobotModel& model)
{
  const bool a_obj = (a == cfg.object_id);
  const bool b_obj = (b == cfg.object_id);
  const bool a_table = (a == cfg.table_name);
  const bool b_table = (b == cfg.table_name);
  const bool a_col = (a == cfg.column_name);
  const bool b_col = (b == cfg.column_name);
  const bool a_robot = model.hasLinkModel(a);
  const bool b_robot = model.hasLinkModel(b);
  if (a_obj || b_obj)
  {
    const std::string other = a_obj ? b : a;
    if (other == cfg.table_name)
    {
      return CollisionCategory::PART_TABLE;
    }
    if (other == cfg.column_name)
    {
      return CollisionCategory::PART_COLUMN;
    }
    if (model.hasLinkModel(other))
    {
      if (isTouchLink(other, cfg.touch_links))
      {
        return CollisionCategory::PART_TOUCH_ROBOT;
      }
      return CollisionCategory::PART_NON_TOUCH_ROBOT;
    }
    return CollisionCategory::OTHER;
  }
  if ((a_table && b_robot) || (b_table && a_robot))
  {
    return CollisionCategory::ROBOT_TABLE;
  }
  if ((a_col && b_robot) || (b_col && a_robot))
  {
    return CollisionCategory::ROBOT_COLUMN;
  }
  if (a_robot && b_robot)
  {
    return CollisionCategory::ROBOT_SELF;
  }
  return CollisionCategory::OTHER;
}

bool acmEntryAllowed(const planning_scene::PlanningScene& scene, const std::string& a,
                     const std::string& b)
{
  collision_detection::AllowedCollision::Type type;
  return scene.getAllowedCollisionMatrix().getEntry(a, b, type) &&
         type == collision_detection::AllowedCollision::ALWAYS;
}

namespace
{
void fillSnapshot(CollisionSnapshot& snap, const collision_detection::CollisionResult& res,
                  const CollisionDiagConfig& cfg, const moveit::core::RobotModel& model)
{
  snap.collision = res.collision;
  snap.contact_count = 0;
  snap.depth_available = false;
  snap.contacts.clear();
  for (const auto& item : res.contacts)
  {
    ContactRecord rec;
    rec.a = item.first.first;
    rec.b = item.first.second;
    rec.pair_key = normalizePairKey(rec.a, rec.b);
    rec.category = classifyContactPair(rec.a, rec.b, cfg, model);
    rec.contact_count = static_cast<int>(item.second.size());
    snap.contact_count += rec.contact_count;
    rec.depth_available = false;
    rec.depth = 0.0;
    if (!item.second.empty())
    {
      const double depth = item.second.front().depth;
      if (std::isfinite(depth) && std::abs(depth) > std::numeric_limits<double>::epsilon())
      {
        rec.depth_available = true;
        rec.depth = depth;
        snap.depth_available = true;
      }
    }
    snap.contacts.push_back(rec);
  }
  std::sort(snap.contacts.begin(), snap.contacts.end(),
            [](const ContactRecord& lhs, const ContactRecord& rhs) {
              return lhs.pair_key < rhs.pair_key;
            });
}

CollisionSnapshot checkSceneContacts(const planning_scene::PlanningScene& scene,
                                     const CollisionDiagConfig& cfg, bool self_only)
{
  collision_detection::CollisionRequest req;
  req.contacts = true;
  req.max_contacts = 200;
  req.max_contacts_per_pair = 8;
  collision_detection::CollisionResult res;
  if (self_only)
  {
    scene.checkSelfCollision(req, res);
  }
  else
  {
    scene.checkCollision(req, res);
  }
  CollisionSnapshot snap;
  fillSnapshot(snap, res, cfg, *scene.getRobotModel());
  return snap;
}
}  // namespace

CollisionSnapshot collectCollisionContacts(const planning_scene::PlanningScene& scene,
                                           const CollisionDiagConfig& cfg)
{
  return checkSceneContacts(scene, cfg, false);
}

planning_scene::PlanningScenePtr cloneDiagnosticScene(const planning_scene::PlanningScene& src)
{
  return planning_scene::PlanningScene::clone(src.diff());
}

void applyJointsToScene(planning_scene::PlanningScene& scene,
                        const std::map<std::string, double>& joints)
{
  applyJoints(scene.getCurrentStateNonConst(), joints);
}

void detachObjectDiagnostic(planning_scene::PlanningScene& scene, const std::string& object_id)
{
  if (!scene.getCurrentState().hasAttachedBody(object_id))
  {
    return;
  }
  moveit_msgs::msg::AttachedCollisionObject msg;
  msg.object.id = object_id;
  msg.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processAttachedCollisionObjectMsg(msg);
}

void removeWorldObjectDiagnostic(planning_scene::PlanningScene& scene, const std::string& name)
{
  moveit_msgs::msg::CollisionObject msg;
  msg.id = name;
  msg.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processCollisionObjectMsg(msg);
}

void stripWorldAndAttachedDiagnostic(planning_scene::PlanningScene& scene)
{
  std::vector<const moveit::core::AttachedBody*> attached;
  scene.getCurrentState().getAttachedBodies(attached);
  for (const auto* body : attached)
  {
    detachObjectDiagnostic(scene, body->getName());
  }
  for (const auto& name : scene.getWorld()->getObjectIds())
  {
    removeWorldObjectDiagnostic(scene, name);
  }
}

DifferentialCollision diagnoseIkCollisions(const planning_scene::PlanningScene& lift_scene,
                                           const std::map<std::string, double>& joints,
                                           const CollisionDiagConfig& cfg)
{
  DifferentialCollision report;
  {
    auto scene = cloneDiagnosticScene(lift_scene);
    applyJointsToScene(*scene, joints);
    report.full = checkSceneContacts(*scene, cfg, false);
  }
  {
    auto scene = cloneDiagnosticScene(lift_scene);
    applyJointsToScene(*scene, joints);
    detachObjectDiagnostic(*scene, cfg.object_id);
    report.no_part = checkSceneContacts(*scene, cfg, false);
  }
  {
    auto scene = cloneDiagnosticScene(lift_scene);
    applyJointsToScene(*scene, joints);
    stripWorldAndAttachedDiagnostic(*scene);
    report.self_only = checkSceneContacts(*scene, cfg, true);
  }
  {
    auto scene = cloneDiagnosticScene(lift_scene);
    applyJointsToScene(*scene, joints);
    removeWorldObjectDiagnostic(*scene, cfg.table_name);
    report.no_table = checkSceneContacts(*scene, cfg, false);
  }
  {
    auto scene = cloneDiagnosticScene(lift_scene);
    applyJointsToScene(*scene, joints);
    removeWorldObjectDiagnostic(*scene, cfg.column_name);
    report.no_column = checkSceneContacts(*scene, cfg, false);
  }
  return report;
}

AttachedGeometryReport inspectAttachedGeometry(const planning_scene::PlanningScene& scene,
                                               const std::string& object_id)
{
  AttachedGeometryReport report;
  const auto& state = scene.getCurrentState();
  if (!state.hasAttachedBody(object_id))
  {
    return report;
  }
  const auto* body = state.getAttachedBody(object_id);
  if (!body)
  {
    return report;
  }
  report.present = true;
  report.attached_link = body->getAttachedLinkName();
  report.touch_links.assign(body->getTouchLinks().begin(), body->getTouchLinks().end());
  const auto& shapes = body->getShapes();
  if (!shapes.empty() && shapes.front())
  {
    switch (shapes.front()->type)
    {
      case shapes::CYLINDER:
      {
        const auto* cyl = static_cast<const shapes::Cylinder*>(shapes.front().get());
        report.shape = "cylinder";
        report.radius = cyl->radius;
        report.height = cyl->length;
        break;
      }
      case shapes::BOX:
      {
        const auto* box = static_cast<const shapes::Box*>(shapes.front().get());
        report.shape = "box";
        report.radius = box->size[0];
        report.height = box->size[2];
        break;
      }
      default:
        report.shape = "unknown";
        break;
    }
  }
  const auto& worlds = body->getGlobalCollisionBodyTransforms();
  if (!worlds.empty())
  {
    report.world_pose = worlds.front();
    report.local_z_in_world = worlds.front().linear() * Eigen::Vector3d::UnitZ();
  }
  return report;
}
}  // namespace fr_task_planner
