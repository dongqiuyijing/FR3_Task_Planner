// KEYPOSE OPTIMIZATION V1: multi-IK keypose candidates only.
// No execute, no FollowJointTrajectory, no gripper command, no path planning.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <yaml-cpp/yaml.h>

namespace
{
constexpr char kGroupA[] = "arm_a";
constexpr char kGroupB[] = "arm_b";
constexpr char kTcpA[] = "arm_a_gripper_tcp";
constexpr char kTcpB[] = "arm_b_gripper_tcp";
constexpr char kTable[] = "table";
constexpr char kColumn[] = "mounting_column";
constexpr char kPart[] = "small_part";

const std::vector<std::string> kArmA = {"arm_a_j1", "arm_a_j2", "arm_a_j3",
                                        "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kArmB = {"arm_b_j1", "arm_b_j2", "arm_b_j3",
                                        "arm_b_j4", "arm_b_j5", "arm_b_j6"};
const std::vector<std::string> kTouchA = {
    "arm_a_gripper_base_link", "arm_a_rail_155", "arm_a_slider_l", "arm_a_slider_r",
    "arm_a_finger_l",          "arm_a_finger_r", "arm_a_gripper_gap_link", "arm_a_gripper_tcp"};
const std::vector<std::string> kTouchB = {
    "arm_b_gripper_base_link", "arm_b_rail_155", "arm_b_slider_l", "arm_b_slider_r",
    "arm_b_finger_l",          "arm_b_finger_r", "arm_b_gripper_gap_link", "arm_b_gripper_tcp"};

struct Weights
{
  double w[6] = {8.0, 8.0, 1.0, 1.0, 1.0, 0.15};
};

struct IkCfg
{
  int max_attempts = 96;
  int max_unique = 24;
  int top_n = 3;
  int handover_top_n = 5;
  double min_distance = 0.12;
  double timeout_exact = 0.20;
  double timeout_nearby = 0.08;
  double nearby = 1.6;
  double nearby_local = 0.6;
  double pos_tol = 0.003;
  double ori_tol_deg = 2.0;
  double face_center_tol = 0.002;
  double face_normal_tol_deg = 3.0;
  double face_up_tol_deg = 5.0;
  unsigned rng_seed = 42;
};

struct WorkcellBox
{
  std::string name;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3d dim = Eigen::Vector3d::Ones();
};

struct IkRejectStats
{
  int fk = 0;
  int bounds = 0;
  int collision = 0;
  int dup = 0;
  std::map<std::string, int> pairs;
  std::vector<std::string> samples;
};

struct Candidate
{
  int index = 0;
  std::vector<double> joints;
  Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d object = Eigen::Isometry3d::Identity();
  double cost = 1e9;
  std::vector<double> delta = {0, 0, 0, 0, 0, 0};
  bool collision_ok = false;
  bool limits_ok = false;
  bool fk_ok = false;
  bool face_ok = true;
  bool baseline = false;
  std::string seed;
  std::string collision_pair = "none";
  double fk_pos_err = 0.0;
  double fk_ori_err_deg = 0.0;
  double face_center_err = 0.0;
  double face_normal_err_deg = 0.0;
  double face_up_err_deg = 0.0;
  Eigen::Vector3d face_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d face_normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d face_up = Eigen::Vector3d::UnitY();
  double joint_limit_margin = 0.0;
};

struct HandoverPair
{
  int index = 0;
  Candidate a;
  Candidate b;
  Candidate a_pre;
  Candidate b_pre;
  Eigen::Isometry3d world_object = Eigen::Isometry3d::Identity();
  double cost = 1e9;
  std::vector<double> d_a_from_face3 = {0, 0, 0, 0, 0, 0};
  std::vector<double> d_a_to_home = {0, 0, 0, 0, 0, 0};
  std::vector<double> d_b_from_pre = {0, 0, 0, 0, 0, 0};
  bool collision_ok = false;
  std::string collision_pair = "none";
  std::string note;
};

void emit(const std::string& s)
{
  std::cout << s << std::endl;
}

std::string fmt(const std::vector<double>& v, int prec = 6)
{
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o << std::setprecision(prec) << "[";
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i)
      o << ", ";
    o << v[i];
  }
  o << "]";
  return o.str();
}

std::string fmt3(const Eigen::Vector3d& v)
{
  return fmt({v.x(), v.y(), v.z()}, 9);
}

Eigen::Quaterniond quatOf(const Eigen::Isometry3d& T)
{
  Eigen::Quaterniond q(T.linear());
  q.normalize();
  return q;
}

bool yamlVec(const YAML::Node& n, std::vector<double>& out)
{
  if (!n || !n.IsSequence())
    return false;
  out.clear();
  for (const auto& v : n)
    out.push_back(v.as<double>());
  return !out.empty();
}

bool yamlVec3(const YAML::Node& n, Eigen::Vector3d& out)
{
  std::vector<double> v;
  if (!yamlVec(n, v) || v.size() < 3)
    return false;
  out = Eigen::Vector3d(v[0], v[1], v[2]);
  return true;
}

Eigen::Isometry3d isoXyzw(const Eigen::Vector3d& xyz, const Eigen::Vector4d& xyzw)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = xyz;
  Eigen::Quaterniond q(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q.normalize();
  T.linear() = q.toRotationMatrix();
  return T;
}

bool loadXyzw(const YAML::Node& n, Eigen::Isometry3d& T)
{
  Eigen::Vector3d xyz;
  std::vector<double> q;
  if (!n || !yamlVec3(n["xyz"], xyz) || !yamlVec(n["xyzw"], q) || q.size() < 4)
    return false;
  T = isoXyzw(xyz, Eigen::Vector4d(q[0], q[1], q[2], q[3]));
  return true;
}

geometry_msgs::msg::Pose poseMsg(const Eigen::Isometry3d& T)
{
  geometry_msgs::msg::Pose p;
  p.position.x = T.translation().x();
  p.position.y = T.translation().y();
  p.position.z = T.translation().z();
  const auto q = quatOf(T);
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
  return p;
}

void poseError(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double& pos, double& ori_deg)
{
  pos = (a.translation() - b.translation()).norm();
  const double d = std::min(1.0, std::abs(quatOf(a).dot(quatOf(b))));
  ori_deg = 2.0 * std::acos(d) * 180.0 / M_PI;
}

double angDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  const double na = a.norm();
  const double nb = b.norm();
  if (na < 1e-9 || nb < 1e-9)
    return 180.0;
  const double c = std::max(-1.0, std::min(1.0, a.dot(b) / (na * nb)));
  return std::acos(c) * 180.0 / M_PI;
}

std::vector<double> jointsOf(const moveit::core::RobotState& st, const std::vector<std::string>& n)
{
  std::vector<double> q(n.size(), 0.0);
  for (size_t i = 0; i < n.size(); ++i)
    q[i] = st.getVariablePosition(n[i]);
  return q;
}

void setJoints(moveit::core::RobotState& st, const std::vector<std::string>& n,
               const std::vector<double>& q)
{
  for (size_t i = 0; i < n.size() && i < q.size(); ++i)
    st.setVariablePosition(n[i], q[i]);
}

std::vector<double> clampJoints(const moveit::core::RobotModelConstPtr& model,
                                const std::vector<std::string>& names, std::vector<double> q)
{
  for (size_t i = 0; i < names.size() && i < q.size(); ++i)
  {
    const auto* j = model->getJointModel(names[i]);
    if (!j || j->getVariableBounds().empty() || !j->getVariableBounds().front().position_bounded_)
      continue;
    q[i] = std::min(j->getVariableBounds().front().max_position_,
                    std::max(j->getVariableBounds().front().min_position_, q[i]));
  }
  return q;
}

Eigen::Vector3d mirrorYz(const Eigen::Vector3d& p)
{
  return Eigen::Vector3d(-p.x(), p.y(), p.z());
}

struct J12Sol
{
  double j1 = 0.0;
  double j2 = 0.0;
};

bool jointLimits(const moveit::core::RobotModelConstPtr& model, const std::string& name, double& lo,
                 double& hi)
{
  const auto* j = model ? model->getJointModel(name) : nullptr;
  if (!j || j->getVariableBounds().empty() || !j->getVariableBounds().front().position_bounded_)
    return false;
  lo = j->getVariableBounds().front().min_position_;
  hi = j->getVariableBounds().front().max_position_;
  return true;
}

bool wrapToLimits(double& q, double lo, double hi)
{
  for (int k = 0; k < 4; ++k)
  {
    if (q >= lo - 1e-9 && q <= hi + 1e-9)
      return true;
    if (q > hi)
      q -= 2.0 * M_PI;
    else if (q < lo)
      q += 2.0 * M_PI;
  }
  return q >= lo - 1e-9 && q <= hi + 1e-9;
}

// Analytical J1/J2 from elbow in the arm's own base frame.
// FR3: J2 origin (0,0,0.14), upperarm length 0.28 along -X of upperarm_link.
std::vector<J12Sol> j12FromElbowBase(const Eigen::Vector3d& e,
                                     const moveit::core::RobotModelConstPtr& model,
                                     const std::string& j1_name, const std::string& j2_name)
{
  constexpr double kL = 0.28;
  constexpr double kSz = 0.14;
  std::vector<J12Sol> out;
  double j1lo = -3.0543, j1hi = 3.0543, j2lo = -4.6251, j2hi = 1.4835;
  jointLimits(model, j1_name, j1lo, j1hi);
  jointLimits(model, j2_name, j2lo, j2hi);
  double s = (kSz - e.z()) / kL;
  if (std::abs(s) > 1.0 + 1e-6)
    return out;
  s = std::max(-1.0, std::min(1.0, s));
  const double a = std::asin(s);
  const double raw[] = {a, M_PI - a, a - 2.0 * M_PI, (M_PI - a) - 2.0 * M_PI};
  std::vector<double> j2s;
  for (double j2 : raw)
  {
    if (j2 < j2lo - 1e-6 || j2 > j2hi + 1e-6)
      continue;
    bool dup = false;
    for (double u : j2s)
      if (std::abs(u - j2) < 1e-4)
        dup = true;
    if (!dup)
      j2s.push_back(j2);
  }
  for (double j2 : j2s)
  {
    const double k = -kL * std::cos(j2);
    if (std::abs(k) < 1e-6)
      continue;
    double j1 = std::atan2(e.y() / k, e.x() / k);
    if (!wrapToLimits(j1, j1lo, j1hi))
      continue;
    out.push_back({j1, j2});
  }
  return out;
}

struct ArmGeom
{
  Eigen::Vector3d shoulder = Eigen::Vector3d::Zero();
  Eigen::Vector3d elbow = Eigen::Vector3d::Zero();
  Eigen::Vector3d upperarm_dir = Eigen::Vector3d::UnitX();
};

ArmGeom armGeom(moveit::core::RobotState& st, bool is_a)
{
  ArmGeom g;
  const char* up = is_a ? "arm_a_upperarm_link" : "arm_b_upperarm_link";
  const char* el = is_a ? "arm_a_forearm_link" : "arm_b_forearm_link";
  g.shoulder = st.getGlobalLinkTransform(up).translation();
  g.elbow = st.getGlobalLinkTransform(el).translation();
  const Eigen::Vector3d v = g.elbow - g.shoulder;
  g.upperarm_dir = v.norm() > 1e-9 ? v.normalized() : Eigen::Vector3d::UnitX();
  return g;
}

ArmGeom geomOf(planning_scene::PlanningScene& scene, bool is_a, const std::vector<double>& q)
{
  moveit::core::RobotState st(scene.getCurrentState());
  setJoints(st, is_a ? kArmA : kArmB, q);
  st.update();
  return armGeom(st, is_a);
}

std::vector<J12Sol> mirrorJ12FromA(planning_scene::PlanningScene& scene,
                                   const moveit::core::RobotModelConstPtr& model,
                                   const std::vector<double>& a_q)
{
  moveit::core::RobotState st(scene.getCurrentState());
  setJoints(st, kArmA, a_q);
  st.update();
  const ArmGeom ga = armGeom(st, true);
  const Eigen::Vector3d el_m = mirrorYz(ga.elbow);
  const Eigen::Isometry3d T_wb = st.getGlobalLinkTransform("arm_b_base_link");
  const Eigen::Vector3d e_b = T_wb.inverse() * el_m;
  auto sols = j12FromElbowBase(e_b, model, "arm_b_j1", "arm_b_j2");
  if (a_q.size() >= 2)
  {
    std::sort(sols.begin(), sols.end(), [&](const J12Sol& u, const J12Sol& v) {
      return std::abs(u.j2 - a_q[1]) < std::abs(v.j2 - a_q[1]);
    });
  }
  return sols;
}

double geometricMirrorCost(planning_scene::PlanningScene& scene, const std::vector<double>& b_q,
                           const Eigen::Vector3d& el_m, const Eigen::Vector3d& dir_m)
{
  const ArmGeom gb = geomOf(scene, false, b_q);
  return 50.0 * (gb.elbow - el_m).norm() + 0.35 * angDeg(gb.upperarm_dir, dir_m) +
         18.0 * std::max(0.0, gb.upperarm_dir.z());
}

std::vector<std::pair<std::string, std::vector<double>>>
makeBMirrorSeeds(planning_scene::PlanningScene& scene, const moveit::core::RobotModelConstPtr& model,
                 const std::vector<double>& a_q, const std::vector<std::vector<double>>& tmpls)
{
  auto sols = mirrorJ12FromA(scene, model, a_q);
  std::vector<std::pair<std::string, std::vector<double>>> seeds;
  int fi = 0;
  for (const auto& sol : sols)
  {
    for (size_t ti = 0; ti < tmpls.size(); ++ti)
    {
      std::vector<double> q = tmpls[ti].size() >= 6 ? tmpls[ti] : std::vector<double>(6, 0.0);
      if (q.size() < 6)
        q.resize(6, 0.0);
      q[0] = sol.j1;
      q[1] = sol.j2;
      q = clampJoints(model, kArmB, q);
      seeds.push_back({"mirror" + std::to_string(fi) + "_t" + std::to_string(ti), q});
    }
    ++fi;
  }
  return seeds;
}

std::vector<double> deltaJoints(const std::vector<double>& from, const std::vector<double>& to)
{
  std::vector<double> d(6, 0.0);
  for (int i = 0; i < 6; ++i)
    if (static_cast<int>(from.size()) > i && static_cast<int>(to.size()) > i)
      d[i] = to[i] - from[i];
  return d;
}

double weightedCost(const std::vector<double>& d, const Weights& w)
{
  double c = 0.0;
  for (int i = 0; i < 6 && i < static_cast<int>(d.size()); ++i)
    c += w.w[i] * std::abs(d[i]);
  return c;
}

double jointL2(const std::vector<double>& a, const std::vector<double>& b)
{
  double s = 0.0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}

int findDup(const std::vector<Candidate>& cands, const std::vector<double>& q, double min_d)
{
  for (size_t i = 0; i < cands.size(); ++i)
    if (jointL2(cands[i].joints, q) < min_d)
      return static_cast<int>(i);
  return -1;
}

double limitMargin(const moveit::core::RobotState& st, const moveit::core::JointModelGroup* g)
{
  double m = 1e9;
  if (!g)
    return 0.0;
  for (const auto* j : g->getActiveJointModels())
  {
    const auto& b = j->getVariableBounds();
    if (b.empty() || !b.front().position_bounded_)
      continue;
    const double q = st.getVariablePosition(j->getName());
    m = std::min(m, q - b.front().min_position_);
    m = std::min(m, b.front().max_position_ - q);
  }
  return std::isfinite(m) ? m : 0.0;
}

void perturb(moveit::core::RobotState& st, const moveit::core::JointModelGroup* g, double radius,
             std::mt19937& rng, bool j6_heavy, bool preserve_j12)
{
  std::uniform_real_distribution<double> dist6(-radius * 2.5, radius * 2.5);
  for (const auto* j : g->getActiveJointModels())
  {
    const std::string& name = j->getName();
    const bool is_j6 = name.size() >= 2 && name.substr(name.size() - 2) == "j6";
    const bool is_j12 = name.size() >= 2 && (name.substr(name.size() - 2) == "j1" ||
                                             name.substr(name.size() - 2) == "j2");
    double r = radius;
    if (j6_heavy && is_j6)
      r = radius * 2.5;
    if ((j6_heavy || preserve_j12) && is_j12)
      r = preserve_j12 ? 0.12 : radius * 0.08;
    std::uniform_real_distribution<double> d(-r, r);
    double q = st.getVariablePosition(name) + (is_j6 && j6_heavy ? dist6(rng) : d(rng));
    const auto& b = j->getVariableBounds();
    if (!b.empty() && b.front().position_bounded_)
      q = std::min(b.front().max_position_, std::max(b.front().min_position_, q));
    st.setVariablePosition(name, q);
  }
  st.update();
}

void clearObject(planning_scene::PlanningScene& scene)
{
  moveit_msgs::msg::AttachedCollisionObject det;
  det.object.id = kPart;
  det.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processAttachedCollisionObjectMsg(det);
  moveit_msgs::msg::CollisionObject rem;
  rem.id = kPart;
  rem.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processCollisionObjectMsg(rem);
}

void addBox(planning_scene::PlanningScene& scene, const WorkcellBox& box)
{
  moveit_msgs::msg::CollisionObject obj;
  obj.id = box.name;
  obj.header.frame_id = scene.getPlanningFrame();
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;
  obj.pose.orientation.w = 1.0;
  obj.primitives.resize(1);
  obj.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
  obj.primitives[0].dimensions = {box.dim.x(), box.dim.y(), box.dim.z()};
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = box.xyz;
  obj.primitive_poses.push_back(poseMsg(T));
  scene.processCollisionObjectMsg(obj);
}

void addWorldCylinder(planning_scene::PlanningScene& scene, const Eigen::Isometry3d& world,
                      double r, double h)
{
  clearObject(scene);
  moveit_msgs::msg::CollisionObject obj;
  obj.id = kPart;
  obj.header.frame_id = scene.getPlanningFrame();
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;
  obj.pose.orientation.w = 1.0;
  obj.primitives.resize(1);
  obj.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  obj.primitives[0].dimensions = {h, r};
  obj.primitive_poses.push_back(poseMsg(world));
  scene.processCollisionObjectMsg(obj);
}

void attachCylinder(planning_scene::PlanningScene& scene, const std::string& link,
                    const Eigen::Isometry3d& in_link, const std::vector<std::string>& touch,
                    double r, double h)
{
  clearObject(scene);
  moveit_msgs::msg::AttachedCollisionObject att;
  att.link_name = link;
  att.touch_links = touch;
  att.object.id = kPart;
  att.object.header.frame_id = link;
  att.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  att.object.pose.orientation.w = 1.0;
  att.object.primitives.resize(1);
  att.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  att.object.primitives[0].dimensions = {h, r};
  att.object.primitive_poses.push_back(poseMsg(in_link));
  scene.processAttachedCollisionObjectMsg(att);
}

bool allowedPair(const std::string& a, const std::string& b, bool allow_part_table,
                 bool allow_ab_fingers)
{
  const bool part_table = (a == kPart && b == kTable) || (a == kTable && b == kPart);
  if (allow_part_table && part_table)
    return true;
  auto is_touch = [](const std::string& n, const std::vector<std::string>& t) {
    return std::find(t.begin(), t.end(), n) != t.end();
  };
  if ((a == kPart && is_touch(b, kTouchA)) || (b == kPart && is_touch(a, kTouchA)))
    return true;
  if ((a == kPart && is_touch(b, kTouchB)) || (b == kPart && is_touch(a, kTouchB)))
    return true;
  (void)allow_ab_fingers;
  if ((a == "arm_a_base_link" && b == kColumn) || (a == kColumn && b == "arm_a_base_link"))
    return true;
  if ((a == "arm_b_base_link" && b == kColumn) || (a == kColumn && b == "arm_b_base_link"))
    return true;
  return false;
}

bool colliding(planning_scene::PlanningScene& scene, moveit::core::RobotState& st, std::string& pair,
               bool allow_part_table, bool allow_ab_fingers)
{
  st.update();
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 80;
  req.max_contacts_per_pair = 1;
  scene.checkCollision(req, res, st);
  if (!res.collision)
    return false;
  if (res.contacts.empty())
  {
    pair = "unspecified";
    return true;
  }
  bool bad = false;
  pair = "none";
  for (const auto& c : res.contacts)
  {
    const std::string& a = c.first.first;
    const std::string& b = c.first.second;
    if (allowedPair(a, b, allow_part_table, allow_ab_fingers))
      continue;
    pair = a + " <-> " + b;
    bad = true;
    break;
  }
  return bad;
}

enum class PartMode
{
  None,
  World,
  OnA,
  OnB
};

void placePart(planning_scene::PlanningScene& scene, PartMode mode, const Eigen::Isometry3d& tcp_obj,
               const Eigen::Isometry3d& world_obj, double r, double h)
{
  if (mode == PartMode::None)
  {
    clearObject(scene);
    return;
  }
  if (mode == PartMode::World)
  {
    addWorldCylinder(scene, world_obj, r, h);
    return;
  }
  if (mode == PartMode::OnA)
    attachCylinder(scene, kTcpA, tcp_obj, kTouchA, r, h);
  else
    attachCylinder(scene, kTcpB, tcp_obj, kTouchB, r, h);
}

void applyState(planning_scene::PlanningScene& scene, const std::vector<double>& a,
                const std::vector<double>& b, double qa, double qb)
{
  auto st = scene.getCurrentStateNonConst();
  setJoints(st, kArmA, a);
  setJoints(st, kArmB, b);
  if (st.getRobotModel()->hasJointModel("arm_a_gripper_joint"))
    st.setVariablePosition("arm_a_gripper_joint", qa);
  if (st.getRobotModel()->hasJointModel("arm_b_gripper_joint"))
    st.setVariablePosition("arm_b_gripper_joint", qb);
  st.update();
  scene.setCurrentState(st);
}

bool checkState(planning_scene::PlanningScene& scene, const std::vector<double>& a,
                const std::vector<double>& b, double qa, double qb, PartMode mode,
                const Eigen::Isometry3d& tcp_obj, const Eigen::Isometry3d& world_obj, double r,
                double h, bool allow_part_table, bool allow_ab_fingers, std::string& pair)
{
  placePart(scene, mode, tcp_obj, world_obj, r, h);
  applyState(scene, a, b, qa, qb);
  auto st = scene.getCurrentStateNonConst();
  return colliding(scene, st, pair, allow_part_table, allow_ab_fingers);
}

void writeVec(std::ostream& os, const std::string& indent, const std::string& key,
              const std::vector<double>& v)
{
  os << indent << key << ": [";
  os.setf(std::ios::fixed);
  os << std::setprecision(12);
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i)
      os << ", ";
    os << v[i];
  }
  os << "]\n";
}

void writePose(std::ostream& os, const std::string& indent, const std::string& key,
               const Eigen::Isometry3d& T)
{
  const auto q = quatOf(T);
  os << indent << key << ":\n";
  writeVec(os, indent + "  ", "xyz",
           {T.translation().x(), T.translation().y(), T.translation().z()});
  writeVec(os, indent + "  ", "xyzw", {q.x(), q.y(), q.z(), q.w()});
}

void writeCand(std::ostream& os, const Candidate& c, bool face)
{
  os << "    - index: " << c.index << "\n";
  os << "      seed: \"" << c.seed << "\"\n";
  os << "      baseline: " << (c.baseline ? "true" : "false") << "\n";
  writeVec(os, "      ", "joint_values", c.joints);
  writePose(os, "      ", "task_space_pose", c.tcp);
  os << "      cost: " << std::setprecision(9) << c.cost << "\n";
  writeVec(os, "      ", "delta_joints", c.delta);
  os << "      collision_check: " << (c.collision_ok ? "PASS" : "FAIL") << "\n";
  if (!c.collision_ok)
    os << "      collision_pair: \"" << c.collision_pair << "\"\n";
  os << "      joint_limit_check: " << (c.limits_ok ? "PASS" : "FAIL") << "\n";
  os << "      joint_limit_margin: " << c.joint_limit_margin << "\n";
  os << "      fk_ok: " << (c.fk_ok ? "true" : "false") << "\n";
  os << "      fk_position_error_m: " << c.fk_pos_err << "\n";
  os << "      fk_orientation_error_deg: " << c.fk_ori_err_deg << "\n";
  if (face)
  {
    writeVec(os, "      ", "face_center",
             {c.face_center.x(), c.face_center.y(), c.face_center.z()});
    writeVec(os, "      ", "face_normal",
             {c.face_normal.x(), c.face_normal.y(), c.face_normal.z()});
    writeVec(os, "      ", "face_up", {c.face_up.x(), c.face_up.y(), c.face_up.z()});
    os << "      face_center_error_m: " << c.face_center_err << "\n";
    os << "      face_normal_error_deg: " << c.face_normal_err_deg << "\n";
    os << "      face_up_error_deg: " << c.face_up_err_deg << "\n";
    os << "      face_pose_check: " << (c.face_ok ? "PASS" : "FAIL") << "\n";
  }
}

void fillFace(Candidate& c, const Eigen::Isometry3d& T_tcp_obj, const Eigen::Vector3d& c_obj,
              const Eigen::Vector3d& n_obj, const Eigen::Vector3d& u_obj,
              const Eigen::Vector3d& p1, const Eigen::Vector3d& d1, const Eigen::Vector3d& up,
              const IkCfg& ik)
{
  c.object = c.tcp * T_tcp_obj;
  c.face_center = c.object * c_obj;
  c.face_normal = (c.object.linear() * n_obj).normalized();
  c.face_up = (c.object.linear() * u_obj).normalized();
  c.face_center_err = (c.face_center - p1).norm();
  c.face_normal_err_deg = angDeg(c.face_normal, d1);
  c.face_up_err_deg = angDeg(c.face_up, up);
  c.face_ok = c.face_center_err <= ik.face_center_tol &&
              c.face_normal_err_deg <= ik.face_normal_tol_deg &&
              c.face_up_err_deg <= ik.face_up_tol_deg;
}

std::vector<Candidate> searchIk(planning_scene::PlanningScene& scene, const std::string& group_name,
                                const std::string& tcp, const Eigen::Isometry3d& target,
                                const std::vector<double>& other_arm, const std::vector<double>& ref,
                                const std::vector<std::pair<std::string, std::vector<double>>>& seeds,
                                const Weights& w, const IkCfg& ik, PartMode mode,
                                const Eigen::Isometry3d& tcp_obj, const Eigen::Isometry3d& world_obj,
                                double r, double h, double qa, double qb, bool allow_part_table,
                                bool j6_heavy, bool is_a, unsigned seed_off,
                                IkRejectStats* stats = nullptr, std::vector<Candidate>* rejected = nullptr,
                                bool preserve_j12 = false)
{
  const auto model = scene.getRobotModel();
  const auto* group = model->getJointModelGroup(group_name);
  const auto& names = is_a ? kArmA : kArmB;
  std::mt19937 rng(ik.rng_seed + seed_off);
  std::vector<Candidate> unique;
  int attempts = 0;
  int success = 0;
  IkRejectStats local_stats;
  if (!stats)
    stats = &local_stats;

  auto consider = [&](moveit::core::RobotState& ik_state, const std::string& seed_name,
                      double timeout) {
    if (attempts >= ik.max_attempts || static_cast<int>(unique.size()) >= ik.max_unique)
      return;
    ++attempts;
    if (!ik_state.setFromIK(group, target, tcp, timeout))
      return;
    ++success;
    Candidate c;
    c.joints = jointsOf(ik_state, names);
    c.seed = seed_name;
    c.tcp = ik_state.getGlobalLinkTransform(tcp);
    poseError(target, c.tcp, c.fk_pos_err, c.fk_ori_err_deg);
    c.fk_ok = c.fk_pos_err <= ik.pos_tol && c.fk_ori_err_deg <= ik.ori_tol_deg;
    c.limits_ok = ik_state.satisfiesBounds(group);
    c.joint_limit_margin = limitMargin(ik_state, group);
    if (!c.fk_ok)
    {
      ++stats->fk;
      return;
    }
    if (!c.limits_ok)
    {
      ++stats->bounds;
      return;
    }
    if (findDup(unique, c.joints, ik.min_distance) >= 0)
    {
      ++stats->dup;
      return;
    }
    std::vector<double> qa_j = is_a ? c.joints : other_arm;
    std::vector<double> qb_j = is_a ? other_arm : c.joints;
    c.collision_ok = !checkState(scene, qa_j, qb_j, qa, qb, mode, tcp_obj, world_obj, r, h,
                                 allow_part_table, mode == PartMode::OnB || mode == PartMode::OnA,
                                 c.collision_pair);
    if (!c.collision_ok)
    {
      ++stats->collision;
      stats->pairs[c.collision_pair]++;
      if (stats->samples.size() < 8)
        stats->samples.push_back(c.collision_pair + " seed=" + seed_name + " q=" + fmt(c.joints, 4));
      if (rejected && static_cast<int>(rejected->size()) < 12)
      {
        c.delta = deltaJoints(ref, c.joints);
        c.cost = weightedCost(c.delta, w);
        rejected->push_back(c);
      }
      return;
    }
    c.delta = deltaJoints(ref, c.joints);
    c.cost = weightedCost(c.delta, w);
    unique.push_back(std::move(c));
  };

  for (const auto& seed : seeds)
  {
    if (attempts >= ik.max_attempts || static_cast<int>(unique.size()) >= ik.max_unique)
      break;
    moveit::core::RobotState exact(scene.getCurrentState());
    setJoints(exact, is_a ? kArmA : kArmB, seed.second);
    setJoints(exact, is_a ? kArmB : kArmA, other_arm);
    exact.update();
    consider(exact, seed.first + "_exact", ik.timeout_exact);
    const int nearby_n = j6_heavy ? 10 : 6;
    for (int i = 0; i < nearby_n; ++i)
    {
      if (attempts >= ik.max_attempts || static_cast<int>(unique.size()) >= ik.max_unique)
        break;
      moveit::core::RobotState nearby(exact);
      const double radius = (i < nearby_n / 2) ? ik.nearby_local : ik.nearby;
      perturb(nearby, group, radius, rng, j6_heavy, preserve_j12);
      consider(nearby, seed.first + "_nearby", ik.timeout_nearby);
    }
    if (j6_heavy)
    {
      static const double j6off[] = {-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
      for (double off : j6off)
      {
        if (attempts >= ik.max_attempts || static_cast<int>(unique.size()) >= ik.max_unique)
          break;
        moveit::core::RobotState s(exact);
        auto q = seed.second;
        if (q.size() >= 6)
        {
          q[5] += off;
          const auto* j6 = model->getJointModel(names[5]);
          if (j6 && !j6->getVariableBounds().empty() && j6->getVariableBounds().front().position_bounded_)
          {
            q[5] = std::min(j6->getVariableBounds().front().max_position_,
                            std::max(j6->getVariableBounds().front().min_position_, q[5]));
          }
        }
        setJoints(s, names, q);
        s.update();
        consider(s, seed.first + "_j6off", ik.timeout_exact);
      }
    }
  }

  std::sort(unique.begin(), unique.end(),
            [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
  std::ostringstream rs;
  rs << "  IK " << group_name << " attempts=" << attempts << " success=" << success
     << " unique_valid=" << unique.size() << " reject fk=" << stats->fk
     << " bounds=" << stats->bounds << " col=" << stats->collision << " dup=" << stats->dup;
  if (!stats->pairs.empty())
  {
    rs << " pairs:";
    std::vector<std::pair<int, std::string>> ranked;
    for (const auto& kv : stats->pairs)
      ranked.push_back({kv.second, kv.first});
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < ranked.size() && i < 4; ++i)
      rs << " [" << ranked[i].second << "=" << ranked[i].first << "]";
  }
  emit(rs.str());
  return unique;
}

std::vector<Candidate> topN(const std::vector<Candidate>& in, int n)
{
  std::vector<Candidate> out;
  for (size_t i = 0; i < in.size() && static_cast<int>(out.size()) < n; ++i)
  {
    Candidate c = in[i];
    c.index = static_cast<int>(out.size()) + 1;
    out.push_back(c);
  }
  return out;
}

WorkcellBox loadBox(const YAML::Node& n, const std::string& fallback)
{
  WorkcellBox b;
  b.name = n["name"] ? n["name"].as<std::string>() : fallback;
  if (n["initial_pose"] && n["initial_pose"]["position"])
  {
    b.xyz.x() = n["initial_pose"]["position"]["x"].as<double>();
    b.xyz.y() = n["initial_pose"]["position"]["y"].as<double>();
    b.xyz.z() = n["initial_pose"]["position"]["z"].as<double>();
  }
  if (n["dimensions"])
  {
    b.dim.x() = n["dimensions"]["x"].as<double>();
    b.dim.y() = n["dimensions"]["y"].as<double>();
    b.dim.z() = n["dimensions"]["z"].as<double>();
  }
  return b;
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("keypose_candidate_search", opt);

  emit("========== KEYPOSE OPTIMIZATION V1 ==========");
  emit("IK CANDIDATES ONLY. NO EXECUTION. NO PATH PLANNING.");

  const std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string search_yaml = node->get_parameter_or(
      "search_yaml", home_dir + "/fr_task_ws/src/fr_task_planner/config/"
                                "keypose_optimization_v1/search.yaml");
  YAML::Node cfg = YAML::LoadFile(search_yaml);
  const std::string out_dir = cfg["output_dir"].as<std::string>();
  Weights w;
  Weights w2;
  if (cfg["cost_weights"])
  {
    w.w[0] = cfg["cost_weights"]["j1"].as<double>(8.0);
    w.w[1] = cfg["cost_weights"]["j2"].as<double>(8.0);
    w.w[2] = cfg["cost_weights"]["j3"].as<double>(1.0);
    w.w[3] = cfg["cost_weights"]["j4"].as<double>(1.0);
    w.w[4] = cfg["cost_weights"]["j5"].as<double>(1.0);
    w.w[5] = cfg["cost_weights"]["j6"].as<double>(0.15);
  }
  if (cfg["face2_cost_weights"])
  {
    w2.w[0] = cfg["face2_cost_weights"]["j1"].as<double>(20.0);
    w2.w[1] = cfg["face2_cost_weights"]["j2"].as<double>(20.0);
    w2.w[2] = cfg["face2_cost_weights"]["j3"].as<double>(12.0);
    w2.w[3] = cfg["face2_cost_weights"]["j4"].as<double>(12.0);
    w2.w[4] = cfg["face2_cost_weights"]["j5"].as<double>(12.0);
    w2.w[5] = cfg["face2_cost_weights"]["j6"].as<double>(0.05);
  }
  IkCfg ik;
  if (cfg["ik"])
  {
    ik.max_attempts = cfg["ik"]["max_attempts"].as<int>(96);
    ik.max_unique = cfg["ik"]["max_unique_internal"].as<int>(24);
    ik.top_n = cfg["ik"]["top_n"].as<int>(3);
    ik.handover_top_n = cfg["ik"]["handover_top_n"].as<int>(5);
    ik.min_distance = cfg["ik"]["min_solution_distance"].as<double>(0.12);
    ik.timeout_exact = cfg["ik"]["timeout_exact"].as<double>(0.20);
    ik.timeout_nearby = cfg["ik"]["timeout_nearby"].as<double>(0.08);
    ik.nearby = cfg["ik"]["nearby_radius"].as<double>(1.6);
    ik.nearby_local = cfg["ik"]["nearby_radius_local"].as<double>(0.6);
    ik.pos_tol = cfg["ik"]["pos_tol_m"].as<double>(0.003);
    ik.ori_tol_deg = cfg["ik"]["ori_tol_deg"].as<double>(2.0);
    ik.face_center_tol = cfg["ik"]["face_center_tol_m"].as<double>(0.002);
    ik.face_normal_tol_deg = cfg["ik"]["face_normal_tol_deg"].as<double>(3.0);
    ik.face_up_tol_deg = cfg["ik"]["face_up_tol_deg"].as<double>(5.0);
    ik.rng_seed = static_cast<unsigned>(cfg["ik"]["rng_seed"].as<int>(42));
  }
  const double qa_open = cfg["gripper"]["q_a_open"].as<double>(0.0);
  const double qa_grasp = cfg["gripper"]["q_a_grasp"].as<double>(0.083);
  const double qb_open = cfg["gripper"]["q_b_open"].as<double>(0.0);
  const double qb_hold = cfg["gripper"]["q_b_hold"].as<double>(0.083);
  const double radius = cfg["object"]["radius_m"].as<double>(0.0075);
  const double height = cfg["object"]["height_m"].as<double>(0.035);
  const double a_retract = cfg["pre_handover"]["a_retract_m"].as<double>(0.10);
  const double b_retract = cfg["pre_handover"]["b_retract_m"].as<double>(0.20);

  emit("cost_weights J1..J6 = " +
       fmt({w.w[0], w.w[1], w.w[2], w.w[3], w.w[4], w.w[5]}));
  emit("face2_cost_weights J1..J6 = " +
       fmt({w2.w[0], w2.w[1], w2.w[2], w2.w[3], w2.w[4], w2.w[5]}));

  YAML::Node home_y = YAML::LoadFile(cfg["home_yaml"].as<std::string>());
  YAML::Node s15 = YAML::LoadFile(cfg["step15_winner"].as<std::string>());
  YAML::Node s12 = YAML::LoadFile(cfg["step12c_winner"].as<std::string>());
  YAML::Node han_y = YAML::LoadFile(cfg["handover_yaml"].as<std::string>());
  YAML::Node bseq = YAML::LoadFile(cfg["b_inspection_yaml"].as<std::string>());
  YAML::Node six = YAML::LoadFile(cfg["six_face_yaml"].as<std::string>());
  YAML::Node work = YAML::LoadFile(cfg["workcell_yaml"].as<std::string>());

  std::vector<double> home_a, home_b, pre_b, han_a, han_b, pregrasp, grasp, lift, face1, face2, face3;
  yamlVec(home_y["home_a"], home_a);
  yamlVec(home_y["home_b"], home_b);
  yamlVec(home_y["pre_b"], pre_b);
  yamlVec(six["handover_a"], han_a);
  yamlVec(six["handover_b"], han_b);
  yamlVec(s15["pregrasp_joints_rad"], pregrasp);
  yamlVec(s15["grasp_joints_rad"], grasp);
  yamlVec(s15["lift_joints_rad"], lift);
  yamlVec(s15["a_joints_rad"], face1);
  yamlVec(s15["b_joints_rad"], face2);
  yamlVec(s15["c_joints_rad"], face3);

  Eigen::Isometry3d T_tcpA_obj = Eigen::Isometry3d::Identity();
  T_tcpA_obj.linear() = Eigen::Quaterniond(0.0, -1.0, 0.0, 0.0).toRotationMatrix();
  Eigen::Isometry3d T_tcpB_obj = Eigen::Isometry3d::Identity();
  T_tcpB_obj.translation() = Eigen::Vector3d(0, 0, 0.008);
  T_tcpB_obj.linear() = Eigen::Quaterniond(0.707107, 0.0, 0.0, -0.707107).toRotationMatrix();

  const Eigen::Vector3d p1(s12["p1"][0].as<double>(), s12["p1"][1].as<double>(),
                           s12["p1"][2].as<double>());
  const Eigen::Vector3d d1(s12["surface_target_normal"][0].as<double>(),
                           s12["surface_target_normal"][1].as<double>(),
                           s12["surface_target_normal"][2].as<double>());
  const Eigen::Vector3d up_world(0.0, 0.707106781, 0.707106781);

  struct FaceDef
  {
    std::string name;
    std::string physical;
    Eigen::Vector3d c_obj;
    Eigen::Vector3d n_obj;
    Eigen::Vector3d u_obj;
    Eigen::Isometry3d object;
  };
  auto loadFace = [&](const YAML::Node& n, const std::string& fallback_name) {
    FaceDef f;
    f.name = fallback_name;
    f.physical = n["physical_id"] ? n["physical_id"].as<std::string>() : "";
    yamlVec3(n["center_in_object"], f.c_obj);
    yamlVec3(n["normal_in_object"], f.n_obj);
    yamlVec3(n["up_in_object"], f.u_obj);
    loadXyzw(n["object_world"], f.object);
    return f;
  };
  FaceDef f_b4 = loadFace(bseq["inspection_views"]["side_pos_x"], "B_FACE4");
  FaceDef f_b5 = loadFace(bseq["inspection_views"]["side_neg_x"], "B_FACE5");
  FaceDef f_b6 = loadFace(bseq["inspection_views"]["top_circle"], "B_FACE6");

  FaceDef f_a1, f_a2, f_a3;
  f_a1.name = "A_FACE1";
  f_a1.physical = "+Y";
  f_a1.c_obj = Eigen::Vector3d(0.0, radius, 0.0);
  f_a1.n_obj = Eigen::Vector3d(0.0, 1.0, 0.0);
  f_a1.u_obj = Eigen::Vector3d(0.0, 0.0, 1.0);
  loadXyzw(s12["A"]["object_world"], f_a1.object);
  f_a2.name = "A_FACE2";
  f_a2.physical = "-Y";
  f_a2.c_obj = Eigen::Vector3d(0.0, -radius, 0.0);
  f_a2.n_obj = Eigen::Vector3d(0.0, -1.0, 0.0);
  f_a2.u_obj = Eigen::Vector3d(0.0, 0.0, 1.0);
  loadXyzw(s12["B"]["object_world"], f_a2.object);
  f_a3.name = "A_FACE3";
  f_a3.physical = "-Z";
  f_a3.c_obj = Eigen::Vector3d(0.0, 0.0, -height * 0.5);
  f_a3.n_obj = Eigen::Vector3d(0.0, 0.0, -1.0);
  f_a3.u_obj = Eigen::Vector3d(0.0, 1.0, 0.0);
  loadXyzw(s12["C_bottom"]["object_world"], f_a3.object);

  robot_model_loader::RobotModelLoader loader(node);
  auto model = loader.getModel();
  if (!model || model->getName() != "fairino3_dual_robot")
  {
    emit("FAIL: expected fairino3_dual_robot model");
    rclcpp::shutdown();
    return 1;
  }
  if (!model->hasJointModelGroup(kGroupA) || !model->hasJointModelGroup(kGroupB) ||
      !model->hasLinkModel(kTcpA) || !model->hasLinkModel(kTcpB))
  {
    emit("FAIL: missing arm groups or TCP links");
    rclcpp::shutdown();
    return 1;
  }
  auto* ga = model->getJointModelGroup(kGroupA);
  auto* gb = model->getJointModelGroup(kGroupB);
  if (!ga->getSolverInstance() || !gb->getSolverInstance() || !ga->canSetStateFromIK(kTcpA) ||
      !gb->canSetStateFromIK(kTcpB))
  {
    emit("FAIL: IK solver missing for arm_a / arm_b TCP");
    rclcpp::shutdown();
    return 1;
  }

  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  addBox(*scene, loadBox(work["table"], kTable));
  addBox(*scene, loadBox(work["column"], kColumn));
  auto& acm = scene->getAllowedCollisionMatrixNonConst();
  acm.setEntry("arm_a_base_link", kColumn, true);
  acm.setEntry("arm_b_base_link", kColumn, true);
  for (const auto& t : kTouchA)
    acm.setEntry(kPart, t, true);
  for (const auto& t : kTouchB)
    acm.setEntry(kPart, t, true);

  moveit::core::RobotState start(model);
  start.setToDefaultValues();
  setJoints(start, kArmA, home_a);
  setJoints(start, kArmB, home_b);
  start.update();
  scene->setCurrentState(start);

  auto fkTcp = [&](bool is_a, const std::vector<double>& q) {
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, is_a ? kArmA : kArmB, q);
    setJoints(st, is_a ? kArmB : kArmA, is_a ? home_b : home_a);
    st.update();
    return st.getGlobalLinkTransform(is_a ? kTcpA : kTcpB);
  };

  const Eigen::Isometry3d tcp_home_a = fkTcp(true, home_a);
  const Eigen::Isometry3d tcp_pregrasp = fkTcp(true, pregrasp);
  const Eigen::Isometry3d tcp_grasp = fkTcp(true, grasp);
  const Eigen::Isometry3d tcp_lift = fkTcp(true, lift);
  const Eigen::Isometry3d tcp_a1 = f_a1.object * T_tcpA_obj.inverse();
  const Eigen::Isometry3d tcp_a2 = f_a2.object * T_tcpA_obj.inverse();
  const Eigen::Isometry3d tcp_a3 = f_a3.object * T_tcpA_obj.inverse();
  const Eigen::Isometry3d tcp_b4 = f_b4.object * T_tcpB_obj.inverse();
  const Eigen::Isometry3d tcp_b5 = f_b5.object * T_tcpB_obj.inverse();
  const Eigen::Isometry3d tcp_b6 = f_b6.object * T_tcpB_obj.inverse();
  const Eigen::Isometry3d world_grasp_obj = tcp_grasp * T_tcpA_obj;

  std::map<std::string, std::vector<Candidate>> arm_a;
  std::map<std::string, std::vector<Candidate>> arm_b;
  std::vector<HandoverPair> pairs;

  auto keepBaseline = [&](std::vector<Candidate>& cands, const std::vector<double>& q, bool is_a,
                          const Eigen::Isometry3d& target, const std::vector<double>& ref,
                          const Weights& ww, PartMode mode, const Eigen::Isometry3d& world_obj,
                          bool allow_part_table, const FaceDef* face) {
    Candidate c;
    c.joints = q;
    c.seed = "baseline";
    c.baseline = true;
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, is_a ? kArmA : kArmB, q);
    setJoints(st, is_a ? kArmB : kArmA, is_a ? home_b : home_a);
    st.update();
    c.tcp = st.getGlobalLinkTransform(is_a ? kTcpA : kTcpB);
    poseError(target, c.tcp, c.fk_pos_err, c.fk_ori_err_deg);
    c.fk_ok = c.fk_pos_err <= ik.pos_tol && c.fk_ori_err_deg <= ik.ori_tol_deg;
    c.limits_ok = st.satisfiesBounds(is_a ? ga : gb);
    c.joint_limit_margin = limitMargin(st, is_a ? ga : gb);
    std::vector<double> qa_j = is_a ? q : home_a;
    std::vector<double> qb_j = is_a ? home_b : q;
    const double qa = (mode == PartMode::OnA || mode == PartMode::World) ? qa_grasp : qa_open;
    const double qb = (mode == PartMode::OnB) ? qb_hold : qb_open;
    c.collision_ok = !checkState(*scene, qa_j, qb_j, qa, qb, mode, is_a ? T_tcpA_obj : T_tcpB_obj,
                                 world_obj, radius, height, allow_part_table, true, c.collision_pair);
    c.delta = deltaJoints(ref, q);
    c.cost = weightedCost(c.delta, ww);
    if (face)
    {
      const Eigen::Vector3d exp_up = (face->object.linear() * face->u_obj).normalized();
      fillFace(c, is_a ? T_tcpA_obj : T_tcpB_obj, face->c_obj, face->n_obj, face->u_obj, p1, d1,
               exp_up, ik);
    }
    const bool face_pass = !face || c.face_ok;
    if (c.fk_ok && c.limits_ok && c.collision_ok && face_pass &&
        findDup(cands, q, ik.min_distance) < 0)
      cands.insert(cands.begin(), c);
  };

  auto runA = [&](const std::string& name, const Eigen::Isometry3d& target,
                  const std::vector<double>& ref, const std::vector<double>& baseline_q,
                  const Weights& ww, PartMode mode, const Eigen::Isometry3d& world_obj,
                  bool allow_part_table, bool j6_heavy, const FaceDef* face,
                  const std::vector<std::pair<std::string, std::vector<double>>>& extra) {
    emit("---- " + name + " ----");
    std::vector<std::pair<std::string, std::vector<double>>> seeds = extra;
    seeds.push_back({"baseline", baseline_q});
    seeds.push_back({"ref", ref});
    seeds.push_back({"home", home_a});
    auto found =
        searchIk(*scene, kGroupA, kTcpA, target, home_b, ref, seeds, ww, ik, mode, T_tcpA_obj,
                 world_obj, radius, height, (mode == PartMode::None) ? qa_open : qa_grasp, qb_open,
                 allow_part_table, j6_heavy, true, static_cast<unsigned>(name.size() * 17));
    keepBaseline(found, baseline_q, true, target, ref, ww, mode, world_obj, allow_part_table, face);
    if (face)
    {
      const Eigen::Vector3d exp_up = (face->object.linear() * face->u_obj).normalized();
      for (auto& c : found)
        fillFace(c, T_tcpA_obj, face->c_obj, face->n_obj, face->u_obj, p1, d1, exp_up, ik);
      found.erase(std::remove_if(found.begin(), found.end(),
                                 [](const Candidate& c) { return !c.face_ok; }),
                  found.end());
    }
    std::sort(found.begin(), found.end(),
              [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
    arm_a[name] = topN(found, ik.top_n);
    if (arm_a[name].empty())
      emit("  NO LEGAL CANDIDATE for " + name);
    else
      emit("  kept " + std::to_string(arm_a[name].size()) + "  best_cost=" +
           std::to_string(arm_a[name].front().cost));
  };

  auto runB = [&](const std::string& name, const Eigen::Isometry3d& target,
                  const std::vector<double>& ref, const std::vector<double>& baseline_q,
                  const Weights& ww, PartMode mode, const Eigen::Isometry3d& world_obj,
                  bool allow_part_table, const FaceDef* face,
                  const std::vector<std::pair<std::string, std::vector<double>>>& extra) {
    emit("---- " + name + " ----");
    std::vector<std::pair<std::string, std::vector<double>>> seeds = extra;
    seeds.push_back({"baseline", baseline_q});
    seeds.push_back({"ref", ref});
    seeds.push_back({"home", home_b});
    seeds.push_back({"pre", pre_b});
    auto found =
        searchIk(*scene, kGroupB, kTcpB, target, home_a, ref, seeds, ww, ik, mode, T_tcpB_obj,
                 world_obj, radius, height, qa_open, (mode == PartMode::OnB) ? qb_hold : qb_open,
                 allow_part_table, false, false, static_cast<unsigned>(name.size() * 29));
    keepBaseline(found, baseline_q, false, target, ref, ww, mode, world_obj, allow_part_table, face);
    if (face)
    {
      const Eigen::Vector3d exp_up = (face->object.linear() * face->u_obj).normalized();
      for (auto& c : found)
        fillFace(c, T_tcpB_obj, face->c_obj, face->n_obj, face->u_obj, p1, d1, exp_up, ik);
      found.erase(std::remove_if(found.begin(), found.end(),
                                 [](const Candidate& c) { return !c.face_ok; }),
                  found.end());
    }
    std::sort(found.begin(), found.end(),
              [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
    arm_b[name] = topN(found, ik.top_n);
    if (arm_b[name].empty())
      emit("  NO LEGAL CANDIDATE for " + name);
    else
      emit("  kept " + std::to_string(arm_b[name].size()) + "  best_cost=" +
           std::to_string(arm_b[name].front().cost));
  };

  // A_HOME: keep existing, do not modify.
  {
    Candidate c;
    c.index = 1;
    c.joints = home_a;
    c.tcp = tcp_home_a;
    c.seed = "dual8_home";
    c.baseline = true;
    c.cost = 0.0;
    c.delta = {0, 0, 0, 0, 0, 0};
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, kArmA, home_a);
    setJoints(st, kArmB, home_b);
    st.update();
    c.limits_ok = st.satisfiesBounds(ga);
    c.joint_limit_margin = limitMargin(st, ga);
    c.fk_ok = true;
    c.collision_ok = !checkState(*scene, home_a, home_b, qa_open, qb_open, PartMode::None,
                                 T_tcpA_obj, world_grasp_obj, radius, height, false, false,
                                 c.collision_pair);
    arm_a["A_HOME"] = {c};
    emit("---- A_HOME ---- kept existing dual8_home.yaml (not modified)");
  }

  runA("A_PREGRASP", tcp_pregrasp, home_a, pregrasp, w, PartMode::World, world_grasp_obj, true,
       false, nullptr, {});
  runA("A_GRASP", tcp_grasp, pregrasp, grasp, w, PartMode::World, world_grasp_obj, true, false,
       nullptr, {{"pregrasp", pregrasp}});
  runA("A_LIFT", tcp_lift, grasp, lift, w, PartMode::OnA, world_grasp_obj, true, false, nullptr,
       {{"grasp", grasp}});
  runA("A_FACE1", tcp_a1, lift, face1, w, PartMode::OnA, f_a1.object, false, false, &f_a1,
       {{"lift", lift}});
  runA("A_FACE2", tcp_a2, face1, face2, w2, PartMode::OnA, f_a2.object, false, true, &f_a2,
       {{"face1", face1}});
  runA("A_FACE3", tcp_a3, face2, face3, w, PartMode::OnA, f_a3.object, false, false, &f_a3,
       {{"face2", face2}, {"face1", face1}});

  // Handover pair search: same T_world_object, freeze existing grasp transforms.
  emit("---- HANDOVER PAIRS ----");
  std::vector<Eigen::Isometry3d> obj_samples;
  obj_samples.push_back(f_a3.object);
  const Eigen::Vector3d dual4a_c(han_y["object"]["center_world"]["x"].as<double>(0.0),
                                 han_y["object"]["center_world"]["y"].as<double>(0.4),
                                 han_y["object"]["center_world"]["z"].as<double>(0.9));
  Eigen::Matrix3d R_dual4a;
  R_dual4a << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  Eigen::Isometry3d T_dual4a = Eigen::Isometry3d::Identity();
  T_dual4a.linear() = R_dual4a;
  T_dual4a.translation() = dual4a_c;
  obj_samples.push_back(T_dual4a);
  const double dxs[] = {0.0, 0.04};
  const double dys[] = {0.0, 0.08};
  const double dzs[] = {-0.16, -0.06, 0.0};
  for (double dx : dxs)
    for (double dy : dys)
      for (double dz : dzs)
      {
        Eigen::Isometry3d T = f_a3.object;
        T.translation() += Eigen::Vector3d(dx, dy, dz);
        obj_samples.push_back(T);
      }
  for (double yaw : {-0.4, 0.4, 1.5708})
  {
    Eigen::Isometry3d T = f_a3.object;
    T.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * T.linear();
    obj_samples.push_back(T);
    Eigen::Isometry3d T2 = T_dual4a;
    T2.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * T2.linear();
    obj_samples.push_back(T2);
  }

  IkCfg ik_h = ik;
  ik_h.max_attempts = 48;
  ik_h.max_unique = 10;
  for (size_t si = 0; si < obj_samples.size(); ++si)
  {
    const auto& Tobj = obj_samples[si];
    const Eigen::Isometry3d tA = Tobj * T_tcpA_obj.inverse();
    const Eigen::Isometry3d tB = Tobj * T_tcpB_obj.inverse();
    Eigen::Isometry3d tApre = tA;
    tApre.translation() -= a_retract * (tA.linear() * Eigen::Vector3d::UnitZ());
    Eigen::Isometry3d tBpre = tB;
    tBpre.translation() -= b_retract * (tB.linear() * Eigen::Vector3d::UnitZ());

    auto a_cands =
        searchIk(*scene, kGroupA, kTcpA, tA, home_b, face3, {{"face3", face3}, {"home", home_a}, {"han", han_a}},
                 w, ik_h, PartMode::OnA, T_tcpA_obj, Tobj, radius, height, qa_grasp, qb_open, false,
                 false, true, 1000 + static_cast<unsigned>(si));
    if (a_cands.empty())
      continue;
    // Prefer small J1/J2 from Face3 and Home.
    for (auto& c : a_cands)
    {
      const auto d_home = deltaJoints(home_a, c.joints);
      c.cost = 8.0 * std::abs(c.delta[0]) + 8.0 * std::abs(c.delta[1]) + 1.0 * std::abs(c.delta[2]) +
               1.0 * std::abs(c.delta[3]) + 1.0 * std::abs(c.delta[4]) + 0.15 * std::abs(c.delta[5]) +
               4.0 * std::abs(d_home[0]) + 4.0 * std::abs(d_home[1]);
    }
    std::sort(a_cands.begin(), a_cands.end(),
              [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });

    const auto& a_ref = a_cands.front().joints;
    const ArmGeom ga_ref = geomOf(*scene, true, a_ref);
    const Eigen::Vector3d el_m = mirrorYz(ga_ref.elbow);
    const Eigen::Vector3d dir_m = mirrorYz(ga_ref.upperarm_dir);
    auto geo = mirrorJ12FromA(*scene, model, a_ref);
    if (si == 0)
    {
      emit("  A upperarm_dir=" + fmt3(ga_ref.upperarm_dir) + " elbow=" + fmt3(ga_ref.elbow));
      emit("  mirrored B elbow=" + fmt3(el_m) + " dir=" + fmt3(dir_m));
      for (size_t gi = 0; gi < geo.size(); ++gi)
        emit("  B geometric family" + std::to_string(gi) + " J1=" +
             std::to_string(geo[gi].j1) + " J2=" + std::to_string(geo[gi].j2));
    }
    auto b_seeds = makeBMirrorSeeds(*scene, model, a_ref, {pre_b, home_b, han_b});
    b_seeds.push_back({"home", home_b});
    b_seeds.push_back({"pre", pre_b});
    b_seeds.push_back({"han", han_b});
    auto b_cands =
        searchIk(*scene, kGroupB, kTcpB, tB, a_cands.front().joints, a_ref, b_seeds, w, ik_h,
                 PartMode::OnA, T_tcpB_obj, Tobj, radius, height, qa_grasp, qb_open, false, false,
                 false, 2000 + static_cast<unsigned>(si), nullptr, nullptr, true);
    if (b_cands.empty())
      continue;
    for (auto& c : b_cands)
      c.cost = geometricMirrorCost(*scene, c.joints, el_m, dir_m);
    std::sort(b_cands.begin(), b_cands.end(),
              [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });

    const int na = std::min(3, static_cast<int>(a_cands.size()));
    const int nb = std::min(3, static_cast<int>(b_cands.size()));
    for (int ia = 0; ia < na; ++ia)
    {
      for (int ib = 0; ib < nb; ++ib)
      {
        HandoverPair p;
        p.a = a_cands[ia];
        p.b = b_cands[ib];
        p.world_object = Tobj;
        p.d_a_from_face3 = deltaJoints(face3, p.a.joints);
        p.d_a_to_home = deltaJoints(p.a.joints, home_a);
        std::string pair1, pair2;
        const bool col_a = checkState(*scene, p.a.joints, p.b.joints, qa_grasp, qb_open, PartMode::OnA,
                                      T_tcpA_obj, Tobj, radius, height, false, true, pair1);
        const bool col_b = checkState(*scene, p.a.joints, p.b.joints, qa_open, qb_hold, PartMode::OnB,
                                      T_tcpB_obj, Tobj, radius, height, false, true, pair2);
        p.collision_ok = !col_a && !col_b;
        if (!p.collision_ok)
        {
          p.collision_pair = col_a ? pair1 : pair2;
          continue;
        }
        auto apre =
            searchIk(*scene, kGroupA, kTcpA, tApre, p.b.joints, face3,
                     {{"face3", face3}, {"han", p.a.joints}}, w, ik_h, PartMode::OnA, T_tcpA_obj, Tobj,
                     radius, height, qa_grasp, qb_open, false, false, true, 3000 + static_cast<unsigned>(si));
        auto bpre_seeds = makeBMirrorSeeds(*scene, model, p.a.joints, {p.b.joints, pre_b, home_b});
        bpre_seeds.push_back({"home", home_b});
        bpre_seeds.push_back({"han", p.b.joints});
        bpre_seeds.push_back({"pre", pre_b});
        auto bpre =
            searchIk(*scene, kGroupB, kTcpB, tBpre, p.a.joints, p.a.joints, bpre_seeds, w, ik_h,
                     PartMode::None, T_tcpB_obj, Tobj, radius, height, qa_grasp, qb_open, false, false,
                     false, 4000 + static_cast<unsigned>(si), nullptr, nullptr, true);
        if (apre.empty() || bpre.empty())
          continue;
        for (auto& c : apre)
        {
          c.cost = 8.0 * std::abs(c.joints[0] - p.a.joints[0]) +
                   8.0 * std::abs(c.joints[1] - p.a.joints[1]);
          for (int i = 2; i < 6 && i < static_cast<int>(c.joints.size()); ++i)
            c.cost += (i == 5 ? 0.15 : 1.0) * std::abs(c.joints[i] - p.a.joints[i]);
        }
        std::sort(apre.begin(), apre.end(),
                  [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
        const ArmGeom ga_pre = geomOf(*scene, true, apre.front().joints);
        const Eigen::Vector3d el_m_pre = mirrorYz(ga_pre.elbow);
        const Eigen::Vector3d dir_m_pre = mirrorYz(ga_pre.upperarm_dir);
        for (auto& c : bpre)
          c.cost = geometricMirrorCost(*scene, c.joints, el_m_pre, dir_m_pre);
        std::sort(bpre.begin(), bpre.end(),
                  [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
        p.a_pre = apre.front();
        p.b_pre = bpre.front();
        p.d_b_from_pre = deltaJoints(p.b_pre.joints, p.b.joints);
        p.cost = 8.0 * std::abs(p.d_a_from_face3[0]) + 8.0 * std::abs(p.d_a_from_face3[1]) +
                 4.0 * std::abs(p.d_a_to_home[0]) + 4.0 * std::abs(p.d_a_to_home[1]) +
                 1.0 * std::abs(p.d_a_from_face3[2]) + 1.0 * std::abs(p.d_a_from_face3[3]) +
                 1.0 * std::abs(p.d_a_from_face3[4]) + 0.15 * std::abs(p.d_a_from_face3[5]) +
                 geometricMirrorCost(*scene, p.b.joints, el_m, dir_m) +
                 geometricMirrorCost(*scene, p.b_pre.joints, el_m_pre, dir_m_pre) +
                 3.0 * std::abs(p.d_b_from_pre[0]) + 3.0 * std::abs(p.d_b_from_pre[1]) +
                 0.5 * (std::abs(p.d_b_from_pre[2]) + std::abs(p.d_b_from_pre[3]) +
                        std::abs(p.d_b_from_pre[4])) +
                 0.1 * std::abs(p.d_b_from_pre[5]);
        p.note = "shared T_world_object; B J1/J2 from YZ-mirror of A elbow in B base";
        pairs.push_back(std::move(p));
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const HandoverPair& a, const HandoverPair& b) { return a.cost < b.cost; });
  if (static_cast<int>(pairs.size()) > ik.handover_top_n)
    pairs.resize(ik.handover_top_n);
  for (size_t i = 0; i < pairs.size(); ++i)
    pairs[i].index = static_cast<int>(i) + 1;
  emit("  handover pairs kept=" + std::to_string(pairs.size()));
  if (pairs.empty())
    emit("  NO LEGAL HANDOVER PAIR");
  else
  {
    const ArmGeom ga_p = geomOf(*scene, true, pairs.front().a_pre.joints);
    const ArmGeom gb_p = geomOf(*scene, false, pairs.front().b_pre.joints);
    const ArmGeom ga_h = geomOf(*scene, true, pairs.front().a.joints);
    const ArmGeom gb_h = geomOf(*scene, false, pairs.front().b.joints);
    emit("  pair1 A_PRE upperarm=" + fmt3(ga_p.upperarm_dir) + " elbow=" + fmt3(ga_p.elbow));
    emit("  pair1 B_PRE upperarm=" + fmt3(gb_p.upperarm_dir) + " elbow=" + fmt3(gb_p.elbow) +
         " vs mirrored=" + fmt3(mirrorYz(ga_p.elbow)));
    emit("  pair1 A_HAN upperarm=" + fmt3(ga_h.upperarm_dir) + " elbow=" + fmt3(ga_h.elbow));
    emit("  pair1 B_HAN upperarm=" + fmt3(gb_h.upperarm_dir) + " elbow=" + fmt3(gb_h.elbow) +
         " vs mirrored=" + fmt3(mirrorYz(ga_h.elbow)));
    emit("  pair1 B_PRE joints=" + fmt(pairs.front().b_pre.joints, 4));
    emit("  pair1 B_HAN joints=" + fmt(pairs.front().b.joints, 4));
  }

  std::vector<double> b_pre_ref = pre_b;
  std::vector<double> b_han_ref = han_b;
  if (!pairs.empty())
  {
    b_pre_ref = pairs.front().b_pre.joints;
    b_han_ref = pairs.front().b.joints;
    std::vector<Candidate> apre_list, ahan_list, bpre_list, bhan_list;
    for (auto& p : pairs)
    {
      p.a_pre.index = static_cast<int>(apre_list.size()) + 1;
      p.a.index = static_cast<int>(ahan_list.size()) + 1;
      p.b_pre.index = static_cast<int>(bpre_list.size()) + 1;
      p.b.index = static_cast<int>(bhan_list.size()) + 1;
      p.a_pre.delta = p.d_a_from_face3;
      p.a.delta = p.d_a_from_face3;
      p.b_pre.delta = deltaJoints(home_b, p.b_pre.joints);
      p.b.delta = p.d_b_from_pre;
      apre_list.push_back(p.a_pre);
      ahan_list.push_back(p.a);
      bpre_list.push_back(p.b_pre);
      bhan_list.push_back(p.b);
    }
    arm_a["A_PRE_HANDOVER"] = topN(apre_list, ik.top_n);
    arm_a["A_HANDOVER"] = topN(ahan_list, ik.top_n);
    arm_b["B_PRE_HANDOVER"] = topN(bpre_list, ik.top_n);
    arm_b["B_HANDOVER"] = topN(bhan_list, ik.top_n);
  }
  else
  {
    emit("  fallback: searching original DUAL-4A handover TCP independently");
    const Eigen::Isometry3d tA = T_dual4a * T_tcpA_obj.inverse();
    const Eigen::Isometry3d tB = T_dual4a * T_tcpB_obj.inverse();
    Eigen::Isometry3d tBpre = tB;
    tBpre.translation() -= b_retract * (tB.linear() * Eigen::Vector3d::UnitZ());
    runA("A_HANDOVER", tA, face3, han_a, w, PartMode::OnA, T_dual4a, false, false, nullptr,
         {{"face3", face3}});
    runA("A_PRE_HANDOVER", tA, face3, face3, w, PartMode::OnA, T_dual4a, false, false, nullptr,
         {{"face3", face3}});
    runB("B_HANDOVER", tB, home_b, han_b, w, PartMode::OnA, T_dual4a, false, nullptr,
         {{"han", han_b}});
    runB("B_PRE_HANDOVER", tBpre, home_b, pre_b, w, PartMode::None, T_dual4a, false, nullptr, {});
  }

  std::vector<double> i1, i2, i3;
  yamlVec(six["i1_b"], i1);
  yamlVec(six["i2_b"], i2);
  yamlVec(six["i3_b"], i3);

  std::vector<double> a_pre_ref = home_a;
  if (arm_a.count("A_PRE_HANDOVER") && !arm_a["A_PRE_HANDOVER"].empty())
    a_pre_ref = arm_a["A_PRE_HANDOVER"].front().joints;
  auto b_face_seeds = makeBMirrorSeeds(*scene, model, a_pre_ref, {b_han_ref, home_b, pre_b});
  b_face_seeds.push_back({"han", b_han_ref});
  if (!i1.empty())
    b_face_seeds.push_back({"dual7_i1", i1});
  if (!i2.empty())
    b_face_seeds.push_back({"dual7_i2", i2});
  if (!i3.empty())
    b_face_seeds.push_back({"dual7_i3", i3});

  auto dumpFaceDiag = [&](const std::string& name, const Eigen::Isometry3d& target, const FaceDef& face,
                          const std::vector<double>& seed_q) {
    std::ofstream os(out_dir + "/b_face_reject_" + name + ".yaml");
    os << "task_version: KEYPOSE_OPTIMIZATION_V1\n";
    os << "keypose: " << name << "\n";
    os << "note: |\n";
    os << "  NO LEGAL CANDIDATE means IK returned solutions, but none passed FK/limits/collision\n";
    os << "  plus complete-pose (center+normal+up) under the local table/column/attached-part scene.\n";
    os << "  This is not 'the robot cannot physically reach a nearby posture'.\n";
    writePose(os, "", "canonical_tcp", target);
    writePose(os, "", "canonical_object", face.object);
    os << "canonical_roll_deg: 0.0\n";
    os << "dual7_roll_deg: " << six["i1_roll_deg"].as<double>(-150.0) << "\n";
    if (!seed_q.empty())
    {
      moveit::core::RobotState st(scene->getCurrentState());
      setJoints(st, kArmB, seed_q);
      setJoints(st, kArmA, home_a);
      st.update();
      const Eigen::Isometry3d fk = st.getGlobalLinkTransform(kTcpB);
      double pos = 0, ori = 0;
      poseError(target, fk, pos, ori);
      writeVec(os, "", "dual7_joints", seed_q);
      writePose(os, "", "dual7_fk_tcp", fk);
      os << "dual7_vs_canonical_pos_err_m: " << pos << "\n";
      os << "dual7_vs_canonical_ori_err_deg: " << ori << "\n";
    }
    auto runMode = [&](const std::string& tag, PartMode mode) {
      IkRejectStats stt;
      std::vector<Candidate> rejected;
      auto found =
          searchIk(*scene, kGroupB, kTcpB, target, home_a, seed_q.empty() ? home_b : seed_q,
                   b_face_seeds, w, ik, mode, T_tcpB_obj, face.object, radius, height, qa_open,
                   (mode == PartMode::OnB) ? qb_hold : qb_open, false, false, false,
                   9000u + static_cast<unsigned>(tag.size()), &stt, &rejected);
      os << tag << ":\n";
      os << "  unique_valid: " << found.size() << "\n";
      os << "  reject_fk: " << stt.fk << "\n";
      os << "  reject_bounds: " << stt.bounds << "\n";
      os << "  reject_collision: " << stt.collision << "\n";
      os << "  reject_dup: " << stt.dup << "\n";
      os << "  collision_pairs:\n";
      if (stt.pairs.empty())
        os << "    []\n";
      for (const auto& kv : stt.pairs)
        os << "    - {pair: \"" << kv.first << "\", count: " << kv.second << "}\n";
      os << "  rejected_samples:\n";
      if (rejected.empty())
        os << "    []\n";
      for (const auto& c : rejected)
      {
        os << "    - seed: \"" << c.seed << "\"\n";
        writeVec(os, "      ", "joint_values", c.joints);
        os << "      collision_pair: \"" << c.collision_pair << "\"\n";
      }
    };
    runMode("with_part_on_b", PartMode::OnB);
    runMode("without_part", PartMode::None);
  };

  runB("B_FACE4", tcp_b4, b_han_ref, i1.empty() ? home_b : i1, w, PartMode::OnB, f_b4.object, false,
       &f_b4, b_face_seeds);
  runB("B_FACE5", tcp_b5, i1.empty() ? b_han_ref : i1, i2.empty() ? home_b : i2, w, PartMode::OnB,
       f_b5.object, false, &f_b5, b_face_seeds);
  runB("B_FACE6", tcp_b6, i2.empty() ? b_han_ref : i2, i3.empty() ? home_b : i3, w, PartMode::OnB,
       f_b6.object, false, &f_b6, {{"face5_seed", i2.empty() ? home_b : i2}});
  if (arm_b["B_FACE4"].empty())
    dumpFaceDiag("B_FACE4", tcp_b4, f_b4, i1);
  if (arm_b["B_FACE5"].empty())
    dumpFaceDiag("B_FACE5", tcp_b5, f_b5, i2.empty() ? i1 : i2);

  // B_HOME candidates: joint-space, not IK. Do not overwrite dual8_home.yaml.
  emit("---- B_HOME (candidates only) ----");
  {
    std::vector<Candidate> homes;
    std::vector<double> home_like = b_pre_ref.empty() ? pre_b : b_pre_ref;
    auto geo_home = mirrorJ12FromA(*scene, model, home_a);
    const ArmGeom ga_home = geomOf(*scene, true, home_a);
    const Eigen::Vector3d el_m_home = mirrorYz(ga_home.elbow);
    const Eigen::Vector3d dir_m_home = mirrorYz(ga_home.upperarm_dir);
    for (size_t gi = 0; gi < geo_home.size(); ++gi)
      emit("  B_HOME geometric family" + std::to_string(gi) + " J1=" +
           std::to_string(geo_home[gi].j1) + " J2=" + std::to_string(geo_home[gi].j2));
    std::vector<std::vector<double>> samples;
    if (home_like.size() >= 6)
      samples.push_back(home_like);
    for (const auto& sol : geo_home)
    {
      auto q = home_like.size() >= 6 ? home_like : std::vector<double>(6, 0.0);
      q[0] = sol.j1;
      q[1] = sol.j2;
      samples.push_back(clampJoints(model, kArmB, q));
      if (b_pre_ref.size() >= 6)
      {
        auto q2 = b_pre_ref;
        q2[0] = sol.j1;
        q2[1] = sol.j2;
        samples.push_back(clampJoints(model, kArmB, q2));
      }
    }
    std::mt19937 rng(ik.rng_seed + 9);
    std::uniform_real_distribution<double> d(-0.18, 0.18);
    if (!geo_home.empty() && home_like.size() >= 6)
    {
      for (int k = 0; k < 16; ++k)
      {
        auto q = home_like;
        q[0] = geo_home.front().j1;
        q[1] = geo_home.front().j2;
        for (int j = 0; j < 6; ++j)
          q[j] += (j < 2 ? 0.08 : 0.20) * d(rng);
        samples.push_back(clampJoints(model, kArmB, q));
      }
    }
    samples.push_back(home_b);
    for (const auto& q : samples)
    {
      Candidate c;
      c.joints = q;
      c.seed = (jointL2(q, home_b) < 1e-6) ? "dual8_home_existing" : "mirror_j12_from_A_HOME";
      c.baseline = (jointL2(q, home_b) < 1e-6);
      moveit::core::RobotState st(scene->getCurrentState());
      setJoints(st, kArmB, q);
      setJoints(st, kArmA, home_a);
      st.update();
      c.tcp = st.getGlobalLinkTransform(kTcpB);
      c.limits_ok = st.satisfiesBounds(gb);
      c.joint_limit_margin = limitMargin(st, gb);
      c.fk_ok = true;
      c.collision_ok = !checkState(*scene, home_a, q, qa_open, qb_open, PartMode::None, T_tcpB_obj,
                                   world_grasp_obj, radius, height, false, false, c.collision_pair);
      if (!c.limits_ok || !c.collision_ok || c.joint_limit_margin < 0.05)
        continue;
      c.delta = deltaJoints(home_like, q);
      c.cost = geometricMirrorCost(*scene, q, el_m_home, dir_m_home) +
               0.4 * (std::abs(c.delta[2]) + std::abs(c.delta[3]) + std::abs(c.delta[4])) +
               0.1 * std::abs(c.delta[5]) + std::max(0.0, 0.12 - c.joint_limit_margin) * 8.0;
      if (c.baseline)
        c.cost += 25.0;
      if (findDup(homes, q, 0.08) >= 0)
        continue;
      homes.push_back(c);
    }
    std::sort(homes.begin(), homes.end(),
              [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
    arm_b["B_HOME"] = topN(homes, ik.top_n);
    if (arm_b["B_HOME"].empty())
    {
      Candidate c;
      c.index = 1;
      c.joints = home_b;
      c.tcp = fkTcp(false, home_b);
      c.seed = "dual8_home_fallback";
      c.baseline = true;
      c.collision_ok = true;
      c.limits_ok = true;
      arm_b["B_HOME"] = {c};
      emit("  NO NEW B_HOME; keeping existing as fallback candidate");
    }
    else
      emit("  kept " + std::to_string(arm_b["B_HOME"].size()) + " B_HOME candidates");
  }

  auto dumpArm = [&](const std::string& path, const std::map<std::string, std::vector<Candidate>>& m,
                     const std::vector<std::string>& order, bool faces) {
    std::ofstream os(path);
    os << "task_version: KEYPOSE_OPTIMIZATION_V1\n";
    os << "execution: false\n";
    os << "cost_weights: " << fmt({w.w[0], w.w[1], w.w[2], w.w[3], w.w[4], w.w[5]}) << "\n";
    os << "face2_cost_weights: " << fmt({w2.w[0], w2.w[1], w2.w[2], w2.w[3], w2.w[4], w2.w[5]})
       << "\n";
    os << "keyposes:\n";
    for (const auto& name : order)
    {
      os << "  " << name << ":\n";
      auto it = m.find(name);
      if (it == m.end() || it->second.empty())
      {
        os << "    legal: false\n";
        os << "    note: \"NO LEGAL CANDIDATE\"\n";
        os << "    candidates: []\n";
        continue;
      }
      os << "    legal: true\n";
      os << "    candidates:\n";
      const bool face = faces && (name.find("FACE") != std::string::npos);
      for (const auto& c : it->second)
        writeCand(os, c, face);
    }
  };

  {
    std::ofstream os(out_dir + "/task_space_targets.yaml");
    os << "task_version: KEYPOSE_OPTIMIZATION_V1\n";
    os << "frame: world\n";
    os << "note: |\n";
    os << "  Reused existing task-space definitions. Did not rewrite baseline trajectories.\n";
    os << "  Arm A Face1/2/3 object poses are STEP12C A/B/C_bottom object_world (complete pose).\n";
    os << "  Arm B Face4/5/6 object poses are dual_arm_b_inspection_sequence.yaml canonical\n";
    os << "  preferred-roll object_world (roll=0), including center, normal and up.\n";
    os << "face_up_issue: |\n";
    os << "  dual_arm_b_inspection_sequence.yaml already stores up_in_object and full object_world.\n";
    os << "  DUAL-7 real run applied extra D1 rolls i1_roll_deg=i2_roll_deg=-150, which rotates\n";
    os << "  the in-plane direction. C++ validation gated center+normal, not face_up.\n";
    os << "  New B_FACE4/5/6 candidates enforce complete pose: P1, D1, and canonical face_up.\n";
    os << "p1: " << fmt3(p1) << "\n";
    os << "d1: " << fmt3(d1) << "\n";
    os << "up_world: " << fmt3(up_world) << "\n";
    os << "T_tcpA_object:\n";
    writePose(os, "  ", "pose", T_tcpA_obj);
    os << "T_tcpB_object:\n";
    writePose(os, "  ", "pose", T_tcpB_obj);
    os << "cost_weights: {j1: " << w.w[0] << ", j2: " << w.w[1] << ", j3: " << w.w[2]
       << ", j4: " << w.w[3] << ", j5: " << w.w[4] << ", j6: " << w.w[5] << "}\n";
    os << "face2_cost_weights: {j1: " << w2.w[0] << ", j2: " << w2.w[1] << ", j3: " << w2.w[2]
       << ", j4: " << w2.w[3] << ", j5: " << w2.w[4] << ", j6: " << w2.w[5] << "}\n";
    auto dumpT = [&](const std::string& n, const Eigen::Isometry3d& T, const FaceDef* f) {
      os << n << ":\n";
      writePose(os, "  ", "tcp", T);
      if (f)
      {
        os << "  physical_id: \"" << f->physical << "\"\n";
        writePose(os, "  ", "object", f->object);
        writeVec(os, "  ", "center_in_object", {f->c_obj.x(), f->c_obj.y(), f->c_obj.z()});
        writeVec(os, "  ", "normal_in_object", {f->n_obj.x(), f->n_obj.y(), f->n_obj.z()});
        writeVec(os, "  ", "up_in_object", {f->u_obj.x(), f->u_obj.y(), f->u_obj.z()});
        const Eigen::Vector3d fc = f->object * f->c_obj;
        const Eigen::Vector3d fn = (f->object.linear() * f->n_obj).normalized();
        const Eigen::Vector3d fu = (f->object.linear() * f->u_obj).normalized();
        writeVec(os, "  ", "face_center", {fc.x(), fc.y(), fc.z()});
        writeVec(os, "  ", "face_normal", {fn.x(), fn.y(), fn.z()});
        writeVec(os, "  ", "face_up", {fu.x(), fu.y(), fu.z()});
      }
    };
    dumpT("A_HOME", tcp_home_a, nullptr);
    dumpT("A_PREGRASP", tcp_pregrasp, nullptr);
    dumpT("A_GRASP", tcp_grasp, nullptr);
    dumpT("A_LIFT", tcp_lift, nullptr);
    dumpT("A_FACE1", tcp_a1, &f_a1);
    dumpT("A_FACE2", tcp_a2, &f_a2);
    dumpT("A_FACE3", tcp_a3, &f_a3);
    dumpT("B_FACE4", tcp_b4, &f_b4);
    dumpT("B_FACE5", tcp_b5, &f_b5);
    dumpT("B_FACE6", tcp_b6, &f_b6);
  }

  dumpArm(out_dir + "/arm_a_candidates.yaml", arm_a,
          {"A_HOME", "A_PREGRASP", "A_GRASP", "A_LIFT", "A_FACE1", "A_FACE2", "A_FACE3",
           "A_PRE_HANDOVER", "A_HANDOVER"},
          true);
  dumpArm(out_dir + "/arm_b_candidates.yaml", arm_b,
          {"B_HOME", "B_PRE_HANDOVER", "B_HANDOVER", "B_FACE4", "B_FACE5", "B_FACE6"}, true);

  {
    std::ofstream os(out_dir + "/handover_candidates.yaml");
    os << "task_version: KEYPOSE_OPTIMIZATION_V1\n";
    os << "execution: false\n";
    os << "note: A_HANDOVER and B_HANDOVER share T_world_object. TCP poses may differ.\n";
    os << "cost_weights: " << fmt({w.w[0], w.w[1], w.w[2], w.w[3], w.w[4], w.w[5]}) << "\n";
    os << "pairs:\n";
    if (pairs.empty())
      os << "  []\n";
    for (const auto& p : pairs)
    {
      os << "  - index: " << p.index << "\n";
      os << "    cost: " << std::setprecision(9) << p.cost << "\n";
      os << "    collision_check: " << (p.collision_ok ? "PASS" : "FAIL") << "\n";
      os << "    joint_limit_check: "
         << ((p.a.limits_ok && p.b.limits_ok) ? "PASS" : "FAIL") << "\n";
      writePose(os, "    ", "T_world_object", p.world_object);
      writeVec(os, "    ", "A_delta_from_FACE3", p.d_a_from_face3);
      writeVec(os, "    ", "A_delta_to_HOME", p.d_a_to_home);
      writeVec(os, "    ", "B_delta_from_PRE", p.d_b_from_pre);
      writeVec(os, "    ", "A_PRE_HANDOVER_joints", p.a_pre.joints);
      writeVec(os, "    ", "A_HANDOVER_joints", p.a.joints);
      writeVec(os, "    ", "B_PRE_HANDOVER_joints", p.b_pre.joints);
      writeVec(os, "    ", "B_HANDOVER_joints", p.b.joints);
      writePose(os, "    ", "A_HANDOVER_tcp", p.a.tcp);
      writePose(os, "    ", "B_HANDOVER_tcp", p.b.tcp);
      {
        const ArmGeom ga = geomOf(*scene, true, p.a.joints);
        const ArmGeom gb = geomOf(*scene, false, p.b.joints);
        const ArmGeom gap = geomOf(*scene, true, p.a_pre.joints);
        const ArmGeom gbp = geomOf(*scene, false, p.b_pre.joints);
        writeVec(os, "    ", "A_HANDOVER_upperarm_dir",
                 {ga.upperarm_dir.x(), ga.upperarm_dir.y(), ga.upperarm_dir.z()});
        writeVec(os, "    ", "B_HANDOVER_upperarm_dir",
                 {gb.upperarm_dir.x(), gb.upperarm_dir.y(), gb.upperarm_dir.z()});
        writeVec(os, "    ", "A_HANDOVER_elbow", {ga.elbow.x(), ga.elbow.y(), ga.elbow.z()});
        writeVec(os, "    ", "B_HANDOVER_elbow", {gb.elbow.x(), gb.elbow.y(), gb.elbow.z()});
        const Eigen::Vector3d el_m = mirrorYz(ga.elbow);
        writeVec(os, "    ", "B_HANDOVER_elbow_target_yz_mirror", {el_m.x(), el_m.y(), el_m.z()});
        os << "    B_HANDOVER_elbow_mirror_err_m: " << (gb.elbow - el_m).norm() << "\n";
        writeVec(os, "    ", "A_PRE_upperarm_dir",
                 {gap.upperarm_dir.x(), gap.upperarm_dir.y(), gap.upperarm_dir.z()});
        writeVec(os, "    ", "B_PRE_upperarm_dir",
                 {gbp.upperarm_dir.x(), gbp.upperarm_dir.y(), gbp.upperarm_dir.z()});
      }
      os << "    note: \"" << p.note << "\"\n";
    }
  }

  {
    std::ofstream os(out_dir + "/preview_index.yaml");
    os << "task_version: KEYPOSE_OPTIMIZATION_V1\n";
    os << "arm_a: " << out_dir << "/arm_a_candidates.yaml\n";
    os << "arm_b: " << out_dir << "/arm_b_candidates.yaml\n";
    os << "handover: " << out_dir << "/handover_candidates.yaml\n";
    os << "targets: " << out_dir << "/task_space_targets.yaml\n";
    os << "home_a: " << fmt(home_a, 12) << "\n";
    os << "home_b: " << fmt(home_b, 12) << "\n";
    os << "q_a_open: " << qa_open << "\n";
    os << "q_a_grasp: " << qa_grasp << "\n";
    os << "q_b_open: " << qb_open << "\n";
    os << "q_b_hold: " << qb_hold << "\n";
  }

  emit("wrote " + out_dir);
  emit("KEYPOSE SEARCH COMPLETE. Waiting for per-keypose confirmation.");
  rclcpp::shutdown();
  return 0;
}
