// B_FACE4/B_FACE5 root-cause diagnosis only.
// Read-only on URDF/SRDF/Home/trajectories. Does not rewrite candidates or run hardware.
#include <algorithm>
#include <chrono>
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
#include <moveit/robot_model/link_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
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

std::string fmt3(const Eigen::Vector3d& v, int prec = 6)
{
  return fmt({v.x(), v.y(), v.z()}, prec);
}

std::string deg6(const std::vector<double>& q)
{
  std::vector<double> d;
  for (double v : q)
    d.push_back(v * 180.0 / M_PI);
  return fmt(d, 2);
}

Eigen::Quaterniond quatOf(const Eigen::Isometry3d& T)
{
  Eigen::Quaterniond q(T.linear());
  q.normalize();
  return q;
}

std::vector<double> rpyDeg(const Eigen::Isometry3d& T)
{
  const Eigen::Vector3d e = T.linear().eulerAngles(2, 1, 0);  // ZYX
  return {e[2] * 180.0 / M_PI, e[1] * 180.0 / M_PI, e[0] * 180.0 / M_PI};
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
  return j12FromElbowBase(e_b, model, "arm_b_j1", "arm_b_j2");
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

double jointL2(const std::vector<double>& a, const std::vector<double>& b)
{
  double s = 0.0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}

int findDup(const std::vector<std::vector<double>>& qs, const std::vector<double>& q, double min_d)
{
  for (size_t i = 0; i < qs.size(); ++i)
    if (jointL2(qs[i], q) < min_d)
      return static_cast<int>(i);
  return -1;
}

void perturb(moveit::core::RobotState& st, const moveit::core::JointModelGroup* g, double radius,
             std::mt19937& rng)
{
  for (const auto* j : g->getActiveJointModels())
  {
    std::uniform_real_distribution<double> d(-radius, radius);
    double q = st.getVariablePosition(j->getName()) + d(rng);
    const auto& b = j->getVariableBounds();
    if (!b.empty() && b.front().position_bounded_)
      q = std::min(b.front().max_position_, std::max(b.front().min_position_, q));
    st.setVariablePosition(j->getName(), q);
  }
  st.update();
}

struct WorkcellBox
{
  std::string name;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3d dim = Eigen::Vector3d::Ones();
};

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

bool allowedPair(const std::string& a, const std::string& b, bool allow_part_table)
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
  if ((a == "arm_a_base_link" && b == kColumn) || (a == kColumn && b == "arm_a_base_link"))
    return true;
  if ((a == "arm_b_base_link" && b == kColumn) || (a == kColumn && b == "arm_b_base_link"))
    return true;
  return false;
}

struct ContactHit
{
  std::string a;
  std::string b;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  double depth = 0.0;
  bool allowed = false;
};

std::vector<ContactHit> allContacts(planning_scene::PlanningScene& scene, moveit::core::RobotState& st,
                                    bool allow_part_table)
{
  st.update();
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 120;
  req.max_contacts_per_pair = 2;
  scene.checkCollision(req, res, st);
  std::vector<ContactHit> out;
  for (const auto& kv : res.contacts)
  {
    for (const auto& c : kv.second)
    {
      ContactHit h;
      h.a = kv.first.first;
      h.b = kv.first.second;
      h.pos = c.pos;
      h.normal = c.normal;
      h.depth = c.depth;
      h.allowed = allowedPair(h.a, h.b, allow_part_table);
      out.push_back(h);
    }
  }
  return out;
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

enum class PartMode
{
  None,
  OnB
};

void placePart(planning_scene::PlanningScene& scene, PartMode mode, const Eigen::Isometry3d& tcp_obj,
               double r, double h)
{
  if (mode == PartMode::None)
  {
    clearObject(scene);
    return;
  }
  attachCylinder(scene, kTcpB, tcp_obj, kTouchB, r, h);
}

std::vector<ContactHit> checkContacts(planning_scene::PlanningScene& scene,
                                      const std::vector<double>& a, const std::vector<double>& b,
                                      double qa, double qb, PartMode mode,
                                      const Eigen::Isometry3d& tcp_obj, double r, double h)
{
  placePart(scene, mode, tcp_obj, r, h);
  applyState(scene, a, b, qa, qb);
  auto st = scene.getCurrentStateNonConst();
  return allContacts(scene, st, false);
}

bool hasIllegal(const std::vector<ContactHit>& hits, std::string& pair)
{
  for (const auto& h : hits)
  {
    if (h.allowed)
      continue;
    pair = h.a + " <-> " + h.b;
    return true;
  }
  return false;
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
  writeVec(os, indent + "  ", "rpy_zyx_deg", rpyDeg(T));
}

double extraRollDeg(const Eigen::Vector3d& from_up, const Eigen::Vector3d& to_up,
                    const Eigen::Vector3d& d1)
{
  const Eigen::Vector3d n = d1.normalized();
  Eigen::Vector3d a = from_up - n * from_up.dot(n);
  Eigen::Vector3d b = to_up - n * to_up.dot(n);
  if (a.norm() < 1e-9 || b.norm() < 1e-9)
    return 0.0;
  a.normalize();
  b.normalize();
  return std::atan2(n.dot(a.cross(b)), a.dot(b)) * 180.0 / M_PI;
}

Eigen::Isometry3d objectFromFace(const Eigen::Vector3d& p1, const Eigen::Vector3d& n_w,
                                 const Eigen::Vector3d& u_w, const Eigen::Vector3d& c_obj,
                                 const Eigen::Vector3d& n_obj, const Eigen::Vector3d& u_obj)
{
  Eigen::Vector3d nw = n_w.normalized();
  Eigen::Vector3d uw = u_w - nw * u_w.dot(nw);
  if (uw.norm() < 1e-9)
    uw = Eigen::Vector3d::UnitX();
  uw.normalize();
  Eigen::Matrix3d src, dst;
  src.col(0) = n_obj.normalized();
  src.col(1) = u_obj.normalized();
  src.col(2) = src.col(0).cross(src.col(1)).normalized();
  dst.col(0) = nw;
  dst.col(1) = uw;
  dst.col(2) = dst.col(0).cross(dst.col(1)).normalized();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = dst * src.inverse();
  T.translation() = p1 - T.linear() * c_obj;
  return T;
}

Eigen::Isometry3d rolledAboutD1(const Eigen::Isometry3d& canonical, const Eigen::Vector3d& p1,
                               const Eigen::Vector3d& d1, const Eigen::Vector3d& c_obj,
                               double roll_deg)
{
  Eigen::Isometry3d out = canonical;
  const Eigen::Matrix3d r =
      Eigen::AngleAxisd(roll_deg * M_PI / 180.0, d1.normalized()).toRotationMatrix();
  out.linear() = r * canonical.linear();
  out.translation() = p1 - out.linear() * c_obj;
  return out;
}

std::string j2Family(double j2)
{
  if (j2 < -3.4)
    return "folded_down";
  if (j2 < -2.0)
    return "folded";
  if (j2 < 0.0)
    return "mid_neg";
  return "open_pos";
}

std::string j1Bin(double j1)
{
  const int deg = static_cast<int>(std::lround(j1 * 180.0 / M_PI / 30.0) * 30);
  return "j1_" + std::to_string(deg);
}

std::string configSig(const std::vector<double>& q)
{
  if (q.size() < 6)
    return "short";
  std::ostringstream o;
  o << j1Bin(q[0]) << "|" << j2Family(q[1]) << "|elbow" << (q[2] >= 0 ? "+" : "-") << "|j4"
    << (q[3] >= 0 ? "+" : "-") << "|wrist" << (q[4] >= 0 ? "+" : "-");
  return o.str();
}

struct IkHit
{
  std::string seed;
  std::vector<double> joints;
  Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity();
  bool limits_ok = false;
  bool collision_ok = false;
  std::string pair = "none";
  double fk_pos = 0.0;
  double fk_ori = 0.0;
};

struct FaceGeom
{
  std::string name;
  std::string physical;
  Eigen::Vector3d c_obj = Eigen::Vector3d::Zero();
  Eigen::Vector3d n_obj = Eigen::Vector3d::UnitX();
  Eigen::Vector3d u_obj = Eigen::Vector3d::UnitZ();
  Eigen::Isometry3d object = Eigen::Isometry3d::Identity();
  Eigen::Vector3d face_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d face_normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d face_up = Eigen::Vector3d::UnitX();
};

void fillWorldFace(FaceGeom& f)
{
  f.face_center = f.object * f.c_obj;
  f.face_normal = (f.object.linear() * f.n_obj).normalized();
  f.face_up = (f.object.linear() * f.u_obj).normalized();
}

void dumpLink(std::ostream& os, const std::string& indent, const std::string& name,
              moveit::core::RobotState& st)
{
  if (!st.getRobotModel()->hasLinkModel(name))
    return;
  const Eigen::Isometry3d T = st.getGlobalLinkTransform(name);
  writePose(os, indent, name, T);
  const auto* link = st.getRobotModel()->getLinkModel(name);
  if (!link)
    return;
  os << indent << name << "_shape_count: " << link->getShapes().size() << "\n";
  os << indent << name << "_extents: " << fmt3(link->getShapeExtentsAtOrigin()) << "\n";
}

visualization_msgs::msg::Marker cubeMarker(int id, const Eigen::Vector3d& xyz,
                                           const Eigen::Vector3d& dim, float r, float g, float b,
                                           float a, const std::string& ns)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = "world";
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::msg::Marker::CUBE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.position.x = xyz.x();
  m.pose.position.y = xyz.y();
  m.pose.position.z = xyz.z();
  m.pose.orientation.w = 1.0;
  m.scale.x = dim.x();
  m.scale.y = dim.y();
  m.scale.z = dim.z();
  m.color.r = r;
  m.color.g = g;
  m.color.b = b;
  m.color.a = a;
  return m;
}

visualization_msgs::msg::Marker arrowMarker(int id, const Eigen::Vector3d& p, const Eigen::Vector3d& v,
                                            float r, float g, float b, const std::string& ns)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = "world";
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::msg::Marker::ARROW;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = 0.008;
  m.scale.y = 0.014;
  m.scale.z = 0.018;
  m.color.r = r;
  m.color.g = g;
  m.color.b = b;
  m.color.a = 1.0;
  geometry_msgs::msg::Point s, e;
  s.x = p.x();
  s.y = p.y();
  s.z = p.z();
  e.x = p.x() + 0.10 * v.x();
  e.y = p.y() + 0.10 * v.y();
  e.z = p.z() + 0.10 * v.z();
  m.points = {s, e};
  return m;
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("keypose_b_face_diag", opt);

  emit("========== B_FACE4/5 ROOT CAUSE DIAG ==========");
  emit("NO EXECUTION. NO PATH PLANNING. NO CANDIDATE REWRITE.");

  const std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string search_yaml = node->get_parameter_or(
      "search_yaml", home_dir + "/fr_task_ws/src/fr_task_planner/config/"
                                "keypose_optimization_v1/search.yaml");
  const bool hold = node->get_parameter_or("hold", false);
  YAML::Node cfg = YAML::LoadFile(search_yaml);
  const std::string out_dir = node->get_parameter_or("output_dir", cfg["output_dir"].as<std::string>());

  YAML::Node home_y = YAML::LoadFile(cfg["home_yaml"].as<std::string>());
  YAML::Node s12 = YAML::LoadFile(cfg["step12c_winner"].as<std::string>());
  YAML::Node bseq = YAML::LoadFile(cfg["b_inspection_yaml"].as<std::string>());
  YAML::Node six = YAML::LoadFile(cfg["six_face_yaml"].as<std::string>());
  YAML::Node work = YAML::LoadFile(cfg["workcell_yaml"].as<std::string>());
  YAML::Node han_y = YAML::LoadFile(out_dir + "/handover_candidates.yaml");
  YAML::Node arm_a_y = YAML::LoadFile(out_dir + "/arm_a_candidates.yaml");
  YAML::Node arm_b_y = YAML::LoadFile(out_dir + "/arm_b_candidates.yaml");
  YAML::Node tgt_y = YAML::LoadFile(out_dir + "/task_space_targets.yaml");

  std::vector<double> home_a, home_b, pre_b, han_b, i1, i2, i3, a_pre, a_face1, a_face2, b_face6;
  yamlVec(home_y["home_a"], home_a);
  yamlVec(home_y["home_b"], home_b);
  yamlVec(home_y["pre_b"], pre_b);
  yamlVec(six["handover_b"], han_b);
  yamlVec(six["i1_b"], i1);
  yamlVec(six["i2_b"], i2);
  yamlVec(six["i3_b"], i3);
  if (han_y["pairs"] && han_y["pairs"][0])
    yamlVec(han_y["pairs"][0]["A_PRE_HANDOVER_joints"], a_pre);
  if (arm_a_y["keyposes"]["A_FACE1"]["candidates"][0])
    yamlVec(arm_a_y["keyposes"]["A_FACE1"]["candidates"][0]["joint_values"], a_face1);
  if (arm_a_y["keyposes"]["A_FACE2"]["candidates"][0])
    yamlVec(arm_a_y["keyposes"]["A_FACE2"]["candidates"][0]["joint_values"], a_face2);
  if (arm_b_y["keyposes"]["B_FACE6"]["candidates"][0])
    yamlVec(arm_b_y["keyposes"]["B_FACE6"]["candidates"][0]["joint_values"], b_face6);
  if (han_y["pairs"] && han_y["pairs"][0] && han_b.empty())
    yamlVec(han_y["pairs"][0]["B_HANDOVER_joints"], han_b);

  const double qa_open = cfg["gripper"]["q_a_open"].as<double>(0.0);
  const double qb_hold = cfg["gripper"]["q_b_hold"].as<double>(0.083);
  const double radius = cfg["object"]["radius_m"].as<double>(0.0075);
  const double height = cfg["object"]["height_m"].as<double>(0.035);
  const double dual7_roll = six["i1_roll_deg"].as<double>(-150.0);

  Eigen::Isometry3d T_tcpA_obj = Eigen::Isometry3d::Identity();
  T_tcpA_obj.linear() = Eigen::Quaterniond(0.0, 1.0, 0.0, 0.0).toRotationMatrix();
  Eigen::Isometry3d T_tcpB_obj = Eigen::Isometry3d::Identity();
  T_tcpB_obj.translation() = Eigen::Vector3d(0, 0, 0.008);
  T_tcpB_obj.linear() = Eigen::Quaterniond(0.707106343559, 0.0, 0.0, -0.707107218813).toRotationMatrix();
  loadXyzw(tgt_y["T_tcpA_object"]["pose"], T_tcpA_obj);
  loadXyzw(tgt_y["T_tcpB_object"]["pose"], T_tcpB_obj);

  const Eigen::Vector3d p1(s12["p1"][0].as<double>(), s12["p1"][1].as<double>(),
                           s12["p1"][2].as<double>());
  const Eigen::Vector3d d1(s12["surface_target_normal"][0].as<double>(),
                           s12["surface_target_normal"][1].as<double>(),
                           s12["surface_target_normal"][2].as<double>());
  const Eigen::Vector3d up_world(0.0, 0.707106781, 0.707106781);

  auto loadFace = [&](const YAML::Node& n, const std::string& name) {
    FaceGeom f;
    f.name = name;
    f.physical = n["physical_id"] ? n["physical_id"].as<std::string>() : "";
    yamlVec3(n["center_in_object"], f.c_obj);
    yamlVec3(n["normal_in_object"], f.n_obj);
    yamlVec3(n["up_in_object"], f.u_obj);
    loadXyzw(n["object_world"] ? n["object_world"] : n["object"], f.object);
    fillWorldFace(f);
    return f;
  };
  FaceGeom f_a1 = loadFace(tgt_y["A_FACE1"], "A_FACE1");
  FaceGeom f_a2 = loadFace(tgt_y["A_FACE2"], "A_FACE2");
  FaceGeom f_b4 = loadFace(tgt_y["B_FACE4"], "B_FACE4");
  FaceGeom f_b5 = loadFace(tgt_y["B_FACE5"], "B_FACE5");
  (void)bseq;

  robot_model_loader::RobotModelLoader loader(node);
  auto model = loader.getModel();
  if (!model || model->getName() != "fairino3_dual_robot")
  {
    emit("FAIL: expected fairino3_dual_robot");
    rclcpp::shutdown();
    return 1;
  }
  auto* gb = model->getJointModelGroup(kGroupB);
  if (!gb || !gb->getSolverInstance() || !gb->canSetStateFromIK(kTcpB))
  {
    emit("FAIL: arm_b IK missing");
    rclcpp::shutdown();
    return 1;
  }

  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  const WorkcellBox table = loadBox(work["table"], kTable);
  const WorkcellBox column = loadBox(work["column"], kColumn);
  addBox(*scene, table);
  addBox(*scene, column);
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

  auto fkB = [&](const std::vector<double>& q) {
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, kArmA, home_a);
    setJoints(st, kArmB, q);
    st.update();
    return st;
  };

  const Eigen::Isometry3d tcp_b4 = f_b4.object * T_tcpB_obj.inverse();
  const Eigen::Isometry3d tcp_b5 = f_b5.object * T_tcpB_obj.inverse();

  // Correct YZ-mirror of A inspection (center, normal, in-plane), then B grasp transform.
  const Eigen::Vector3d up_m4 = mirrorYz(f_a1.face_up);
  const Eigen::Vector3d up_m5 = mirrorYz(f_a2.face_up);
  const Eigen::Vector3d n_m4 = mirrorYz(f_a1.face_normal).normalized();
  const Eigen::Vector3d n_m5 = mirrorYz(f_a2.face_normal).normalized();
  const Eigen::Vector3d c_m4 = mirrorYz(f_a1.face_center);
  const Eigen::Vector3d c_m5 = mirrorYz(f_a2.face_center);
  const Eigen::Isometry3d obj_m4 =
      objectFromFace(c_m4, n_m4, up_m4, f_b4.c_obj, f_b4.n_obj, f_b4.u_obj);
  const Eigen::Isometry3d obj_m5 =
      objectFromFace(c_m5, n_m5, up_m5, f_b5.c_obj, f_b5.n_obj, f_b5.u_obj);
  const Eigen::Isometry3d tcp_m4 = obj_m4 * T_tcpB_obj.inverse();
  const Eigen::Isometry3d tcp_m5 = obj_m5 * T_tcpB_obj.inverse();
  const double roll_m4 = extraRollDeg(f_b4.face_up, up_m4, d1);
  const double roll_m5 = extraRollDeg(f_b5.face_up, up_m5, d1);
  const Eigen::Isometry3d obj_m4_roll = rolledAboutD1(f_b4.object, p1, d1, f_b4.c_obj, roll_m4);
  const Eigen::Isometry3d obj_m5_roll = rolledAboutD1(f_b5.object, p1, d1, f_b5.c_obj, roll_m5);
  const Eigen::Isometry3d tcp_m4_roll = obj_m4_roll * T_tcpB_obj.inverse();
  const Eigen::Isometry3d tcp_m5_roll = obj_m5_roll * T_tcpB_obj.inverse();

  moveit::core::RobotState st_i1 = fkB(i1);
  moveit::core::RobotState st_i2 = fkB(i2);
  const Eigen::Isometry3d tcp_d7_4 = st_i1.getGlobalLinkTransform(kTcpB);
  const Eigen::Isometry3d tcp_d7_5 = st_i2.getGlobalLinkTransform(kTcpB);
  const Eigen::Isometry3d obj_d7_4 = tcp_d7_4 * T_tcpB_obj;
  const Eigen::Isometry3d obj_d7_5 = tcp_d7_5 * T_tcpB_obj;
  const Eigen::Vector3d d7_c4 = obj_d7_4 * f_b4.c_obj;
  const Eigen::Vector3d d7_n4 = (obj_d7_4.linear() * f_b4.n_obj).normalized();
  const Eigen::Vector3d d7_u4 = (obj_d7_4.linear() * f_b4.u_obj).normalized();
  const Eigen::Vector3d d7_c5 = obj_d7_5 * f_b5.c_obj;
  const Eigen::Vector3d d7_n5 = (obj_d7_5.linear() * f_b5.n_obj).normalized();
  const Eigen::Vector3d d7_u5 = (obj_d7_5.linear() * f_b5.u_obj).normalized();
  const double d7_roll4 = extraRollDeg(f_b4.face_up, d7_u4, d1);
  const double d7_roll5 = extraRollDeg(f_b5.face_up, d7_u5, d1);

  double p_m4 = 0, o_m4 = 0, p_m5 = 0, o_m5 = 0;
  double p_d7c4 = 0, o_d7c4 = 0, p_d7c5 = 0, o_d7c5 = 0;
  double p_d7m4 = 0, o_d7m4 = 0, p_d7m5 = 0, o_d7m5 = 0;
  double p_cm4 = 0, o_cm4 = 0, p_cm5 = 0, o_cm5 = 0;
  poseError(tcp_b4, tcp_m4, p_cm4, o_cm4);
  poseError(tcp_b5, tcp_m5, p_cm5, o_cm5);
  poseError(tcp_d7_4, tcp_b4, p_d7c4, o_d7c4);
  poseError(tcp_d7_5, tcp_b5, p_d7c5, o_d7c5);
  poseError(tcp_d7_4, tcp_m4, p_d7m4, o_d7m4);
  poseError(tcp_d7_5, tcp_m5, p_d7m5, o_d7m5);
  poseError(tcp_m4, tcp_m4_roll, p_m4, o_m4);
  poseError(tcp_m5, tcp_m5_roll, p_m5, o_m5);

  emit("--- 1. target pose ---");
  emit("A_FACE1 center " + fmt3(f_a1.face_center) + " n " + fmt3(f_a1.face_normal) + " up " +
       fmt3(f_a1.face_up));
  emit("A_FACE2 center " + fmt3(f_a2.face_center) + " n " + fmt3(f_a2.face_normal) + " up " +
       fmt3(f_a2.face_up));
  emit("B_FACE4 center " + fmt3(f_b4.face_center) + " n " + fmt3(f_b4.face_normal) + " up " +
       fmt3(f_b4.face_up));
  emit("B_FACE5 center " + fmt3(f_b5.face_center) + " n " + fmt3(f_b5.face_normal) + " up " +
       fmt3(f_b5.face_up));
  emit("canonical roll=0 source: dual_arm_b_inspection_sequence.yaml preferred-roll "
       "compute_view_target(inspection_up=up_world), NOT B TCP roll=0");
  emit("A_FACE1 extra-roll vs preferred-up " +
       std::to_string(extraRollDeg(up_world, f_a1.face_up, d1)) + " deg");
  emit("B_FACE4 extra-roll vs preferred-up " +
       std::to_string(extraRollDeg(up_world, f_b4.face_up, d1)) + " deg");
  emit("mirror-of-A_FACE1 extra-roll vs canonical B_FACE4 " + std::to_string(roll_m4) + " deg");
  emit("DUAL-7 i1 extra-roll vs canonical B_FACE4 " + std::to_string(d7_roll4) + " deg");

  emit("--- 2. mirror TCP error ---");
  emit("symmetry plane: world YZ (x -> -x)");
  emit("canonical vs correct-mirror FACE4 pos=" + std::to_string(p_cm4) +
       " ori_deg=" + std::to_string(o_cm4));
  emit("canonical vs correct-mirror FACE5 pos=" + std::to_string(p_cm5) +
       " ori_deg=" + std::to_string(o_cm5));
  emit("DUAL-7 vs canonical FACE4 pos=" + std::to_string(p_d7c4) +
       " ori_deg=" + std::to_string(o_d7c4));
  emit("DUAL-7 vs correct-mirror FACE4 pos=" + std::to_string(p_d7m4) +
       " ori_deg=" + std::to_string(o_d7m4));
  emit("objectFromFace vs rolledAboutD1 FACE4 pos=" + std::to_string(p_m4) +
       " ori_deg=" + std::to_string(o_m4));

  auto collectIk = [&](const Eigen::Isometry3d& target,
                       const std::vector<std::pair<std::string, std::vector<double>>>& seeds,
                       int nearby_n, unsigned rng_off, int max_att) {
    std::mt19937 rng(42u + rng_off);
    std::vector<IkHit> all;
    std::vector<std::vector<double>> unique_q;
    int attempts = 0;
    int success = 0;
    for (const auto& seed : seeds)
    {
      if (attempts >= max_att)
        break;
      moveit::core::RobotState exact(scene->getCurrentState());
      setJoints(exact, kArmA, home_a);
      setJoints(exact, kArmB, seed.second);
      exact.update();
      auto tryOnce = [&](moveit::core::RobotState& st, const std::string& sname, double timeout) {
        if (attempts >= max_att)
          return;
        ++attempts;
        if (!st.setFromIK(gb, target, kTcpB, timeout))
          return;
        ++success;
        IkHit h;
        h.seed = sname;
        h.joints = jointsOf(st, kArmB);
        h.tcp = st.getGlobalLinkTransform(kTcpB);
        poseError(target, h.tcp, h.fk_pos, h.fk_ori);
        h.limits_ok = st.satisfiesBounds(gb);
        if (h.fk_pos > 0.003 || h.fk_ori > 2.0 || !h.limits_ok)
          return;
        if (findDup(unique_q, h.joints, 0.15) >= 0)
        {
          all.push_back(h);
          return;
        }
        auto hits = checkContacts(*scene, home_a, h.joints, qa_open, qb_hold, PartMode::OnB,
                                  T_tcpB_obj, radius, height);
        h.collision_ok = !hasIllegal(hits, h.pair);
        unique_q.push_back(h.joints);
        all.push_back(h);
      };
      tryOnce(exact, seed.first + "_exact", 0.20);
      for (int i = 0; i < nearby_n; ++i)
      {
        moveit::core::RobotState nearby(exact);
        perturb(nearby, gb, i < nearby_n / 2 ? 0.6 : 1.6, rng);
        tryOnce(nearby, seed.first + "_nearby", 0.08);
      }
    }
    emit("  IK attempts=" + std::to_string(attempts) + " success=" + std::to_string(success) +
         " unique_fk=" + std::to_string(unique_q.size()));
    struct Pack
    {
      int attempts = 0;
      int success = 0;
      std::vector<IkHit> unique;
    };
    Pack p;
    p.attempts = attempts;
    p.success = success;
    for (const auto& h : all)
    {
      bool seen = false;
      for (const auto& u : p.unique)
        if (jointL2(u.joints, h.joints) < 0.15)
          seen = true;
      if (!seen && h.limits_ok && h.fk_pos <= 0.003)
        p.unique.push_back(h);
    }
    return p;
  };

  std::vector<std::pair<std::string, std::vector<double>>> orig_seeds =
      makeBMirrorSeeds(*scene, model, a_pre.empty() ? home_a : a_pre, {han_b, home_b, pre_b});
  orig_seeds.push_back({"han", han_b});
  orig_seeds.push_back({"dual7_i1", i1});
  orig_seeds.push_back({"dual7_i2", i2});
  orig_seeds.push_back({"dual7_i3", i3});
  orig_seeds.push_back({"home", home_b});
  orig_seeds.push_back({"pre", pre_b});

  emit("--- 3. IK coverage (canonical FACE4, original-style seeds, keep collision) ---");
  auto cov4 = collectIk(tcp_b4, orig_seeds, 6, 1, 96);
  std::map<std::string, int> groups4;
  int legal4 = 0;
  for (const auto& h : cov4.unique)
  {
    groups4[configSig(h.joints)]++;
    if (h.collision_ok)
      ++legal4;
  }
  emit("  unique configs=" + std::to_string(cov4.unique.size()) +
       " collision_free=" + std::to_string(legal4) + " groups=" + std::to_string(groups4.size()));
  for (const auto& kv : groups4)
    emit("    " + kv.first + " n=" + std::to_string(kv.second));

  emit("--- 3b. IK coverage canonical FACE5 ---");
  auto cov5 = collectIk(tcp_b5, orig_seeds, 6, 2, 96);
  std::map<std::string, int> groups5;
  int legal5 = 0;
  for (const auto& h : cov5.unique)
  {
    groups5[configSig(h.joints)]++;
    if (h.collision_ok)
      ++legal5;
  }
  emit("  unique configs=" + std::to_string(cov5.unique.size()) +
       " collision_free=" + std::to_string(legal5) + " groups=" + std::to_string(groups5.size()));

  std::vector<std::pair<std::string, std::vector<double>>> targeted;
  targeted.push_back({"dual7_i1", i1});
  targeted.push_back({"dual7_i2", i2});
  targeted.push_back({"face6", b_face6.empty() ? i3 : b_face6});
  auto a1_mirrors = makeBMirrorSeeds(*scene, model, a_face1, {i1, b_face6.empty() ? i3 : b_face6, han_b});
  for (const auto& s : a1_mirrors)
    targeted.push_back(s);
  auto a2_mirrors = makeBMirrorSeeds(*scene, model, a_face2, {i2, b_face6.empty() ? i3 : b_face6});
  for (const auto& s : a2_mirrors)
    targeted.push_back({"a2_" + s.first, s.second});

  emit("--- 3c. targeted seeds (few attempts, limits+collision kept) ---");
  struct TRow
  {
    std::string label;
    std::string seed;
    bool ik = false;
    bool col_ok = false;
    std::string pair;
    std::vector<double> q;
  };
  std::vector<TRow> trows;
  auto oneSeed = [&](const std::string& label, const Eigen::Isometry3d& target,
                     const std::string& sname, const std::vector<double>& q0) {
    TRow row;
    row.label = label;
    row.seed = sname;
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, kArmA, home_a);
    setJoints(st, kArmB, q0);
    st.update();
    if (!st.setFromIK(gb, target, kTcpB, 0.25))
    {
      trows.push_back(row);
      emit("  " + label + " " + sname + " IK=FAIL");
      return;
    }
    row.ik = true;
    row.q = jointsOf(st, kArmB);
    if (!st.satisfiesBounds(gb))
    {
      row.pair = "bounds";
      trows.push_back(row);
      emit("  " + label + " " + sname + " IK=ok bounds=FAIL");
      return;
    }
    auto hits = checkContacts(*scene, home_a, row.q, qa_open, qb_hold, PartMode::OnB, T_tcpB_obj,
                              radius, height);
    row.col_ok = !hasIllegal(hits, row.pair);
    trows.push_back(row);
    emit("  " + label + " " + sname + " IK=ok col=" + (row.col_ok ? "PASS" : row.pair) +
         " qdeg=" + deg6(row.q) + " sig=" + configSig(row.q));
  };
  for (const auto& s : targeted)
  {
    oneSeed("FACE4_canonical", tcp_b4, s.first, s.second);
    oneSeed("FACE4_mirror", tcp_m4, s.first, s.second);
    oneSeed("FACE4_dual7tcp", tcp_d7_4, s.first, s.second);
  }
  for (const auto& s : targeted)
  {
    oneSeed("FACE5_canonical", tcp_b5, s.first, s.second);
    oneSeed("FACE5_mirror", tcp_m5, s.first, s.second);
    oneSeed("FACE5_dual7tcp", tcp_d7_5, s.first, s.second);
  }

  emit("--- 4. DUAL-7 static collision, Arm A = Home ---");
  auto reportCol = [&](const std::string& tag, const std::vector<double>& qb, PartMode mode) {
    auto hits = checkContacts(*scene, home_a, qb, qa_open, mode == PartMode::OnB ? qb_hold : 0.0,
                              mode, T_tcpB_obj, radius, height);
    std::string pair;
    const bool bad = hasIllegal(hits, pair);
    emit("  " + tag + (bad ? " COLLIDE " + pair : " NO_ILLEGAL_COLLISION") +
         " contacts=" + std::to_string(hits.size()));
    std::map<std::string, int> pairs;
    for (const auto& h : hits)
    {
      const std::string k = h.a + " <-> " + h.b + (h.allowed ? " [ACM]" : "");
      pairs[k]++;
      if (!h.allowed)
        emit("    ILLEGAL " + h.a + " <-> " + h.b + " pos=" + fmt3(h.pos) +
             " depth=" + std::to_string(h.depth));
    }
    return hits;
  };
  auto hits_i1_part = reportCol("DUAL7_I1 with_part A=Home", i1, PartMode::OnB);
  auto hits_i1_none = reportCol("DUAL7_I1 no_part A=Home", i1, PartMode::None);
  auto hits_i2_part = reportCol("DUAL7_I2 with_part A=Home", i2, PartMode::OnB);
  auto hits_i2_none = reportCol("DUAL7_I2 no_part A=Home", i2, PartMode::None);
  auto hits_f6 = reportCol("B_FACE6 legal with_part A=Home", b_face6.empty() ? i3 : b_face6,
                           PartMode::OnB);

  const Eigen::Isometry3d Ta_base = st_i1.getGlobalLinkTransform("arm_a_base_link");
  const Eigen::Isometry3d Tb_base = st_i1.getGlobalLinkTransform("arm_b_base_link");
  const Eigen::Isometry3d Tb_fore = st_i1.getGlobalLinkTransform("arm_b_forearm_link");
  const Eigen::Isometry3d Tb_up = st_i1.getGlobalLinkTransform("arm_b_upperarm_link");
  emit("  column xyz=" + fmt3(column.xyz) + " dim=" + fmt3(column.dim));
  emit("  arm_a_base " + fmt3(Ta_base.translation()));
  emit("  arm_b_base " + fmt3(Tb_base.translation()));
  emit("  arm_b_forearm origin " + fmt3(Tb_fore.translation()));
  emit("  arm_b_upperarm origin " + fmt3(Tb_up.translation()));
  const Eigen::Vector3d col_min = column.xyz - 0.5 * column.dim;
  const Eigen::Vector3d col_max = column.xyz + 0.5 * column.dim;
  const Eigen::Vector3d fo = Tb_fore.translation();
  Eigen::Vector3d closest = fo.cwiseMax(col_min).cwiseMin(col_max);
  emit("  forearm_origin to column AABB dist " + std::to_string((fo - closest).norm()) +
       " closest " + fmt3(closest));
  emit("  A Home during B inspect: " + fmt(home_a, 6) + " deg " + deg6(home_a));
  (void)tcp_m4_roll;
  (void)tcp_m5_roll;

  std::string dummy;
  const bool i1_bad = hasIllegal(hits_i1_part, dummy);
  const bool i2_bad = hasIllegal(hits_i2_part, dummy);

  const std::string out_path = out_dir + "/b_face45_root_cause.yaml";
  std::ofstream os(out_path);
  os << "task_version: KEYPOSE_B_FACE45_DIAG\n";
  os << "execution: false\n";
  os << "note: |\n";
  os << "  Independent B_FACE4/5 diagnosis. Did not rewrite candidates, ACM, or collision geometry.\n";
  os << "symmetry_plane: world_YZ  # x -> -x\n";
  os << "canonical_roll0_source: |\n";
  os << "  dual_arm_b_inspection_sequence.yaml roll_deg=0 is preferred-roll object pose from\n";
  os << "  compute_view_target(normal->D1, up_in_object->inspection_up=up_world).\n";
  os << "  It is NOT 'B TCP joint-6 / TCP roll = 0'. FACE4/5 face_up matches up_world in the YZ plane.\n";
  os << "  A_FACE1 extra-roll from that preferred up is about -90 deg (face_up = +X).\n";
  os << "  Correct B_FACE4 mirror extra-roll is about " << roll_m4
     << " deg (face_up = -X), not DUAL-7 -150.\n";
  writeVec(os, "", "p1", {p1.x(), p1.y(), p1.z()});
  writeVec(os, "", "d1", {d1.x(), d1.y(), d1.z()});
  writeVec(os, "", "up_world_preferred", {up_world.x(), up_world.y(), up_world.z()});
  writePose(os, "", "T_tcpA_object", T_tcpA_obj);
  writePose(os, "", "T_tcpB_object", T_tcpB_obj);
  os << "A_FACE1:\n";
  os << "  physical_id: \"+Y\"\n";
  writeVec(os, "  ", "face_center", {f_a1.face_center.x(), f_a1.face_center.y(), f_a1.face_center.z()});
  writeVec(os, "  ", "face_normal", {f_a1.face_normal.x(), f_a1.face_normal.y(), f_a1.face_normal.z()});
  writeVec(os, "  ", "face_up", {f_a1.face_up.x(), f_a1.face_up.y(), f_a1.face_up.z()});
  os << "  extra_roll_from_preferred_deg: " << extraRollDeg(up_world, f_a1.face_up, d1) << "\n";
  writePose(os, "  ", "tcp", f_a1.object * T_tcpA_obj.inverse());
  os << "A_FACE2:\n";
  os << "  physical_id: \"-Y\"\n";
  writeVec(os, "  ", "face_center", {f_a2.face_center.x(), f_a2.face_center.y(), f_a2.face_center.z()});
  writeVec(os, "  ", "face_normal", {f_a2.face_normal.x(), f_a2.face_normal.y(), f_a2.face_normal.z()});
  writeVec(os, "  ", "face_up", {f_a2.face_up.x(), f_a2.face_up.y(), f_a2.face_up.z()});
  os << "  extra_roll_from_preferred_deg: " << extraRollDeg(up_world, f_a2.face_up, d1) << "\n";
  writePose(os, "  ", "tcp", f_a2.object * T_tcpA_obj.inverse());
  os << "B_FACE4_canonical:\n";
  os << "  physical_id: \"+X\"\n";
  os << "  roll_deg: 0.0\n";
  writeVec(os, "  ", "face_center", {f_b4.face_center.x(), f_b4.face_center.y(), f_b4.face_center.z()});
  writeVec(os, "  ", "face_normal", {f_b4.face_normal.x(), f_b4.face_normal.y(), f_b4.face_normal.z()});
  writeVec(os, "  ", "face_up", {f_b4.face_up.x(), f_b4.face_up.y(), f_b4.face_up.z()});
  writePose(os, "  ", "tcp", tcp_b4);
  writePose(os, "  ", "object", f_b4.object);
  os << "B_FACE5_canonical:\n";
  os << "  physical_id: \"-X\"\n";
  os << "  roll_deg: 0.0\n";
  writeVec(os, "  ", "face_center", {f_b5.face_center.x(), f_b5.face_center.y(), f_b5.face_center.z()});
  writeVec(os, "  ", "face_normal", {f_b5.face_normal.x(), f_b5.face_normal.y(), f_b5.face_normal.z()});
  writeVec(os, "  ", "face_up", {f_b5.face_up.x(), f_b5.face_up.y(), f_b5.face_up.z()});
  writePose(os, "  ", "tcp", tcp_b5);
  os << "B_FACE4_correct_yz_mirror_of_A_FACE1:\n";
  os << "  method: |\n";
  os << "    Mirror A face center/normal/up through world YZ, then reconstruct B object with\n";
  os << "    B +X face axes and T_tcpB = T_obj * T_tcpB_obj^{-1}. Not joint copy, not M*R*M.\n";
  writeVec(os, "  ", "face_center", {c_m4.x(), c_m4.y(), c_m4.z()});
  writeVec(os, "  ", "face_normal", {n_m4.x(), n_m4.y(), n_m4.z()});
  writeVec(os, "  ", "face_up", {up_m4.x(), up_m4.y(), up_m4.z()});
  os << "  extra_roll_from_canonical_deg: " << roll_m4 << "\n";
  writePose(os, "  ", "tcp", tcp_m4);
  writePose(os, "  ", "object", obj_m4);
  os << "B_FACE5_correct_yz_mirror_of_A_FACE2:\n";
  writeVec(os, "  ", "face_center", {c_m5.x(), c_m5.y(), c_m5.z()});
  writeVec(os, "  ", "face_normal", {n_m5.x(), n_m5.y(), n_m5.z()});
  writeVec(os, "  ", "face_up", {up_m5.x(), up_m5.y(), up_m5.z()});
  os << "  extra_roll_from_canonical_deg: " << roll_m5 << "\n";
  writePose(os, "  ", "tcp", tcp_m5);
  os << "DUAL7_I1:\n";
  os << "  recorded_roll_deg: " << dual7_roll << "\n";
  os << "  measured_extra_roll_from_canonical_deg: " << d7_roll4 << "\n";
  writeVec(os, "  ", "joints", i1);
  writePose(os, "  ", "tcp", tcp_d7_4);
  writeVec(os, "  ", "face_center", {d7_c4.x(), d7_c4.y(), d7_c4.z()});
  writeVec(os, "  ", "face_normal", {d7_n4.x(), d7_n4.y(), d7_n4.z()});
  writeVec(os, "  ", "face_up", {d7_u4.x(), d7_u4.y(), d7_u4.z()});
  os << "DUAL7_I2:\n";
  os << "  recorded_roll_deg: " << six["i2_roll_deg"].as<double>(-150.0) << "\n";
  os << "  measured_extra_roll_from_canonical_deg: " << d7_roll5 << "\n";
  writeVec(os, "  ", "joints", i2);
  writePose(os, "  ", "tcp", tcp_d7_5);
  writeVec(os, "  ", "face_center", {d7_c5.x(), d7_c5.y(), d7_c5.z()});
  writeVec(os, "  ", "face_normal", {d7_n5.x(), d7_n5.y(), d7_n5.z()});
  writeVec(os, "  ", "face_up", {d7_u5.x(), d7_u5.y(), d7_u5.z()});
  os << "tcp_errors:\n";
  os << "  FACE4:\n";
  os << "    canonical_vs_mirror_pos_m: " << p_cm4 << "\n";
  os << "    canonical_vs_mirror_ori_deg: " << o_cm4 << "\n";
  os << "    dual7_vs_canonical_pos_m: " << p_d7c4 << "\n";
  os << "    dual7_vs_canonical_ori_deg: " << o_d7c4 << "\n";
  os << "    dual7_vs_mirror_pos_m: " << p_d7m4 << "\n";
  os << "    dual7_vs_mirror_ori_deg: " << o_d7m4 << "\n";
  os << "    objectFromFace_vs_rolledAboutD1_pos_m: " << p_m4 << "\n";
  os << "    objectFromFace_vs_rolledAboutD1_ori_deg: " << o_m4 << "\n";
  os << "  FACE5:\n";
  os << "    canonical_vs_mirror_pos_m: " << p_cm5 << "\n";
  os << "    canonical_vs_mirror_ori_deg: " << o_cm5 << "\n";
  os << "    dual7_vs_canonical_pos_m: " << p_d7c5 << "\n";
  os << "    dual7_vs_canonical_ori_deg: " << o_d7c5 << "\n";
  os << "    dual7_vs_mirror_pos_m: " << p_d7m5 << "\n";
  os << "    dual7_vs_mirror_ori_deg: " << o_d7m5 << "\n";
  os << "ik_coverage_canonical_FACE4:\n";
  os << "  attempts: " << cov4.attempts << "\n";
  os << "  ik_success: " << cov4.success << "\n";
  os << "  unique_fk_limit_ok: " << cov4.unique.size() << "\n";
  os << "  unique_collision_free: " << legal4 << "\n";
  os << "  distinct_config_groups: " << groups4.size() << "\n";
  os << "  groups:\n";
  for (const auto& kv : groups4)
    os << "    - {sig: \"" << kv.first << "\", count: " << kv.second << "}\n";
  os << "  unique_samples:\n";
  for (size_t i = 0; i < cov4.unique.size() && i < 16; ++i)
  {
    const auto& h = cov4.unique[i];
    os << "    - seed: \"" << h.seed << "\"\n";
    writeVec(os, "      ", "joint_values", h.joints);
    os << "      sig: \"" << configSig(h.joints) << "\"\n";
    os << "      collision_ok: " << (h.collision_ok ? "true" : "false") << "\n";
    os << "      collision_pair: \"" << h.pair << "\"\n";
  }
  os << "ik_coverage_canonical_FACE5:\n";
  os << "  attempts: " << cov5.attempts << "\n";
  os << "  ik_success: " << cov5.success << "\n";
  os << "  unique_fk_limit_ok: " << cov5.unique.size() << "\n";
  os << "  unique_collision_free: " << legal5 << "\n";
  os << "  distinct_config_groups: " << groups5.size() << "\n";
  os << "  groups:\n";
  for (const auto& kv : groups5)
    os << "    - {sig: \"" << kv.first << "\", count: " << kv.second << "}\n";
  os << "targeted_seed_tests:\n";
  for (const auto& r : trows)
  {
    os << "  - label: \"" << r.label << "\"\n";
    os << "    seed: \"" << r.seed << "\"\n";
    os << "    ik_ok: " << (r.ik ? "true" : "false") << "\n";
    os << "    collision_ok: " << (r.col_ok ? "true" : "false") << "\n";
    os << "    collision_pair: \"" << r.pair << "\"\n";
    if (!r.q.empty())
    {
      writeVec(os, "    ", "joint_values", r.q);
      os << "    sig: \"" << configSig(r.q) << "\"\n";
    }
  }

  auto dumpHits = [&](const std::string& key, const std::vector<ContactHit>& hits) {
    std::string pair;
    os << key << ":\n";
    os << "  illegal: " << (hasIllegal(hits, pair) ? "true" : "false") << "\n";
    os << "  first_illegal_pair: \"" << pair << "\"\n";
    os << "  contacts:\n";
    for (const auto& h : hits)
    {
      os << "    - pair: \"" << h.a << " <-> " << h.b << "\"\n";
      os << "      allowed_acm: " << (h.allowed ? "true" : "false") << "\n";
      os << "      depth: " << h.depth << "\n";
      writeVec(os, "      ", "pos", {h.pos.x(), h.pos.y(), h.pos.z()});
    }
  };
  os << "dual7_static_collision:\n";
  os << "  arm_a_state: dual8_home_a  # Phase7 return Home, not arbitrary initial\n";
  writeVec(os, "  ", "home_a", home_a);
  writeVec(os, "  ", "i1_b", i1);
  writeVec(os, "  ", "i2_b", i2);
  os << "  column:\n";
  writeVec(os, "    ", "xyz", {column.xyz.x(), column.xyz.y(), column.xyz.z()});
  writeVec(os, "    ", "dim", {column.dim.x(), column.dim.y(), column.dim.z()});
  writeVec(os, "    ", "aabb_min", {col_min.x(), col_min.y(), col_min.z()});
  writeVec(os, "    ", "aabb_max", {col_max.x(), col_max.y(), col_max.z()});
  dumpLink(os, "  ", "arm_a_base_link", st_i1);
  dumpLink(os, "  ", "arm_b_base_link", st_i1);
  dumpLink(os, "  ", "arm_b_upperarm_link", st_i1);
  dumpLink(os, "  ", "arm_b_forearm_link", st_i1);
  dumpLink(os, "  ", "arm_a_forearm_link", st_i1);
  os << "  forearm_origin_to_column_aabb_m: " << (fo - closest).norm() << "\n";
  writeVec(os, "  ", "forearm_origin", {fo.x(), fo.y(), fo.z()});
  os << "  forearm_collision_geometry: mesh STL fairino3_v6/forearm_link.STL (same as visual)\n";
  dumpHits("  i1_with_part", hits_i1_part);
  dumpHits("  i1_without_part", hits_i1_none);
  dumpHits("  i2_with_part", hits_i2_part);
  dumpHits("  i2_without_part", hits_i2_none);
  dumpHits("  face6_with_part", hits_f6);

  os << "rviz_preview:\n";
  os << "  DIAG_DUAL7_I1:\n";
  writeVec(os, "    ", "arm_a", home_a);
  writeVec(os, "    ", "arm_b", i1);
  os << "    owner: B\n";
  os << "    note: \"DUAL-7 I1 at Home A; collision check in current PlanningScene\"\n";
  os << "    illegal: " << (i1_bad ? "true" : "false") << "\n";
  os << "  DIAG_DUAL7_I2:\n";
  writeVec(os, "    ", "arm_a", home_a);
  writeVec(os, "    ", "arm_b", i2);
  os << "    owner: B\n";
  os << "    note: \"DUAL-7 I2 at Home A\"\n";
  os << "    illegal: " << (i2_bad ? "true" : "false") << "\n";
  os << "  DIAG_FACE6:\n";
  writeVec(os, "    ", "arm_a", home_a);
  writeVec(os, "    ", "arm_b", b_face6.empty() ? i3 : b_face6);
  os << "    owner: B\n";
  os << "    note: \"legal B_FACE6 for comparison\"\n";

  os << "answers_draft:\n";
  os << "  dual7_collides_in_current_model: " << ((i1_bad || i2_bad) ? "true" : "false") << "\n";
  os << "  unique_ik_config_groups_face4: " << groups4.size() << "\n";
  os << "  unique_ik_config_groups_face5: " << groups5.size() << "\n";
  os << "  canonical_roll0_is_tcp_roll0: false\n";
  os << "  canonical_roll0_is_preferred_object_up: true\n";
  os << "  canonical_matches_a_face_mirror: false\n";
  os << "  face4_canonical_vs_mirror_ori_deg: " << o_cm4 << "\n";
  os << "  face5_canonical_vs_mirror_ori_deg: " << o_cm5 << "\n";
  os.close();
  emit("wrote " + out_path);

  rclcpp::QoS qos(1);
  qos.transient_local();
  qos.reliable();
  auto s_pub = node->create_publisher<moveit_msgs::msg::DisplayRobotState>("/keypose/preview/robot_state", qos);
  auto m_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("/keypose/preview/markers", qos);

  moveit_msgs::msg::DisplayRobotState drs;
  drs.state.joint_state.header.frame_id = "world";
  drs.state.joint_state.name = kArmA;
  drs.state.joint_state.name.insert(drs.state.joint_state.name.end(), kArmB.begin(), kArmB.end());
  drs.state.joint_state.name.push_back("arm_a_gripper_joint");
  drs.state.joint_state.name.push_back("arm_b_gripper_joint");
  drs.state.joint_state.position.insert(drs.state.joint_state.position.end(), home_a.begin(), home_a.end());
  drs.state.joint_state.position.insert(drs.state.joint_state.position.end(), i1.begin(), i1.end());
  drs.state.joint_state.position.push_back(qa_open);
  drs.state.joint_state.position.push_back(qb_hold);
  drs.state.is_diff = false;
  s_pub->publish(drs);

  visualization_msgs::msg::MarkerArray arr;
  arr.markers.push_back(cubeMarker(1, column.xyz, column.dim, 0.9f, 0.2f, 0.2f, 0.35f, "workcell"));
  arr.markers.push_back(cubeMarker(2, table.xyz, table.dim, 0.5f, 0.5f, 0.5f, 0.35f, "workcell"));
  arr.markers.push_back(arrowMarker(10, f_b4.face_center, f_b4.face_normal, 0.2f, 0.8f, 1.0f, "canonical"));
  arr.markers.push_back(arrowMarker(11, f_b4.face_center, f_b4.face_up, 0.2f, 0.4f, 1.0f, "canonical"));
  arr.markers.push_back(arrowMarker(12, c_m4, n_m4, 0.2f, 1.0f, 0.3f, "mirror"));
  arr.markers.push_back(arrowMarker(13, c_m4, up_m4, 0.1f, 0.7f, 0.2f, "mirror"));
  arr.markers.push_back(arrowMarker(14, d7_c4, d7_n4, 1.0f, 0.6f, 0.1f, "dual7"));
  arr.markers.push_back(arrowMarker(15, d7_c4, d7_u4, 1.0f, 0.3f, 0.1f, "dual7"));
  int cid = 20;
  for (const auto& h : hits_i1_part)
  {
    if (h.allowed)
      continue;
    arr.markers.push_back(arrowMarker(cid++, h.pos, h.normal, 1.0f, 0.0f, 0.0f, "contact"));
  }
  visualization_msgs::msg::Marker title;
  title.header.frame_id = "world";
  title.ns = "keypose";
  title.id = 1;
  title.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  title.action = visualization_msgs::msg::Marker::ADD;
  title.pose.position.y = 0.55;
  title.pose.position.z = 1.55;
  title.pose.orientation.w = 1.0;
  title.scale.z = 0.04;
  title.color.r = 1;
  title.color.g = 1;
  title.color.b = 1;
  title.color.a = 1;
  title.text = std::string("DIAG DUAL-7 I1 | A=Home | ") + (i1_bad ? "COLLIDES" : "NO ILLEGAL COLLISION");
  arr.markers.push_back(title);
  m_pub->publish(arr);

  emit("published DisplayRobotState DIAG_DUAL7_I1 on /keypose/preview/robot_state");
  if (hold)
  {
    emit("hold=true, spinning for RViz");
    rclcpp::spin(node);
  }
  else
  {
    rclcpp::sleep_for(std::chrono::milliseconds(400));
  }
  rclcpp::shutdown();
  return 0;
}
