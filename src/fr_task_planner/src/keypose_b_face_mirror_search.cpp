// B_FACE4/5 correct-mirror IK search. Local PlanningScene only. No path, no hardware.
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
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <yaml-cpp/yaml.h>

namespace
{
constexpr char kGroupB[] = "arm_b";
constexpr char kTcpB[] = "arm_b_gripper_tcp";
constexpr char kTable[] = "table";
constexpr char kColumn[] = "mounting_column";
constexpr char kPart[] = "small_part";
constexpr double kMaxSec = 180.0;

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

void emit(const std::string& s) { std::cout << s << std::endl; }

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

Eigen::Vector3d mirrorYz(const Eigen::Vector3d& p) { return Eigen::Vector3d(-p.x(), p.y(), p.z()); }

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

struct J12Sol
{
  double j1 = 0.0;
  double j2 = 0.0;
};

std::vector<J12Sol> j12FromElbowBase(const Eigen::Vector3d& e,
                                     const moveit::core::RobotModelConstPtr& model)
{
  constexpr double kL = 0.28;
  constexpr double kSz = 0.14;
  std::vector<J12Sol> out;
  double j1lo = -3.0543, j1hi = 3.0543, j2lo = -4.6251, j2hi = 1.4835;
  jointLimits(model, "arm_b_j1", j1lo, j1hi);
  jointLimits(model, "arm_b_j2", j2lo, j2hi);
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
  Eigen::Vector3d elbow = Eigen::Vector3d::Zero();
  Eigen::Vector3d upperarm_dir = Eigen::Vector3d::UnitX();
};

ArmGeom armGeom(moveit::core::RobotState& st, bool is_a)
{
  ArmGeom g;
  const char* up = is_a ? "arm_a_upperarm_link" : "arm_b_upperarm_link";
  const char* el = is_a ? "arm_a_forearm_link" : "arm_b_forearm_link";
  const Eigen::Vector3d sh = st.getGlobalLinkTransform(up).translation();
  g.elbow = st.getGlobalLinkTransform(el).translation();
  const Eigen::Vector3d v = g.elbow - sh;
  g.upperarm_dir = v.norm() > 1e-9 ? v.normalized() : Eigen::Vector3d::UnitX();
  return g;
}

double jointL2(const std::vector<double>& a, const std::vector<double>& b)
{
  double s = 0.0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}

double limitMargin(const moveit::core::RobotState& st, const moveit::core::JointModelGroup* g)
{
  double m = 1e9;
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

void attachCylinder(planning_scene::PlanningScene& scene, const Eigen::Isometry3d& in_link, double r,
                    double h)
{
  clearObject(scene);
  moveit_msgs::msg::AttachedCollisionObject att;
  att.link_name = kTcpB;
  att.touch_links = kTouchB;
  att.object.id = kPart;
  att.object.header.frame_id = kTcpB;
  att.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  att.object.pose.orientation.w = 1.0;
  att.object.primitives.resize(1);
  att.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  att.object.primitives[0].dimensions = {h, r};
  att.object.primitive_poses.push_back(poseMsg(in_link));
  scene.processAttachedCollisionObjectMsg(att);
}

bool allowedPair(const std::string& a, const std::string& b)
{
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

bool colliding(planning_scene::PlanningScene& scene, moveit::core::RobotState& st, std::string& pair)
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
  for (const auto& c : res.contacts)
  {
    const std::string& a = c.first.first;
    const std::string& b = c.first.second;
    if (allowedPair(a, b))
      continue;
    pair = a + " <-> " + b;
    return true;
  }
  return false;
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

bool checkB(planning_scene::PlanningScene& scene, const std::vector<double>& home_a,
            const std::vector<double>& b, double qa, double qb, const Eigen::Isometry3d& tcp_obj,
            double r, double h, std::string& pair)
{
  attachCylinder(scene, tcp_obj, r, h);
  applyState(scene, home_a, b, qa, qb);
  auto st = scene.getCurrentStateNonConst();
  return colliding(scene, st, pair);
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

Eigen::Isometry3d interpPose(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double t)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = (1.0 - t) * a.translation() + t * b.translation();
  Eigen::Quaterniond qa(a.linear());
  Eigen::Quaterniond qb(b.linear());
  qa.normalize();
  qb.normalize();
  T.linear() = qa.slerp(t, qb).toRotationMatrix();
  return T;
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

std::string configSig(const std::vector<double>& q)
{
  if (q.size() < 6)
    return "short";
  const int j1 = static_cast<int>(std::lround(q[0] * 180.0 / M_PI / 30.0) * 30);
  std::string j2 = q[1] < -3.4 ? "folded_down" : (q[1] < -2.0 ? "folded" : (q[1] < 0 ? "mid" : "open"));
  std::ostringstream o;
  o << "j1_" << j1 << "|" << j2 << "|e" << (q[2] >= 0 ? "+" : "-") << "|w" << (q[4] >= 0 ? "+" : "-");
  return o.str();
}

struct Hit
{
  std::string seed;
  std::vector<double> joints;
  Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity();
  bool limits_ok = false;
  bool collision_ok = false;
  bool face_ok = false;
  std::string pair = "none";
  double fk_pos = 0.0;
  double fk_ori = 0.0;
  double face_c = 0.0;
  double face_n = 0.0;
  double face_u = 0.0;
  double elbow_err = 0.0;
  double upperarm_err_deg = 0.0;
  double fold_cost = 0.0;
  double j12_err = 0.0;
  double margin = 0.0;
  Eigen::Vector3d face_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d face_normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d face_up = Eigen::Vector3d::UnitX();
};

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("keypose_b_face_mirror_search", opt);
  emit("========== B_FACE4/5 MIRROR IK SEARCH ==========");
  emit("NO PATH PLANNING. NO EXECUTION. COLLISION NOT RELAXED.");

  const std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string search_yaml = node->get_parameter_or(
      "search_yaml", home_dir + "/fr_task_ws/src/fr_task_planner/config/"
                                "keypose_optimization_v1/search.yaml");
  YAML::Node cfg = YAML::LoadFile(search_yaml);
  const std::string out_dir = node->get_parameter_or("output_dir", cfg["output_dir"].as<std::string>());

  YAML::Node home_y = YAML::LoadFile(cfg["home_yaml"].as<std::string>());
  YAML::Node six = YAML::LoadFile(cfg["six_face_yaml"].as<std::string>());
  YAML::Node work = YAML::LoadFile(cfg["workcell_yaml"].as<std::string>());
  YAML::Node arm_a_y = YAML::LoadFile(out_dir + "/arm_a_candidates.yaml");
  YAML::Node arm_b_y = YAML::LoadFile(out_dir + "/arm_b_candidates.yaml");
  YAML::Node tgt_y = YAML::LoadFile(out_dir + "/task_space_targets.yaml");
  YAML::Node s12 = YAML::LoadFile(cfg["step12c_winner"].as<std::string>());

  std::vector<double> home_a, home_b, i1, i2, i3, a1, a2, face6, han_b;
  yamlVec(home_y["home_a"], home_a);
  yamlVec(home_y["home_b"], home_b);
  yamlVec(six["i1_b"], i1);
  yamlVec(six["i2_b"], i2);
  yamlVec(six["i3_b"], i3);
  yamlVec(six["handover_b"], han_b);
  yamlVec(arm_a_y["keyposes"]["A_FACE1"]["candidates"][0]["joint_values"], a1);
  yamlVec(arm_a_y["keyposes"]["A_FACE2"]["candidates"][0]["joint_values"], a2);
  yamlVec(arm_b_y["keyposes"]["B_FACE6"]["candidates"][0]["joint_values"], face6);
  const double qa_open = 0.0;
  const double qb_hold = 0.083;
  const double radius = 0.0075;
  const double height = 0.035;

  Eigen::Isometry3d T_tcpB_obj = Eigen::Isometry3d::Identity();
  loadXyzw(tgt_y["T_tcpB_object"]["pose"], T_tcpB_obj);
  const Eigen::Vector3d p1(s12["p1"][0].as<double>(), s12["p1"][1].as<double>(),
                           s12["p1"][2].as<double>());
  const Eigen::Vector3d d1(s12["surface_target_normal"][0].as<double>(),
                           s12["surface_target_normal"][1].as<double>(),
                           s12["surface_target_normal"][2].as<double>());

  Eigen::Vector3d a1_c, a1_n, a1_u, a2_c, a2_n, a2_u;
  yamlVec3(tgt_y["A_FACE1"]["face_center"], a1_c);
  yamlVec3(tgt_y["A_FACE1"]["face_normal"], a1_n);
  yamlVec3(tgt_y["A_FACE1"]["face_up"], a1_u);
  yamlVec3(tgt_y["A_FACE2"]["face_center"], a2_c);
  yamlVec3(tgt_y["A_FACE2"]["face_normal"], a2_n);
  yamlVec3(tgt_y["A_FACE2"]["face_up"], a2_u);
  const Eigen::Vector3d b4_c_obj(0.0075, 0, 0);
  const Eigen::Vector3d b4_n_obj(1, 0, 0);
  const Eigen::Vector3d b4_u_obj(0, 0, 1);
  const Eigen::Vector3d b5_c_obj(-0.0075, 0, 0);
  const Eigen::Vector3d b5_n_obj(-1, 0, 0);
  const Eigen::Vector3d b5_u_obj(0, 0, 1);
  const Eigen::Isometry3d obj4 =
      objectFromFace(mirrorYz(a1_c), mirrorYz(a1_n), mirrorYz(a1_u), b4_c_obj, b4_n_obj, b4_u_obj);
  const Eigen::Isometry3d obj5 =
      objectFromFace(mirrorYz(a2_c), mirrorYz(a2_n), mirrorYz(a2_u), b5_c_obj, b5_n_obj, b5_u_obj);
  const Eigen::Isometry3d tcp4 = obj4 * T_tcpB_obj.inverse();
  const Eigen::Isometry3d tcp5 = obj5 * T_tcpB_obj.inverse();

  robot_model_loader::RobotModelLoader loader(node);
  auto model = loader.getModel();
  auto* gb = model->getJointModelGroup(kGroupB);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  addBox(*scene, loadBox(work["table"], kTable));
  addBox(*scene, loadBox(work["column"], kColumn));
  auto& acm = scene->getAllowedCollisionMatrixNonConst();
  acm.setEntry("arm_a_base_link", kColumn, true);
  acm.setEntry("arm_b_base_link", kColumn, true);
  for (const auto& t : kTouchB)
    acm.setEntry(kPart, t, true);

  moveit::core::RobotState start(model);
  start.setToDefaultValues();
  setJoints(start, kArmA, home_a);
  setJoints(start, kArmB, home_b);
  start.update();
  scene->setCurrentState(start);

  auto geomA = [&](const std::vector<double>& q) {
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, kArmA, q);
    st.update();
    return armGeom(st, true);
  };
  const ArmGeom ga1 = geomA(a1);
  const ArmGeom ga2 = geomA(a2);
  const Eigen::Vector3d el_m4 = mirrorYz(ga1.elbow);
  const Eigen::Vector3d dir_m4 = mirrorYz(ga1.upperarm_dir);
  const Eigen::Vector3d el_m5 = mirrorYz(ga2.elbow);
  const Eigen::Vector3d dir_m5 = mirrorYz(ga2.upperarm_dir);

  auto j12s = [&](const Eigen::Vector3d& el_m) {
    moveit::core::RobotState st(scene->getCurrentState());
    st.update();
    const Eigen::Isometry3d Tb = st.getGlobalLinkTransform("arm_b_base_link");
    return j12FromElbowBase(Tb.inverse() * el_m, model);
  };
  const auto j12_4 = j12s(el_m4);
  const auto j12_5 = j12s(el_m5);
  emit("A_FACE1 mirrored J12 families: " + std::to_string(j12_4.size()));
  for (size_t i = 0; i < j12_4.size(); ++i)
    emit("  F4 f" + std::to_string(i) + " J1=" + std::to_string(j12_4[i].j1 * 180 / M_PI) +
         " J2=" + std::to_string(j12_4[i].j2 * 180 / M_PI));
  emit("A_FACE2 mirrored J12 families: " + std::to_string(j12_5.size()));
  for (size_t i = 0; i < j12_5.size(); ++i)
    emit("  F5 f" + std::to_string(i) + " J1=" + std::to_string(j12_5[i].j1 * 180 / M_PI) +
         " J2=" + std::to_string(j12_5[i].j2 * 180 / M_PI));

  auto fillFace = [&](Hit& h, const Eigen::Isometry3d& T_tcp_obj, const Eigen::Vector3d& c_obj,
                      const Eigen::Vector3d& n_obj, const Eigen::Vector3d& u_obj,
                      const Eigen::Vector3d& want_c, const Eigen::Vector3d& want_n,
                      const Eigen::Vector3d& want_u) {
    const Eigen::Isometry3d obj = h.tcp * T_tcp_obj;
    h.face_center = obj * c_obj;
    h.face_normal = (obj.linear() * n_obj).normalized();
    h.face_up = (obj.linear() * u_obj).normalized();
    h.face_c = (h.face_center - want_c).norm();
    h.face_n = angDeg(h.face_normal, want_n);
    h.face_u = angDeg(h.face_up, want_u);
    h.face_ok = h.face_c <= 0.002 && h.face_n <= 3.0 && h.face_u <= 5.0;
  };

  auto score = [&](Hit& h, const Eigen::Vector3d& el_m, const Eigen::Vector3d& dir_m,
                   const std::vector<J12Sol>& j12) {
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, kArmA, home_a);
    setJoints(st, kArmB, h.joints);
    st.update();
    const ArmGeom gb = armGeom(st, false);
    h.elbow_err = (gb.elbow - el_m).norm();
    h.upperarm_err_deg = angDeg(gb.upperarm_dir, dir_m);
    h.fold_cost = 50.0 * h.elbow_err + 0.35 * h.upperarm_err_deg +
                  18.0 * std::max(0.0, gb.upperarm_dir.z());
    h.j12_err = 1e9;
    for (const auto& s : j12)
      h.j12_err = std::min(h.j12_err, std::hypot(h.joints[0] - s.j1, h.joints[1] - s.j2));
  };

  auto makeSeeds = [&](const std::vector<J12Sol>& j12) {
    std::vector<std::pair<std::string, std::vector<double>>> seeds;
    const std::vector<std::vector<double>> tmpls = {i1, i2, face6.empty() ? i3 : face6, han_b, home_b};
    const char* tn[] = {"d7i1", "d7i2", "face6", "han", "home"};
    for (size_t fi = 0; fi < j12.size(); ++fi)
    {
      for (size_t ti = 0; ti < tmpls.size(); ++ti)
      {
        auto q = tmpls[ti];
        if (q.size() < 6)
          continue;
        q[0] = j12[fi].j1;
        q[1] = j12[fi].j2;
        q = clampJoints(model, kArmB, q);
        seeds.push_back({"geo" + std::to_string(fi) + "_" + tn[ti], q});
        auto qn = q;
        qn[2] = -qn[2];
        seeds.push_back({"geo" + std::to_string(fi) + "_" + tn[ti] + "_e", clampJoints(model, kArmB, qn)});
        auto qw = q;
        qw[4] = -qw[4];
        seeds.push_back({"geo" + std::to_string(fi) + "_" + tn[ti] + "_w", clampJoints(model, kArmB, qw)});
      }
    }
    seeds.push_back({"dual7_i1", i1});
    seeds.push_back({"dual7_i2", i2});
    seeds.push_back({"face6", face6.empty() ? i3 : face6});
    const double j6off[] = {-M_PI, -2.0, -1.57, -1.0, 1.0, 1.57, 2.0, M_PI};
    for (double off : j6off)
    {
      auto q = i1;
      q[5] += off;
      seeds.push_back({"d7_j6", clampJoints(model, kArmB, q)});
      auto q5 = i1;
      q5[4] = -q5[4];
      q5[5] += off;
      seeds.push_back({"d7_w_j6", clampJoints(model, kArmB, q5)});
    }
    const double j3s[] = {-2.0, -1.2, -0.5, 0.5, 1.2, 2.0};
    const double j4s[] = {-3.0, -1.5, 0.0, 1.2};
    const double j5s[] = {-2.0, -1.0, 1.0, 2.0};
    for (const auto& sol : j12)
    {
      for (double j3 : j3s)
        for (double j4 : j4s)
          for (double j5 : j5s)
          {
            std::vector<double> q = {sol.j1, sol.j2, j3, j4, j5, i1.size() > 5 ? i1[5] : 0.0};
            seeds.push_back({"grid", clampJoints(model, kArmB, q)});
          }
    }
    return seeds;
  };

  auto searchFace = [&](const std::string& name, const Eigen::Isometry3d& target,
                        const Eigen::Isometry3d& obj, const Eigen::Vector3d& c_obj,
                        const Eigen::Vector3d& n_obj, const Eigen::Vector3d& u_obj,
                        const Eigen::Vector3d& want_c, const Eigen::Vector3d& want_n,
                        const Eigen::Vector3d& want_u, const Eigen::Vector3d& el_m,
                        const Eigen::Vector3d& dir_m, const std::vector<J12Sol>& j12,
                        const Eigen::Isometry3d& dual7_tcp, const std::vector<double>& dual7_q) {
    emit("--- " + name + " ---");
    (void)obj;
    emit("  target xyz " + fmt({target.translation().x(), target.translation().y(),
                                target.translation().z()},
                               6));
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    std::vector<Hit> unique_all;
    std::vector<Hit> legal;
    int attempts = 0;
    int success = 0;
    int timed_out = 0;
    const int max_att = 420;

    auto consider = [&](moveit::core::RobotState& st, const std::string& sname, double timeout) {
      if (elapsed() >= kMaxSec || attempts >= max_att || static_cast<int>(legal.size()) >= 3)
        return;
      ++attempts;
      if (!st.setFromIK(gb, target, kTcpB, timeout))
      {
        ++timed_out;
        return;
      }
      ++success;
      Hit h;
      h.seed = sname;
      h.joints = jointsOf(st, kArmB);
      h.tcp = st.getGlobalLinkTransform(kTcpB);
      poseError(target, h.tcp, h.fk_pos, h.fk_ori);
      h.limits_ok = st.satisfiesBounds(gb);
      h.margin = limitMargin(st, gb);
      if (h.fk_pos > 0.003 || h.fk_ori > 2.0 || !h.limits_ok)
        return;
      for (const auto& u : unique_all)
        if (jointL2(u.joints, h.joints) < 0.15)
          return;
      fillFace(h, T_tcpB_obj, c_obj, n_obj, u_obj, want_c, want_n, want_u);
      if (!h.face_ok)
        return;
      h.collision_ok = !checkB(*scene, home_a, h.joints, qa_open, qb_hold, T_tcpB_obj, radius, height,
                               h.pair);
      score(h, el_m, dir_m, j12);
      unique_all.push_back(h);
      if (h.collision_ok)
      {
        legal.push_back(h);
        emit("  LEGAL #" + std::to_string(legal.size()) + " " + sname + " " + deg6(h.joints) +
             " col=PASS fold=" + std::to_string(h.fold_cost));
      }
      else if (unique_all.size() <= 12)
        emit("  IK " + sname + " " + configSig(h.joints) + " COL " + h.pair);
    };

    // Homotopy from known reachable TCPs. Not a motion plan.
    struct Homotopy
    {
      std::string name;
      Eigen::Isometry3d start;
      std::vector<double> q;
    };
    Eigen::Isometry3d tcp_can4 = Eigen::Isometry3d::Identity();
    loadXyzw(tgt_y["B_FACE4"]["tcp"], tcp_can4);
    Eigen::Isometry3d tcp_can5 = Eigen::Isometry3d::Identity();
    loadXyzw(tgt_y["B_FACE5"]["tcp"], tcp_can5);
    Eigen::Isometry3d tcp_f6 = Eigen::Isometry3d::Identity();
    loadXyzw(tgt_y["B_FACE6"]["tcp"], tcp_f6);
    std::vector<Homotopy> homos = {
        {"homo_d7", dual7_tcp, dual7_q},
        {"homo_f6", tcp_f6, face6.empty() ? i3 : face6},
        {"homo_can4", tcp_can4, i1},
        {"homo_can5", tcp_can5, i2},
    };
    for (const auto& ho : homos)
    {
      if (elapsed() >= kMaxSec || static_cast<int>(legal.size()) >= 3)
        break;
      std::vector<double> q = ho.q;
      bool ok = true;
      const int steps = 24;
      for (int s = 1; s <= steps; ++s)
      {
        if (elapsed() >= kMaxSec)
          break;
        const double t = static_cast<double>(s) / steps;
        const Eigen::Isometry3d mid = interpPose(ho.start, target, t);
        moveit::core::RobotState st(scene->getCurrentState());
        setJoints(st, kArmA, home_a);
        setJoints(st, kArmB, q);
        st.update();
        ++attempts;
        if (!st.setFromIK(gb, mid, kTcpB, s == steps ? 0.25 : 0.10))
        {
          ok = false;
          emit("  " + ho.name + " broke at t=" + std::to_string(t));
          break;
        }
        q = jointsOf(st, kArmB);
        if (s == steps)
        {
          --attempts;
          consider(st, ho.name + "_end", 0.25);
        }
      }
      if (!ok)
      {
        auto qj = q;
        for (double off : {-1.57, 1.57, 3.0, -3.0})
        {
          qj = q;
          if (qj.size() >= 6)
            qj[5] += off;
          qj = clampJoints(model, kArmB, qj);
          moveit::core::RobotState st(scene->getCurrentState());
          setJoints(st, kArmA, home_a);
          setJoints(st, kArmB, qj);
          st.update();
          consider(st, ho.name + "_rescue", 0.20);
        }
      }
    }

    auto seeds = makeSeeds(j12);
    std::mt19937 rng(7);
    for (const auto& seed : seeds)
    {
      if (elapsed() >= kMaxSec || attempts >= max_att || static_cast<int>(legal.size()) >= 3)
        break;
      moveit::core::RobotState exact(scene->getCurrentState());
      setJoints(exact, kArmA, home_a);
      setJoints(exact, kArmB, seed.second);
      exact.update();
      consider(exact, seed.first + "_exact", 0.18);
      for (int n = 0; n < 2; ++n)
      {
        if (elapsed() >= kMaxSec || attempts >= max_att || static_cast<int>(legal.size()) >= 3)
          break;
        moveit::core::RobotState nearby(exact);
        for (int j = 2; j < 6; ++j)
        {
          std::uniform_real_distribution<double> d(-0.8, 0.8);
          nearby.setVariablePosition(kArmB[j], nearby.getVariablePosition(kArmB[j]) + d(rng));
        }
        nearby.enforceBounds(gb);
        nearby.update();
        consider(nearby, seed.first + "_near", 0.08);
      }
    }

    std::sort(legal.begin(), legal.end(), [](const Hit& a, const Hit& b) {
      const bool af = a.joints[1] < -2.0;
      const bool bf = b.joints[1] < -2.0;
      if (af != bf)
        return af;
      return a.fold_cost < b.fold_cost;
    });
    if (legal.size() > 3)
      legal.resize(3);
    std::sort(unique_all.begin(), unique_all.end(),
              [](const Hit& a, const Hit& b) { return a.fold_cost < b.fold_cost; });
    emit("  done t=" + std::to_string(elapsed()) + "s attempts=" + std::to_string(attempts) +
         " ik_ok=" + std::to_string(success) + " unique=" + std::to_string(unique_all.size()) +
         " legal=" + std::to_string(legal.size()) + " kdl_fail=" + std::to_string(timed_out));
    struct Pack
    {
      std::vector<Hit> legal;
      std::vector<Hit> all;
      int attempts = 0;
      int success = 0;
      int kdl_fail = 0;
      double sec = 0;
    };
    Pack p;
    p.legal = legal;
    p.all = unique_all;
    p.attempts = attempts;
    p.success = success;
    p.kdl_fail = timed_out;
    p.sec = elapsed();
    return p;
  };

  moveit::core::RobotState st_d7(scene->getCurrentState());
  setJoints(st_d7, kArmA, home_a);
  setJoints(st_d7, kArmB, i1);
  st_d7.update();
  const Eigen::Isometry3d tcp_d7 = st_d7.getGlobalLinkTransform(kTcpB);
  moveit::core::RobotState st_d72(scene->getCurrentState());
  setJoints(st_d72, kArmA, home_a);
  setJoints(st_d72, kArmB, i2);
  st_d72.update();
  const Eigen::Isometry3d tcp_d72 = st_d72.getGlobalLinkTransform(kTcpB);

  auto r4 = searchFace("B_FACE4", tcp4, obj4, b4_c_obj, b4_n_obj, b4_u_obj, mirrorYz(a1_c),
                       mirrorYz(a1_n).normalized(), mirrorYz(a1_u), el_m4, dir_m4, j12_4, tcp_d7, i1);
  auto r5 = searchFace("B_FACE5", tcp5, obj5, b5_c_obj, b5_n_obj, b5_u_obj, mirrorYz(a2_c),
                       mirrorYz(a2_n).normalized(), mirrorYz(a2_u), el_m5, dir_m5, j12_5, tcp_d72, i2);

  const std::string out_path = out_dir + "/b_face45_mirror_search.yaml";
  std::ofstream os(out_path);
  os << "task_version: KEYPOSE_B_FACE45_MIRROR_SEARCH\n";
  os << "execution: false\n";
  os << "collision_relaxed: false\n";
  writePose(os, "", "B_FACE4_mirror_tcp", tcp4);
  writePose(os, "", "B_FACE4_mirror_object", obj4);
  writePose(os, "", "B_FACE5_mirror_tcp", tcp5);
  writePose(os, "", "B_FACE5_mirror_object", obj5);

  auto dumpFace = [&](const std::string& key, const auto& r) {
    os << key << ":\n";
    os << "  seconds: " << r.sec << "\n";
    os << "  attempts: " << r.attempts << "\n";
    os << "  ik_success: " << r.success << "\n";
    os << "  kdl_no_converge: " << r.kdl_fail << "\n";
    os << "  unique_pose_ok: " << r.all.size() << "\n";
    os << "  unique_legal: " << r.legal.size() << "\n";
    os << "  legal: " << (r.legal.empty() ? "false" : "true") << "\n";
    os << "  candidates:\n";
    int idx = 1;
    for (const auto& h : r.legal)
    {
      os << "    - index: " << idx++ << "\n";
      os << "      seed: \"" << h.seed << "\"\n";
      writeVec(os, "      ", "joint_values", h.joints);
      os << "      sig: \"" << configSig(h.joints) << "\"\n";
      writePose(os, "      ", "tcp", h.tcp);
      os << "      fk_position_error_m: " << h.fk_pos << "\n";
      os << "      fk_orientation_error_deg: " << h.fk_ori << "\n";
      os << "      face_center_error_m: " << h.face_c << "\n";
      os << "      face_normal_error_deg: " << h.face_n << "\n";
      os << "      face_up_error_deg: " << h.face_u << "\n";
      os << "      joint_limit_check: PASS\n";
      os << "      joint_limit_margin: " << h.margin << "\n";
      os << "      collision_check: PASS\n";
      os << "      elbow_mirror_err_m: " << h.elbow_err << "\n";
      os << "      upperarm_mirror_err_deg: " << h.upperarm_err_deg << "\n";
      os << "      j12_vs_geo_mirror_rad: " << h.j12_err << "\n";
      os << "      fold_cost: " << h.fold_cost << "\n";
    }
    if (r.legal.empty())
      os << "    []\n";
    os << "  closest_even_if_colliding:\n";
    if (r.all.empty())
      os << "    none: true\n";
    else
    {
      const auto& h = r.all.front();
      os << "    seed: \"" << h.seed << "\"\n";
      writeVec(os, "    ", "joint_values", h.joints);
      os << "    sig: \"" << configSig(h.joints) << "\"\n";
      os << "    collision_ok: " << (h.collision_ok ? "true" : "false") << "\n";
      os << "    collision_pair: \"" << h.pair << "\"\n";
      os << "    fold_cost: " << h.fold_cost << "\n";
      os << "    elbow_mirror_err_m: " << h.elbow_err << "\n";
      os << "    j12_vs_geo_mirror_rad: " << h.j12_err << "\n";
      os << "    fk_position_error_m: " << h.fk_pos << "\n";
      os << "    fk_orientation_error_deg: " << h.fk_ori << "\n";
    }
    os << "  unique_groups:\n";
    std::map<std::string, int> g;
    for (const auto& h : r.all)
      g[configSig(h.joints)]++;
    for (const auto& kv : g)
      os << "    - {sig: \"" << kv.first << "\", count: " << kv.second << "}\n";
  };
  dumpFace("B_FACE4", r4);
  dumpFace("B_FACE5", r5);
  os << "cannot_prove_unreachable: true\n";
  os.close();
  emit("wrote " + out_path);

  if (!r4.legal.empty() || !r5.legal.empty())
  {
    YAML::Node arm_b = YAML::LoadFile(out_dir + "/arm_b_candidates.yaml");
    std::ifstream in(out_dir + "/arm_b_candidates.yaml");
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    auto replaceBlock = [&](const std::string& key, const auto& r) {
      const std::string start = "  " + key + ":";
      auto p = text.find(start);
      if (p == std::string::npos)
        return;
      auto n = text.find("\n  B_", p + 1);
      if (n == std::string::npos)
        n = text.find("\n  A_", p + 1);
      if (n == std::string::npos)
        n = text.size();
      std::ostringstream blk;
      blk << "  " << key << ":\n";
      if (r.legal.empty())
      {
        blk << "    legal: false\n    note: \"NO LEGAL CANDIDATE\"\n    candidates: []\n";
      }
      else
      {
        blk << "    legal: true\n    note: \"YZ-mirror of A, full face pose\"\n    candidates:\n";
        int idx = 1;
        for (const auto& h : r.legal)
        {
          blk << "    - index: " << idx++ << "\n";
          blk << "      seed: \"" << h.seed << "\"\n";
          blk << "      baseline: false\n";
          blk << "      joint_values: " << fmt(h.joints, 12) << "\n";
          const auto q = quatOf(h.tcp);
          blk << "      task_space_pose:\n";
          blk << "        xyz: " << fmt({h.tcp.translation().x(), h.tcp.translation().y(),
                                         h.tcp.translation().z()},
                                        12)
              << "\n";
          blk << "        xyzw: " << fmt({q.x(), q.y(), q.z(), q.w()}, 12) << "\n";
          blk << "      cost: " << h.fold_cost << "\n";
          blk << "      collision_check: PASS\n";
          blk << "      joint_limit_check: PASS\n";
          blk << "      joint_limit_margin: " << h.margin << "\n";
          blk << "      fk_ok: true\n";
          blk << "      fk_position_error_m: " << h.fk_pos << "\n";
          blk << "      fk_orientation_error_deg: " << h.fk_ori << "\n";
          blk << "      face_center: "
              << fmt({h.face_center.x(), h.face_center.y(), h.face_center.z()}, 12) << "\n";
          blk << "      face_normal: "
              << fmt({h.face_normal.x(), h.face_normal.y(), h.face_normal.z()}, 12) << "\n";
          blk << "      face_up: " << fmt({h.face_up.x(), h.face_up.y(), h.face_up.z()}, 12) << "\n";
          blk << "      face_center_error_m: " << h.face_c << "\n";
          blk << "      face_normal_error_deg: " << h.face_n << "\n";
          blk << "      face_up_error_deg: " << h.face_u << "\n";
          blk << "      face_pose_check: PASS\n";
        }
      }
      text.replace(p, n - p, blk.str());
    };
    replaceBlock("B_FACE4", r4);
    replaceBlock("B_FACE5", r5);
    std::ofstream ao(out_dir + "/arm_b_candidates.yaml");
    ao << text;
    emit("updated arm_b_candidates.yaml B_FACE4/B_FACE5 only");
  }

  rclcpp::shutdown();
  return 0;
}
