// B_FACE4/5: local IK from DUAL-7 I1/I2, J1≈5° J2≈-47° ±10°. No mirror TCP.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
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

const std::vector<std::string> kArmA = {"arm_a_j1", "arm_a_j2", "arm_a_j3",
                                        "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kArmB = {"arm_b_j1", "arm_b_j2", "arm_b_j3",
                                        "arm_b_j4", "arm_b_j5", "arm_b_j6"};
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
  return std::acos(std::max(-1.0, std::min(1.0, a.dot(b) / (na * nb)))) * 180.0 / M_PI;
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

std::vector<double> wrapJoints(const moveit::core::RobotModelConstPtr& model,
                               const std::vector<std::string>& names, std::vector<double> q)
{
  for (size_t i = 0; i < names.size() && i < q.size(); ++i)
  {
    const auto* j = model->getJointModel(names[i]);
    if (!j || j->getVariableBounds().empty() || !j->getVariableBounds().front().position_bounded_)
      continue;
    const double lo = j->getVariableBounds().front().min_position_;
    const double hi = j->getVariableBounds().front().max_position_;
    for (int k = 0; k < 6; ++k)
    {
      if (q[i] >= lo - 1e-9 && q[i] <= hi + 1e-9)
        break;
      if (q[i] > hi)
        q[i] -= 2.0 * M_PI;
      else if (q[i] < lo)
        q[i] += 2.0 * M_PI;
    }
    q[i] = std::min(hi, std::max(lo, q[i]));
  }
  return q;
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
    if (allowedPair(c.first.first, c.first.second))
      continue;
    pair = c.first.first + " <-> " + c.first.second;
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

Eigen::Isometry3d rotateAboutPoint(const Eigen::Isometry3d& T, const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& axis, double ang)
{
  Eigen::Isometry3d R = Eigen::Isometry3d::Identity();
  R.linear() = Eigen::AngleAxisd(ang, axis.normalized()).toRotationMatrix();
  Eigen::Isometry3d W = Eigen::Isometry3d::Identity();
  W.translation() = p;
  return W * R * W.inverse() * T;
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

struct FaceHit
{
  bool ok = false;
  bool collision_ok = false;
  std::string pair = "none";
  std::string block = "none";
  std::vector<double> joints;
  Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d obj = Eigen::Isometry3d::Identity();
  Eigen::Vector3d face_c = Eigen::Vector3d::Zero();
  Eigen::Vector3d face_n = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  double face_c_err = 0;
  double face_n_err = 0;
  double axis_up_err = 0;
  double fk_pos = 0;
  double fk_ori = 0;
  double margin = 0;
  double broke_t = 1;
};

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("keypose_b_face4_solve", opt);
  emit("========== B_FACE4/5 OPEN-FAMILY J1≈5° J2≈-47° ==========");
  emit("Window ±10 deg. No mirror TCP. No random grid. Collision not relaxed.");

  const std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string search_yaml = node->get_parameter_or(
      "search_yaml", home_dir + "/fr_task_ws/src/fr_task_planner/config/"
                                "keypose_optimization_v1/search.yaml");
  YAML::Node cfg = YAML::LoadFile(search_yaml);
  const std::string out_dir = node->get_parameter_or("output_dir", cfg["output_dir"].as<std::string>());
  YAML::Node home_y = YAML::LoadFile(cfg["home_yaml"].as<std::string>());
  YAML::Node work = YAML::LoadFile(cfg["workcell_yaml"].as<std::string>());
  YAML::Node six = YAML::LoadFile(cfg["six_face_yaml"].as<std::string>());
  YAML::Node tgt_y = YAML::LoadFile(out_dir + "/task_space_targets.yaml");
  YAML::Node s12 = YAML::LoadFile(cfg["step12c_winner"].as<std::string>());

  std::vector<double> home_a, i1, i2;
  yamlVec(home_y["home_a"], home_a);
  yamlVec(six["i1_b"], i1);
  yamlVec(six["i2_b"], i2);
  Eigen::Isometry3d T_tcpB_obj = Eigen::Isometry3d::Identity();
  loadXyzw(tgt_y["T_tcpB_object"]["pose"], T_tcpB_obj);
  const Eigen::Vector3d p1(s12["p1"][0].as<double>(), s12["p1"][1].as<double>(),
                           s12["p1"][2].as<double>());
  Eigen::Vector3d d1(s12["surface_target_normal"][0].as<double>(),
                     s12["surface_target_normal"][1].as<double>(),
                     s12["surface_target_normal"][2].as<double>());
  d1.normalize();
  const Eigen::Vector3d up_w(0.0, 0.707106781, 0.707106781);
  const Eigen::Vector3d cam_fwd(0.0, 0.707106781, -0.707106781);
  const double qa_open = 0.0;
  const double qb_hold = 0.083;
  const double radius = 0.0075;
  const double height = 0.035;
  const Eigen::Vector3d c4(0.0075, 0, 0), n4(1, 0, 0), c5(-0.0075, 0, 0), n5(-1, 0, 0);

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

  moveit::core::RobotState st0(model);
  st0.setToDefaultValues();
  setJoints(st0, kArmA, home_a);
  setJoints(st0, kArmB, i1);
  st0.update();
  scene->setCurrentState(st0);

  auto objOf = [&](const std::vector<double>& qb) {
    moveit::core::RobotState s(scene->getCurrentState());
    setJoints(s, kArmB, qb);
    s.update();
    const Eigen::Isometry3d tcp = s.getGlobalLinkTransform(kTcpB);
    return std::pair<Eigen::Isometry3d, Eigen::Isometry3d>(tcp, tcp * T_tcpB_obj);
  };
  auto axisOf = [](const Eigen::Isometry3d& obj) { return obj.linear().col(2).normalized(); };
  auto imageTilt = [&](const Eigen::Vector3d& axis) {
    Eigen::Vector3d ai = axis - cam_fwd * axis.dot(cam_fwd);
    if (ai.norm() < 1e-9)
      return 180.0;
    ai.normalize();
    const double u = ai.x();
    const double vup = ai.dot(up_w);
    return std::atan2(u, vup) * 180.0 / M_PI;
  };

  const auto [tcp_i1, obj_i1] = objOf(i1);
  const auto [tcp_i2, obj_i2] = objOf(i2);
  const Eigen::Vector3d ax1 = axisOf(obj_i1);
  const Eigen::Vector3d ax2 = axisOf(obj_i2);
  emit("I1 joints deg " + deg6(i1));
  emit("I2 joints deg " + deg6(i2));
  emit("I1 cylinder +Z world " + fmt({ax1.x(), ax1.y(), ax1.z()}, 6));
  emit("I1 axis·D1 " + std::to_string(ax1.dot(d1)) + " (0 => fully in image plane)");
  emit("I1 3D ang vs preferred-up " + std::to_string(angDeg(ax1, up_w)));
  emit("I1 image tilt from vertical-up deg " + std::to_string(imageTilt(ax1)));
  emit("I2 image tilt from vertical-up deg " + std::to_string(imageTilt(ax2)));
  emit("STEP12C model camera used (camera_calibrated: false). Not a measured extrinsic.");

  const double need = -imageTilt(ax1) * M_PI / 180.0;
  const double j1_tgt = 5.0 * M_PI / 180.0;
  const double j2_tgt = -47.0 * M_PI / 180.0;
  const double j_tol = 10.0 * M_PI / 180.0;
  emit("D1-roll to align +Z with preferred-up deg " + std::to_string(need * 180 / M_PI));
  emit("J1 target 5 deg, J2 target -47 deg, window ±10 deg");

  auto fillHit = [&](FaceHit& h, moveit::core::RobotState& s, const Eigen::Isometry3d& target,
                     const Eigen::Vector3d& c_obj, const Eigen::Vector3d& n_obj) {
    h.joints = wrapJoints(model, kArmB, jointsOf(s, kArmB));
    setJoints(s, kArmB, h.joints);
    s.update();
    h.tcp = s.getGlobalLinkTransform(kTcpB);
    h.obj = h.tcp * T_tcpB_obj;
    poseError(target, h.tcp, h.fk_pos, h.fk_ori);
    h.face_c = h.obj * c_obj;
    h.face_n = (h.obj.linear() * n_obj).normalized();
    h.axis = axisOf(h.obj);
    h.face_c_err = (h.face_c - p1).norm();
    h.face_n_err = angDeg(h.face_n, d1);
    h.axis_up_err = angDeg(h.axis, up_w);
    h.margin = limitMargin(s, gb);
  };

  const bool only_face6 =
      node->has_parameter("only_face6") && node->get_parameter("only_face6").as_bool();
  if (only_face6)
  {
    emit("========== B_FACE6 LOCAL FROM B_FACE5 ==========");
    YAML::Node bcan = YAML::LoadFile(out_dir + "/arm_b_candidates.yaml");
    std::vector<double> q5;
    if (!yamlVec(bcan["keyposes"]["B_FACE5"]["candidates"][0]["joint_values"], q5) || q5.size() < 6)
    {
      emit("B_FACE5 candidate missing");
      rclcpp::shutdown();
      return 2;
    }
    Eigen::Isometry3d obj_can;
    if (!loadXyzw(tgt_y["B_FACE6"]["object"], obj_can))
    {
      emit("B_FACE6 object target missing");
      rclcpp::shutdown();
      return 2;
    }
    const Eigen::Vector3d c6(0, 0, 0.0175), n6(0, 0, 1), u6(0, -1, 0);
    const double wj[6] = {8.0, 8.0, 1.0, 1.0, 1.0, 0.2};
    emit("B_FACE5 deg " + deg6(q5));

    auto costOf = [&](const std::vector<double>& q) {
      double c = 0;
      for (int j = 0; j < 6; ++j)
        c += wj[j] * std::abs(q[j] - q5[j]);
      return c;
    };
    auto j12 = [&](const std::vector<double>& q) {
      return std::abs(q[0] - q5[0]) + std::abs(q[1] - q5[1]);
    };

    auto dls = [&](const Eigen::Isometry3d& tcp, std::vector<double> q0, bool lock12) {
      moveit::core::RobotState st(scene->getCurrentState());
      setJoints(st, kArmA, home_a);
      setJoints(st, kArmB, wrapJoints(model, kArmB, q0));
      st.update();
      const auto names = gb->getVariableNames();
      for (int it = 0; it < 60; ++it)
      {
        st.update();
        const Eigen::Isometry3d T = st.getGlobalLinkTransform(kTcpB);
        Eigen::Vector3d dp = tcp.translation() - T.translation();
        Eigen::Quaterniond qa(tcp.linear()), qb(T.linear());
        qa.normalize();
        qb.normalize();
        Eigen::AngleAxisd aa(qa * qb.inverse());
        Eigen::Vector3d dw = aa.angle() * aa.axis();
        if (dp.norm() < 4e-4 && std::abs(aa.angle()) < 0.012)
          break;
        Eigen::MatrixXd J;
        st.getJacobian(gb, st.getLinkModel(kTcpB), Eigen::Vector3d::Zero(), J);
        Eigen::VectorXd err(6);
        err << dp, 0.6 * dw;
        const double lam = 0.05;
        Eigen::MatrixXd A = J.transpose() * J;
        A.diagonal().array() += lam * lam;
        Eigen::VectorXd dq = A.ldlt().solve(J.transpose() * err);
        Eigen::VectorXd ns(6);
        for (int j = 0; j < 6; ++j)
          ns[j] = -wj[j] * (st.getVariablePosition(names[j]) - q5[j]);
        Eigen::VectorXd dq_ns = A.ldlt().solve(J.transpose() * (J * ns));
        dq += 0.06 * (ns - dq_ns);
        if (lock12)
        {
          dq[0] = 0;
          dq[1] = 0;
        }
        if (dq.norm() > 0.18)
          dq *= 0.18 / dq.norm();
        for (int j = 0; j < 6; ++j)
          st.setVariablePosition(names[j], st.getVariablePosition(names[j]) + dq[j]);
        st.enforceBounds(gb);
      }
      st.update();
      if (!lock12)
        st.setFromIK(gb, tcp, kTcpB, 0.08);
      return jointsOf(st, kArmB);
    };

    FaceHit best;
    best.block = "no_ik";
    auto consider = [&](const std::string& tag, const Eigen::Isometry3d& tcp, std::vector<double> q) {
      q = wrapJoints(model, kArmB, q);
      moveit::core::RobotState st(scene->getCurrentState());
      setJoints(st, kArmA, home_a);
      setJoints(st, kArmB, q);
      st.update();
      FaceHit h;
      h.joints = q;
      h.tcp = st.getGlobalLinkTransform(kTcpB);
      h.obj = h.tcp * T_tcpB_obj;
      poseError(tcp, h.tcp, h.fk_pos, h.fk_ori);
      h.face_c = h.obj * c6;
      h.face_n = (h.obj.linear() * n6).normalized();
      h.axis = (h.obj.linear() * u6).normalized();
      h.face_c_err = (h.face_c - p1).norm();
      h.face_n_err = angDeg(h.face_n, d1);
      h.axis_up_err = angDeg(h.axis, up_w);
      h.margin = limitMargin(st, gb);
      const bool pose_ok = h.fk_pos <= 0.003 && h.fk_ori <= 2.0 && h.face_c_err <= 0.002 &&
                           h.face_n_err <= 3.0 && st.satisfiesBounds(gb);
      attachCylinder(*scene, T_tcpB_obj, radius, height);
      applyState(*scene, home_a, h.joints, qa_open, qb_hold);
      auto cs = scene->getCurrentStateNonConst();
      h.collision_ok = !colliding(*scene, cs, h.pair);
      h.ok = pose_ok && h.collision_ok;
      h.block = h.ok ? "none" : (!pose_ok ? "pose_tol" : ("collision " + h.pair));
      h.broke_t = costOf(h.joints);
      const bool better = h.ok && (!best.ok || j12(h.joints) + 1e-4 < j12(best.joints) ||
                                   (std::abs(j12(h.joints) - j12(best.joints)) < 1e-4 &&
                                    h.broke_t < best.broke_t));
      if (better || (!best.ok && h.ok))
        best = h;
      if (h.ok)
        emit("  " + tag + " " + deg6(h.joints) + " dj12deg=" +
             std::to_string(j12(h.joints) * 180.0 / M_PI) + " up=" +
             std::to_string(h.axis_up_err));
    };

    const double rolls[] = {0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 180};
    std::vector<std::vector<double>> seeds;
    seeds.push_back(q5);
    for (double o : {M_PI / 2, -M_PI / 2, M_PI, -M_PI})
    {
      auto q = q5;
      q[5] += o;
      seeds.push_back(q);
    }
    for (double o : {0.4, -0.4, 0.8, -0.8})
    {
      auto q = q5;
      q[2] += o;
      seeds.push_back(q);
      q = q5;
      q[3] += o;
      seeds.push_back(q);
      q = q5;
      q[4] += o;
      seeds.push_back(q);
    }
    for (double deg : rolls)
    {
      const Eigen::Isometry3d obj = rotateAboutPoint(obj_can, p1, d1, deg * M_PI / 180.0);
      const Eigen::Isometry3d tcp = obj * T_tcpB_obj.inverse();
      for (const auto& seed : seeds)
      {
        consider("lock " + std::to_string(deg), tcp, dls(tcp, seed, true));
        consider("free " + std::to_string(deg), tcp, dls(tcp, seed, false));
        moveit::core::RobotState st(scene->getCurrentState());
        setJoints(st, kArmA, home_a);
        setJoints(st, kArmB, wrapJoints(model, kArmB, seed));
        st.update();
        if (st.setFromIK(gb, tcp, kTcpB, 0.05))
          consider("kdl " + std::to_string(deg), tcp, jointsOf(st, kArmB));
        if (best.ok && j12(best.joints) < 2.0 * M_PI / 180.0 && best.axis_up_err < 8.0)
          break;
      }
      if (best.ok && j12(best.joints) < 2.0 * M_PI / 180.0 && best.axis_up_err < 8.0)
        break;
    }

    emit(std::string("B_FACE6 legal=") + (best.ok ? "true" : "false") + " " + best.block);
    if (!best.joints.empty())
      emit("best deg " + deg6(best.joints) + " dj12deg=" +
           std::to_string(j12(best.joints) * 180.0 / M_PI) + " cost=" +
           std::to_string(best.broke_t));

    std::ifstream in(out_dir + "/arm_b_candidates.yaml");
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    auto pos6 = text.find("  B_FACE6:");
    if (pos6 != std::string::npos && best.ok)
    {
      std::ostringstream b;
      b.setf(std::ios::fixed);
      b << std::setprecision(12);
      b << "  B_FACE6:\n";
      b << "    legal: true\n";
      b << "    note: \"local from B_FACE5, J1/J2 minimized\"\n";
      b << "    candidates:\n    - index: 1\n      seed: \"B_FACE5\"\n      baseline: false\n";
      b << "      joint_values: [";
      for (size_t i = 0; i < best.joints.size(); ++i)
      {
        if (i)
          b << ", ";
        b << best.joints[i];
      }
      b << "]\n";
      const auto qq = quatOf(best.tcp);
      b << "      task_space_pose:\n        xyz: [" << best.tcp.translation().x() << ", "
        << best.tcp.translation().y() << ", " << best.tcp.translation().z() << "]\n";
      b << "        xyzw: [" << qq.x() << ", " << qq.y() << ", " << qq.z() << ", " << qq.w()
        << "]\n";
      b << "      collision_check: PASS\n      joint_limit_check: PASS\n";
      b << "      face_center_error_m: " << best.face_c_err << "\n";
      b << "      face_normal_error_deg: " << best.face_n_err << "\n";
      b << "      face_up_error_deg: " << best.axis_up_err << "\n";
      b << "      face_pose_check: PASS\n";
      b << "      delta_from_B_FACE5_deg: [";
      for (int j = 0; j < 6; ++j)
      {
        if (j)
          b << ", ";
        b << ((best.joints[j] - q5[j]) * 180.0 / M_PI);
      }
      b << "]\n";
      text.replace(pos6, std::string::npos, b.str());
      std::ofstream outb(out_dir + "/arm_b_candidates.yaml");
      outb << text;
      emit("updated arm_b_candidates.yaml B_FACE6 only");
    }
    rclcpp::shutdown();
    return best.ok ? 0 : 2;
  }

  auto j1err = [&](const std::vector<double>& q) {
    return q.empty() ? 1e9 : std::abs(q[0] - j1_tgt);
  };
  auto j2err = [&](const std::vector<double>& q) {
    return q.size() < 2 ? 1e9 : std::abs(q[1] - j2_tgt);
  };
  auto j12ok = [&](const std::vector<double>& q) { return j1err(q) <= j_tol && j2err(q) <= j_tol; };
  auto j12sum = [&](const std::vector<double>& q) { return j1err(q) + j2err(q); };

  auto better = [&](const FaceHit& a, const FaceHit& b) {
    if (a.ok != b.ok)
      return a.ok;
    if (a.joints.empty())
      return false;
    if (b.joints.empty())
      return true;
    const double dj = j12sum(a.joints) - j12sum(b.joints);
    if (std::abs(dj) > 1e-4)
      return dj < 0;
    return a.axis_up_err < b.axis_up_err;
  };

  auto trySeed = [&](const Eigen::Isometry3d& tcp, const std::vector<double>& seed,
                     const Eigen::Vector3d& c_obj, const Eigen::Vector3d& n_obj) {
    FaceHit h;
    h.block = "ik_fail";
    auto runIk = [&](const std::vector<double>& q0, double tmo) {
      moveit::core::RobotState st(scene->getCurrentState());
      setJoints(st, kArmA, home_a);
      setJoints(st, kArmB, wrapJoints(model, kArmB, q0));
      st.update();
      if (!st.setFromIK(gb, tcp, kTcpB, tmo))
        return false;
      fillHit(h, st, tcp, c_obj, n_obj);
      return true;
    };
    if (!runIk(seed, 0.22))
      return h;
    if (!j12ok(h.joints))
    {
      auto snap = h.joints;
      snap[0] = j1_tgt;
      snap[1] = j2_tgt;
      runIk(snap, 0.18);
    }
    attachCylinder(*scene, T_tcpB_obj, radius, height);
    applyState(*scene, home_a, h.joints, qa_open, qb_hold);
    auto cs = scene->getCurrentStateNonConst();
    h.collision_ok = !colliding(*scene, cs, h.pair);
    moveit::core::RobotState chk(scene->getCurrentState());
    setJoints(chk, kArmB, h.joints);
    chk.update();
    const bool pose_ok = h.fk_pos <= 0.003 && h.fk_ori <= 2.0 && h.face_c_err <= 0.002 &&
                         h.face_n_err <= 3.0 && chk.satisfiesBounds(gb);
    h.ok = pose_ok && h.collision_ok && j12ok(h.joints);
    if (!h.ok)
    {
      if (!pose_ok)
        h.block = "pose_tol";
      else if (!h.collision_ok)
        h.block = "collision " + h.pair;
      else
        h.block = "j1j2_out_of_window";
    }
    else
      h.block = "none";
    return h;
  };

  auto openSeeds = [&](const std::vector<double>& base) {
    std::vector<std::vector<double>> seeds;
    auto add = [&](double j1, double j2, std::vector<double> q) {
      q[0] = j1;
      q[1] = j2;
      seeds.push_back(wrapJoints(model, kArmB, q));
    };
    add(j1_tgt, j2_tgt, base);
    auto q = base;
    q[2] = -q[2];
    add(j1_tgt, j2_tgt, q);
    q = base;
    q[2] = -1.360382;
    add(j1_tgt, j2_tgt, q);
    const double j6s[] = {0.0, M_PI / 2, -M_PI / 2, M_PI};
    for (double o : j6s)
    {
      q = base;
      q[5] += o;
      add(j1_tgt, j2_tgt, q);
    }
    add(0.0, j2_tgt, base);
    add(10.0 * M_PI / 180.0, j2_tgt, base);
    add(j1_tgt, -40.0 * M_PI / 180.0, base);
    add(j1_tgt, -54.0 * M_PI / 180.0, base);
    return seeds;
  };

  auto solveJ12 = [&](const std::string& name, const Eigen::Isometry3d& tcp,
                      const Eigen::Vector3d& c_obj, const Eigen::Vector3d& n_obj,
                      const std::vector<double>& base) {
    emit("--- " + name + " ---");
    FaceHit best;
    best.block = "no_ik";
    for (const auto& seed : openSeeds(base))
    {
      FaceHit h = trySeed(tcp, seed, c_obj, n_obj);
      emit("  try " + deg6(h.joints) +
           (h.joints.size() < 2
                ? std::string(" IK_FAIL")
                : " j1=" + std::to_string(h.joints[0] * 180.0 / M_PI) +
                      " j2=" + std::to_string(h.joints[1] * 180.0 / M_PI)) +
           " col=" + (h.collision_ok ? std::string("PASS") : h.pair) + " " + h.block);
      if (better(h, best))
        best = h;
    }
    emit("  best j1deg=" +
         std::to_string(best.joints.empty() ? 0.0 : best.joints[0] * 180.0 / M_PI) + " j2deg=" +
         std::to_string(best.joints.size() < 2 ? 0.0 : best.joints[1] * 180.0 / M_PI) +
         " legal=" + std::string(best.ok ? "true" : "false") + " " + best.block);
    return best;
  };

  const double rolls[] = {0.0, 1.0, 0.5, 0.75, 0.25, -0.25, 1.25};
  auto solveRolls = [&](const std::string& name, const Eigen::Isometry3d& obj0,
                        const Eigen::Vector3d& c_obj, const Eigen::Vector3d& n_obj,
                        const std::vector<double>& base) {
    FaceHit best;
    best.block = "no_ik";
    for (double f : rolls)
    {
      const Eigen::Isometry3d obj = rotateAboutPoint(obj0, p1, d1, need * f);
      const Eigen::Isometry3d tcp = obj * T_tcpB_obj.inverse();
      FaceHit h = solveJ12(name + " roll_frac=" + std::to_string(f), tcp, c_obj, n_obj, base);
      h.broke_t = f;
      if (better(h, best))
        best = h;
      if (best.ok && j1err(best.joints) <= 5.0 * M_PI / 180.0 &&
          j2err(best.joints) <= 5.0 * M_PI / 180.0 && best.axis_up_err <= 8.0)
        break;
    }
    return best;
  };

  const Eigen::Isometry3d tcp4_al = rotateAboutPoint(obj_i1, p1, d1, need) * T_tcpB_obj.inverse();
  const Eigen::Isometry3d tcp5_al = rotateAboutPoint(obj_i2, p1, d1, need) * T_tcpB_obj.inverse();
  FaceHit r4 = solveJ12("B_FACE4 aligned J1=5 J2=-47", tcp4_al, c4, n4, i1);
  if (!r4.ok)
    r4 = solveJ12("B_FACE4 I1-TCP J1=5 J2=-47", tcp_i1, c4, n4, i1);
  if (!r4.ok)
    r4 = solveRolls("B_FACE4", obj_i1, c4, n4, i1);

  FaceHit r5;
  if (r4.ok)
  {
    auto q5 = r4.joints;
    q5[5] += M_PI;
    q5 = wrapJoints(model, kArmB, q5);
    moveit::core::RobotState st(scene->getCurrentState());
    setJoints(st, kArmA, home_a);
    setJoints(st, kArmB, q5);
    st.update();
    const Eigen::Isometry3d tcp5_now = st.getGlobalLinkTransform(kTcpB);
    fillHit(r5, st, tcp5_now, c5, n5);
    attachCylinder(*scene, T_tcpB_obj, radius, height);
    applyState(*scene, home_a, r5.joints, qa_open, qb_hold);
    auto cs = scene->getCurrentStateNonConst();
    r5.collision_ok = !colliding(*scene, cs, r5.pair);
    r5.ok = r5.face_c_err <= 0.002 && r5.face_n_err <= 3.0 && r5.collision_ok && j12ok(r5.joints);
    emit("B_FACE5 J6+180 " + deg6(r5.joints) + " ok=" + std::to_string(r5.ok) +
         " j1=" + std::to_string(r5.joints[0] * 180.0 / M_PI) +
         " j2=" + std::to_string(r5.joints[1] * 180.0 / M_PI) +
         " axis=" + std::to_string(r5.axis_up_err));
    if (!r5.ok)
      r5.block = r5.collision_ok ? "j6_flip_pose" : ("collision " + r5.pair);
    else
      r5.block = "none";
  }
  if (!r5.ok)
    r5 = solveJ12("B_FACE5 aligned J1=5 J2=-47", tcp5_al, c5, n5, i2);
  if (!r5.ok)
    r5 = solveJ12("B_FACE5 I2-TCP J1=5 J2=-47", tcp_i2, c5, n5, i2);
  if (!r5.ok)
    r5 = solveRolls("B_FACE5", obj_i2, c5, n5, i2);

  const std::string out_path = out_dir + "/b_face45_axis_align.yaml";
  std::ofstream os(out_path);
  os << "task_version: KEYPOSE_B_FACE45_AXIS_ALIGN\n";
  os << "execution: false\n";
  os << "camera_source: STEP12C model camera_rpy, camera_calibrated=false\n";
  os << "i1_image_tilt_from_vertical_up_deg: " << imageTilt(ax1) << "\n";
  os << "i2_image_tilt_from_vertical_up_deg: " << imageTilt(ax2) << "\n";
  os << "axis_in_image_plane: true\n";
  os << "face_up_is_cylinder_axis_for_side_faces: true\n";
  os << "case: B  # real pose tilt, not a definition-only error\n";
  os << "d1_roll_applied_deg: " << (need * 180.0 / M_PI) << "\n";
  os << "j1_target_deg: 5\n";
  os << "j2_target_deg: -47\n";
  os << "j12_window_deg: 10\n";
  writeVec(os, "", "p1", {p1.x(), p1.y(), p1.z()});
  writeVec(os, "", "d1", {d1.x(), d1.y(), d1.z()});
  writeVec(os, "", "up_world", {up_w.x(), up_w.y(), up_w.z()});
  writeVec(os, "", "home_a", home_a);
  os << "I1:\n";
  writeVec(os, "  ", "joints", i1);
  writePose(os, "  ", "tcp", tcp_i1);
  writePose(os, "  ", "object", obj_i1);
  writeVec(os, "  ", "axis", {ax1.x(), ax1.y(), ax1.z()});
  os << "I2:\n";
  writeVec(os, "  ", "joints", i2);
  writePose(os, "  ", "tcp", tcp_i2);
  writePose(os, "  ", "object", obj_i2);
  writeVec(os, "  ", "axis", {ax2.x(), ax2.y(), ax2.z()});
  auto dump = [&](const char* key, const FaceHit& h) {
    os << key << ":\n";
    os << "  legal: " << (h.ok ? "true" : "false") << "\n";
    os << "  block: \"" << h.block << "\"\n";
    writeVec(os, "  ", "joints", h.joints);
    os << "  joints_deg: " << (h.joints.empty() ? "[]" : deg6(h.joints)) << "\n";
    if (!h.joints.empty())
    {
      os << "  j1_deg: " << (h.joints[0] * 180.0 / M_PI) << "\n";
      os << "  j2_deg: " << (h.joints[1] * 180.0 / M_PI) << "\n";
      writePose(os, "  ", "tcp", h.tcp);
      writePose(os, "  ", "object", h.obj);
      writeVec(os, "  ", "face_center", {h.face_c.x(), h.face_c.y(), h.face_c.z()});
      writeVec(os, "  ", "face_normal", {h.face_n.x(), h.face_n.y(), h.face_n.z()});
      writeVec(os, "  ", "axis", {h.axis.x(), h.axis.y(), h.axis.z()});
    }
    os << "  face_center_error_m: " << h.face_c_err << "\n";
    os << "  face_normal_error_deg: " << h.face_n_err << "\n";
    os << "  axis_vs_preferred_up_deg: " << h.axis_up_err << "\n";
    os << "  collision_ok: " << (h.collision_ok ? "true" : "false") << "\n";
    os << "  collision_pair: \"" << h.pair << "\"\n";
    os << "  broke_t: " << h.broke_t << "\n";
  };
  dump("B_FACE4", r4);
  dump("B_FACE5", r5);
  os << "j6_only_switch: "
     << (r4.ok && r5.ok && r4.joints.size() == 6 && r5.joints.size() == 6 &&
                 std::hypot(r4.joints[0] - r5.joints[0], r4.joints[1] - r5.joints[1]) < 0.05 &&
                 std::abs(std::abs(r4.joints[5] - r5.joints[5]) - M_PI) < 0.2
             ? "true"
             : "false")
     << "\n";
  os << "rviz_command: ros2 launch fr_task_planner keypose_b_face4_rviz.launch.py\n";
  os.close();
  emit("wrote " + out_path);
  emit(std::string("B_FACE4 legal=") + (r4.ok ? "true" : "false") + " block=" + r4.block);
  emit(std::string("B_FACE5 legal=") + (r5.ok ? "true" : "false") + " block=" + r5.block);

  if (r4.ok)
  {
    std::ifstream in(out_dir + "/arm_b_candidates.yaml");
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    auto pos4 = text.find("  B_FACE4:");
    auto pos5 = text.find("  B_FACE5:");
    auto pos6 = text.find("  B_FACE6:");
    auto one = [&](const FaceHit& h, const char* name, const char* seed) {
      std::ostringstream b;
      b.setf(std::ios::fixed);
      b << std::setprecision(12);
      b << "  " << name << ":\n";
      b << "    legal: " << (h.ok ? "true" : "false") << "\n";
      if (!h.ok)
      {
        b << "    note: \"" << h.block << "\"\n";
        b << "    candidates: []\n";
        return b.str();
      }
      b << "    note: \"open-family local IK J1≈5° J2≈-47° ±10° from DUAL-7 I1/I2\"\n";
      b << "    candidates:\n    - index: 1\n      seed: \"" << seed << "\"\n      baseline: false\n";
      b << "      joint_values: [";
      for (size_t i = 0; i < h.joints.size(); ++i)
      {
        if (i)
          b << ", ";
        b << h.joints[i];
      }
      b << "]\n";
      const auto q = quatOf(h.tcp);
      b << "      task_space_pose:\n        xyz: [" << h.tcp.translation().x() << ", "
        << h.tcp.translation().y() << ", " << h.tcp.translation().z() << "]\n";
      b << "        xyzw: [" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]\n";
      b << "      collision_check: PASS\n      joint_limit_check: PASS\n";
      b << "      face_center_error_m: " << h.face_c_err << "\n";
      b << "      face_normal_error_deg: " << h.face_n_err << "\n";
      b << "      face_up_error_deg: " << h.axis_up_err << "\n";
      b << "      face_pose_check: PASS\n";
      return b.str();
    };
    if (pos4 != std::string::npos && pos5 != std::string::npos && pos6 != std::string::npos)
    {
      std::string block = one(r4, "B_FACE4", "j1_5_j2_-47") + one(r5, "B_FACE5", "j1_5_j2_-47");
      text.replace(pos4, pos6 - pos4, block);
      std::ofstream outb(out_dir + "/arm_b_candidates.yaml");
      outb << text;
      emit("updated arm_b_candidates.yaml B_FACE4/B_FACE5 only");
    }
  }

  rclcpp::shutdown();
  return (r4.ok && r5.ok) ? 0 : 2;
}
