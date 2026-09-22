// One B_HOME above the confirmed placement hover. Does not replan the rest of the task.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometric_shapes/shapes.h>
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
const std::vector<std::string> kA = {"arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kB = {"arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"};
const std::vector<std::string> kTouchA = {
    "arm_a_gripper_base_link", "arm_a_finger_l", "arm_a_finger_r", "arm_a_gripper_tcp"};
constexpr char kPart[] = "small_part";
constexpr char kTcp[] = "arm_b_gripper_tcp";
constexpr double kTableTop = 0.750;
constexpr double kStep = 0.02;

void emit(const std::string& s) { std::cout << s << std::endl; }

bool vec6(const YAML::Node& n, std::vector<double>& o)
{
  if (!n || !n.IsSequence() || n.size() < 6)
    return false;
  o.clear();
  for (int i = 0; i < 6; ++i)
    o.push_back(n[i].as<double>());
  return true;
}

geometry_msgs::msg::Pose poseOf(const Eigen::Isometry3d& T)
{
  geometry_msgs::msg::Pose p;
  p.position.x = T.translation().x();
  p.position.y = T.translation().y();
  p.position.z = T.translation().z();
  Eigen::Quaterniond q(T.linear());
  q.normalize();
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
  return p;
}

void setJ(moveit::core::RobotState& st, const std::vector<std::string>& n, const std::vector<double>& q)
{
  for (size_t i = 0; i < n.size() && i < q.size(); ++i)
    st.setVariablePosition(n[i], q[i]);
}

bool limitsOk(const moveit::core::RobotState& st)
{
  for (const auto& n : kB)
  {
    const auto* j = st.getRobotModel()->getJointModel(n);
    if (!j || j->getVariableBounds().empty() || !j->getVariableBounds().front().position_bounded_)
      continue;
    const double q = st.getVariablePosition(n);
    if (q < j->getVariableBounds().front().min_position_ - 1e-6 ||
        q > j->getVariableBounds().front().max_position_ + 1e-6)
      return false;
  }
  return true;
}

bool hit(planning_scene::PlanningScene& scene, moveit::core::RobotState& st, std::string& pair)
{
  st.update();
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 20;
  req.max_contacts_per_pair = 1;
  scene.checkCollision(req, res, st);
  if (!res.collision)
    return false;
  for (const auto& c : res.contacts)
  {
    const auto& a = c.first.first;
    const auto& b = c.first.second;
    const bool part_a = (a == kPart && std::find(kTouchA.begin(), kTouchA.end(), b) != kTouchA.end()) ||
                        (b == kPart && std::find(kTouchA.begin(), kTouchA.end(), a) != kTouchA.end());
    const bool base = (a == "arm_a_base_link" && b == "mounting_column") ||
                      (b == "arm_a_base_link" && a == "mounting_column") ||
                      (a == "arm_b_base_link" && b == "mounting_column") ||
                      (b == "arm_b_base_link" && a == "mounting_column");
    if (part_a || base)
      continue;
    pair = a + " <-> " + b;
    return true;
  }
  return false;
}

double meshMinZ(const moveit::core::RobotState& st, const std::string& link)
{
  const auto* lm = st.getLinkModel(link);
  if (!lm)
    return 1e9;
  const auto& shapes = lm->getShapes();
  const auto& origins = lm->getCollisionOriginTransforms();
  const Eigen::Isometry3d Tw = st.getGlobalLinkTransform(lm);
  double z = 1e9;
  for (size_t i = 0; i < shapes.size() && i < origins.size(); ++i)
  {
    if (!shapes[i] || shapes[i]->type != shapes::MESH)
      continue;
    const auto* mesh = static_cast<const shapes::Mesh*>(shapes[i].get());
    const Eigen::Isometry3d T = Tw * origins[i];
    for (unsigned v = 0; v < mesh->vertex_count; ++v)
    {
      Eigen::Vector3d p(mesh->vertices[3 * v], mesh->vertices[3 * v + 1], mesh->vertices[3 * v + 2]);
      z = std::min(z, (T * p).z());
    }
  }
  return z;
}

std::vector<double> jointsOf(const moveit::core::RobotState& st)
{
  std::vector<double> q(6);
  for (int i = 0; i < 6; ++i)
    q[i] = st.getVariablePosition(kB[i]);
  return q;
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("keypose_v1_b_home", opt);
  const std::string dir = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
                          "/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1";
  YAML::Node arm_b = YAML::LoadFile(dir + "/arm_b_candidates.yaml");
  YAML::Node arm_a = YAML::LoadFile(dir + "/arm_a_candidates.yaml");
  YAML::Node search = YAML::LoadFile(dir + "/search.yaml");
  YAML::Node work = YAML::LoadFile(search["workcell_yaml"].as<std::string>());
  std::vector<double> pre, face3, home_a;
  vec6(arm_b["keyposes"]["B_PRE_HANDOVER"]["candidates"][0]["joint_values"], pre);
  vec6(arm_a["keyposes"]["A_FACE3"]["candidates"][0]["joint_values"], face3);
  vec6(arm_a["keyposes"]["A_HOME"]["candidates"][0]["joint_values"], home_a);

  robot_model_loader::RobotModelLoader loader(node);
  auto model = loader.getModel();
  auto* gb = model->getJointModelGroup("arm_b");
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  auto addBox = [&](const char* key) {
    const auto n = work[key];
    moveit_msgs::msg::CollisionObject obj;
    obj.id = n["name"].as<std::string>();
    obj.header.frame_id = scene->getPlanningFrame();
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    obj.pose.orientation.w = 1.0;
    obj.primitives.resize(1);
    obj.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    obj.primitives[0].dimensions = {n["dimensions"]["x"].as<double>(), n["dimensions"]["y"].as<double>(),
                                    n["dimensions"]["z"].as<double>()};
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = Eigen::Vector3d(n["initial_pose"]["position"]["x"].as<double>(),
                                      n["initial_pose"]["position"]["y"].as<double>(),
                                      n["initial_pose"]["position"]["z"].as<double>());
    obj.primitive_poses.push_back(poseOf(T));
    scene->processCollisionObjectMsg(obj);
  };
  addBox("table");
  addBox("column");
  scene->getAllowedCollisionMatrixNonConst().setEntry("arm_a_base_link", "mounting_column", true);
  scene->getAllowedCollisionMatrixNonConst().setEntry("arm_b_base_link", "mounting_column", true);

  const Eigen::Vector3d target(-0.45, 0.40, 0.8755);
  const Eigen::Vector3d down(0, 0, -1);
  std::vector<double> best;
  double best_cost = 1e9;
  Eigen::Isometry3d best_tcp = Eigen::Isometry3d::Identity();
  for (int k = 0; k < 24; ++k)
  {
    const double yaw = k * (M_PI / 12.0);
    Eigen::Matrix3d R;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    R.col(0) = Eigen::Vector3d(c, s, 0);
    R.col(1) = Eigen::Vector3d(s, -c, 0);
    R.col(2) = down;
    Eigen::Isometry3d goal = Eigen::Isometry3d::Identity();
    goal.linear() = R;
    goal.translation() = target;
    moveit::core::RobotState st(model);
    st.setToDefaultValues();
    setJ(st, kA, home_a);
    setJ(st, kB, pre);
    st.update();
    if (!st.setFromIK(gb, goal, kTcp, 0.2))
      continue;
    if (!limitsOk(st))
      continue;
    std::string pair;
    if (hit(*scene, st, pair))
      continue;
    const auto q = jointsOf(st);
    const double cost = 8.0 * std::abs(q[0] - pre[0]) + 8.0 * std::abs(q[1] - pre[1]) +
                        std::abs(q[2] - pre[2]) + std::abs(q[3] - pre[3]) + std::abs(q[4] - pre[4]) +
                        0.15 * std::abs(q[5] - pre[5]);
    if (cost < best_cost)
    {
      best_cost = cost;
      best = q;
      best_tcp = st.getGlobalLinkTransform(kTcp);
    }
  }
  if (best.empty())
  {
    emit("NO_LEGAL_B_HOME");
    rclcpp::shutdown();
    return 2;
  }
  moveit::core::RobotState sol(model);
  sol.setToDefaultValues();
  setJ(sol, kA, home_a);
  setJ(sol, kB, best);
  sol.update();
  const Eigen::Isometry3d tcp = sol.getGlobalLinkTransform(kTcp);
  const Eigen::Vector3d z = tcp.linear().col(2);
  const double pos_err = (tcp.translation() - target).norm();
  const double ang = std::acos(std::max(-1.0, std::min(1.0, z.dot(down)))) * 180.0 / M_PI;
  Eigen::Isometry3d TB = Eigen::Isometry3d::Identity();
  TB.translation() = Eigen::Vector3d(0, 0, 0.008);
  Eigen::Quaterniond tq(0.707106343559, 0, 0, -0.707107218813);
  tq.normalize();
  TB.linear() = tq.toRotationMatrix();
  const Eigen::Isometry3d obj = tcp * TB;
  const double cyl_bottom = obj.translation().z() - 0.0175;
  const double cyl_gap = cyl_bottom - kTableTop;
  double grip_z = 1e9;
  for (const char* link : {"arm_b_finger_l", "arm_b_finger_r", "arm_b_gripper_base_link"})
    grip_z = std::min(grip_z, meshMinZ(sol, link));
  const double grip_gap = grip_z - kTableTop;
  emit("TCP " + std::to_string(tcp.translation().x()) + " " + std::to_string(tcp.translation().y()) + " " +
       std::to_string(tcp.translation().z()));
  emit("pos_err_mm " + std::to_string(1000 * pos_err) + " z_down_deg " + std::to_string(ang));
  emit("cyl_gap_mm " + std::to_string(1000 * cyl_gap) + " grip_gap_mm " + std::to_string(1000 * grip_gap));

  Eigen::Isometry3d TA = Eigen::Isometry3d::Identity();
  TA.linear() = Eigen::Quaterniond(0, 1, 0, 0).toRotationMatrix();
  moveit::core::RobotState hold(model);
  hold.setToDefaultValues();
  setJ(hold, kA, face3);
  hold.update();
  const Eigen::Isometry3d part_in_a = Eigen::Isometry3d::Identity();
  moveit_msgs::msg::AttachedCollisionObject att;
  att.link_name = "arm_a_gripper_tcp";
  att.touch_links = kTouchA;
  att.object.id = kPart;
  att.object.header.frame_id = att.link_name;
  att.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  att.object.primitives.resize(1);
  att.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  att.object.primitives[0].dimensions = {0.035, 0.0075};
  att.object.primitive_poses.push_back(poseOf(TA));
  scene->processAttachedCollisionObjectMsg(att);
  (void)TA;

  std::vector<std::vector<double>> path;
  std::string block;
  bool direct = true;
  double jump = 0;
  for (int j = 0; j < 6; ++j)
    jump = std::max(jump, std::abs(pre[j] - best[j]));
  const int n = std::max(1, static_cast<int>(std::ceil(jump / kStep)));
  for (int s = 0; s <= n; ++s)
  {
    const double t = static_cast<double>(s) / n;
    std::vector<double> q(6);
    for (int j = 0; j < 6; ++j)
      q[j] = best[j] + (pre[j] - best[j]) * t;
    moveit::core::RobotState st(model);
    st.setToDefaultValues();
    setJ(st, kA, face3);
    setJ(st, kB, q);
    if (st.getRobotModel()->hasJointModel("arm_a_gripper_joint"))
      st.setVariablePosition("arm_a_gripper_joint", 0.083);
    std::string pair;
    if (!limitsOk(st) || hit(*scene, st, pair))
    {
      direct = false;
      block = pair.empty() ? "joint_limit" : pair;
      break;
    }
    path.push_back(q);
  }
  if (!direct)
  {
    emit(std::string("DIRECT_FAIL ") + block);
    rclcpp::shutdown();
    return 3;
  }
  emit("CONNECT direct PASS");

  const std::string home_ws = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/fr_task_ws/src/fr_task_planner";
  {
    std::ofstream os(home_ws + "/config/keypose_v1_b_home_preview.yaml");
    os << std::setprecision(12) << std::fixed;
    os << "b_home: [";
    for (int i = 0; i < 6; ++i)
      os << (i ? ", " : "") << best[i];
    os << "]\nb_pre: [";
    for (int i = 0; i < 6; ++i)
      os << (i ? ", " : "") << pre[i];
    os << "]\na_home: [";
    for (int i = 0; i < 6; ++i)
      os << (i ? ", " : "") << home_a[i];
    os << "]\ntcp: [" << tcp.translation().x() << ", " << tcp.translation().y() << ", " << tcp.translation().z()
       << "]\nobject: [" << obj.translation().x() << ", " << obj.translation().y() << ", " << obj.translation().z()
       << "]\nplacement: [-0.45, 0.40, 0.75]\n";
    const auto qq = Eigen::Quaterniond(tcp.linear());
    os << "tcp_xyzw: [" << qq.x() << ", " << qq.y() << ", " << qq.z() << ", " << qq.w() << "]\n";
  }

  auto replaceHomeB = [&](const std::string& file) {
    std::ifstream in(file);
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    const std::string old =
        "home_b: [1.018940697148, -1.214870670667, 1.401150172209, -1.757079497313, -1.570792753708, -0.551858425983]";
    std::ostringstream neu;
    neu << std::setprecision(12) << std::fixed << "home_b: [";
    for (int i = 0; i < 6; ++i)
      neu << (i ? ", " : "") << best[i];
    neu << "]";
    const auto pos = text.find(old);
    if (pos == std::string::npos)
      return false;
    text.replace(pos, old.size(), neu.str());
    std::ofstream out(file);
    out << text;
    return true;
  };
  replaceHomeB(dir + "/preview_index.yaml");
  replaceHomeB(home_ws + "/config/keypose_v1_six_face_trajectory.yaml");

  {
    std::ifstream in(dir + "/arm_b_candidates.yaml");
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    const std::string oldj =
        "joint_values: [1.018940697148, -1.214870670667, 1.401150172209, -1.757079497313, "
        "-1.570792753708, -0.551858425983]";
    const auto pos = text.find("  B_HOME:");
    const auto next = text.find(oldj, pos);
    if (pos != std::string::npos && next != std::string::npos)
    {
      std::ostringstream neu;
      neu << std::setprecision(12) << std::fixed << "joint_values: [";
      for (int i = 0; i < 6; ++i)
        neu << (i ? ", " : "") << best[i];
      neu << "]";
      text.replace(next, oldj.size(), neu.str());
      const std::string oldnote = "note: \"same joints as B_PRE_HANDOVER candidate 1\"";
      const auto np = text.find(oldnote);
      if (np != std::string::npos && np < next)
        text.replace(np, oldnote.size(),
                     "note: \"hover above placement (-0.45, 0.40), TCP +Z down, seeded from B_PRE_HANDOVER\"");
      std::ofstream out(dir + "/arm_b_candidates.yaml");
      out << text;
    }
  }

  {
    std::ifstream in(home_ws + "/config/keypose_v1_six_face_trajectory.yaml");
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    const auto pos = text.find("  - id: b_home_to_pre_handover\n");
    const auto next = text.find("  - id: ", pos + 10);
    std::ostringstream block;
    block << std::setprecision(12) << std::fixed;
    block << "  - id: b_home_to_pre_handover\n";
    block << "    kind: joint_connection\n    moving: arm_b\n    method: joint_direct\n    status: PASS\n";
    block << "    block: \"\"\n    speed_scale: 0.2\n    reference_joint_speed_rad_s: 1.0\n";
    block << "    effective_peak_joint_speed_rad_s: 0.2\n";
    double tsec = 0;
    std::vector<double> times(path.size(), 0);
    for (size_t i = 1; i < path.size(); ++i)
    {
      double step = 0;
      for (int j = 0; j < 6; ++j)
        step = std::max(step, std::abs(path[i][j] - path[i - 1][j]));
      tsec += std::max(step / 1.0, 0.02);
      times[i] = tsec;
    }
    double c1 = 0, c2 = 0, c6 = 0;
    for (size_t i = 1; i < path.size(); ++i)
    {
      c1 += std::abs(path[i][0] - path[i - 1][0]);
      c2 += std::abs(path[i][1] - path[i - 1][1]);
      c6 += std::abs(path[i][5] - path[i - 1][5]);
    }
    block << "    duration_at_scale_1_s: " << tsec << "\n    duration_at_scale_0_2_s: " << (tsec / 0.2) << "\n";
    block << "    endpoint_abs_j1_rad: " << std::abs(pre[0] - best[0]) << "\n";
    block << "    endpoint_abs_j2_rad: " << std::abs(pre[1] - best[1]) << "\n";
    block << "    endpoint_abs_j6_rad: " << std::abs(pre[5] - best[5]) << "\n";
    block << "    cumulative_abs_j1_rad: " << c1 << "\n    cumulative_abs_j2_rad: " << c2
          << "\n    cumulative_abs_j6_rad: " << c6 << "\n";
    block << "    joint_names: [arm_b_j1, arm_b_j2, arm_b_j3, arm_b_j4, arm_b_j5, arm_b_j6]\n";
    auto wj = [&](const char* k, const std::vector<double>& q) {
      block << "    " << k << ": [";
      for (int i = 0; i < 6; ++i)
        block << (i ? ", " : "") << q[i];
      block << "]\n";
    };
    wj("start_joints", best);
    wj("end_joints", pre);
    block << "    points:\n";
    for (size_t i = 0; i < path.size(); ++i)
    {
      block << "    - positions: [";
      for (int j = 0; j < 6; ++j)
        block << (j ? ", " : "") << path[i][j];
      const int sec = static_cast<int>(times[i]);
      block << "]\n      sec: " << sec << "\n      nanosec: " << static_cast<int>((times[i] - sec) * 1e9)
            << "\n      velocities: [0, 0, 0, 0, 0, 0]\n      accelerations: [0, 0, 0, 0, 0, 0]\n";
    }
    if (pos != std::string::npos && next != std::string::npos)
    {
      text.replace(pos, next - pos, block.str());
      std::ofstream out(home_ws + "/config/keypose_v1_six_face_trajectory.yaml");
      out << text;
    }
  }

  std::ostringstream dj;
  dj << std::fixed << std::setprecision(2) << "delta_deg";
  for (int j = 0; j < 6; ++j)
    dj << " " << ((best[j] - pre[j]) * 180.0 / M_PI);
  emit(dj.str());
  std::ostringstream jq;
  jq << std::fixed << std::setprecision(6) << "joints";
  for (double v : best)
    jq << " " << v;
  emit(jq.str());
  rclcpp::shutdown();
  return 0;
}
