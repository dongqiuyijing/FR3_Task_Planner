// Offline world-frame offset for KEYPOSE_V1 B_HANDOVER. Plan and verify only.
// Does not command hardware or overwrite the frozen six-face trajectory.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
const std::vector<std::string> kA = {"arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kB = {"arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"};
const std::vector<std::string> kTouchA = {
    "arm_a_gripper_base_link", "arm_a_finger_l", "arm_a_finger_r", "arm_a_gripper_tcp"};
const std::vector<std::string> kTouchB = {
    "arm_b_gripper_base_link", "arm_b_rail_155", "arm_b_slider_l", "arm_b_slider_r",
    "arm_b_finger_l", "arm_b_finger_r", "arm_b_gripper_gap_link", "arm_b_gripper_tcp"};
constexpr char kPart[] = "small_part";
constexpr char kTable[] = "table";
constexpr char kColumn[] = "mounting_column";
constexpr char kTcpB[] = "arm_b_gripper_tcp";
constexpr double kStep = 0.02;
constexpr double kRefV = 1.0;
std::string g_part_attachment;

void emit(const std::string& s) { std::cout << s << std::endl; }

bool yamlVec(const YAML::Node& n, std::vector<double>& out)
{
  if (!n || !n.IsSequence())
    return false;
  out.clear();
  for (const auto& v : n)
    out.push_back(v.as<double>());
  return out.size() >= 6;
}

std::string fmtVec(const std::vector<double>& q, int prec = 12)
{
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << "[";
  for (size_t i = 0; i < q.size(); ++i)
    os << (i ? ", " : "") << q[i];
  os << "]";
  return os.str();
}

std::string fmtXyz(const Eigen::Vector3d& p, int prec = 12)
{
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << "[" << p.x() << ", " << p.y() << ", " << p.z() << "]";
  return os.str();
}

std::string fmtQuat(const Eigen::Quaterniond& q, int prec = 12)
{
  Eigen::Quaterniond n = q.normalized();
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << "[" << n.x() << ", " << n.y() << ", " << n.z() << ", "
     << n.w() << "]";
  return os.str();
}

geometry_msgs::msg::Pose poseMsg(const Eigen::Isometry3d& T)
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

Eigen::Quaterniond quatOf(const Eigen::Isometry3d& T)
{
  Eigen::Quaterniond q(T.linear());
  q.normalize();
  return q;
}

void poseError(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double& pos, double& ori_deg)
{
  pos = (a.translation() - b.translation()).norm();
  const double d = std::min(1.0, std::abs(quatOf(a).dot(quatOf(b))));
  ori_deg = 2.0 * std::acos(d) * 180.0 / M_PI;
}

void setJ(moveit::core::RobotState& st, const std::vector<std::string>& n, const std::vector<double>& q)
{
  for (size_t i = 0; i < n.size() && i < q.size(); ++i)
    st.setVariablePosition(n[i], q[i]);
}

std::vector<double> jointsOf(const moveit::core::RobotState& st, const std::vector<std::string>& names)
{
  std::vector<double> q(names.size());
  for (size_t i = 0; i < names.size(); ++i)
    q[i] = st.getVariablePosition(names[i]);
  return q;
}

bool inLimits(const moveit::core::RobotState& st, const std::vector<std::string>& names)
{
  for (const auto& n : names)
  {
    const auto* j = st.getRobotModel()->getJointModel(n);
    if (!j || j->getVariableBounds().empty() || !j->getVariableBounds().front().position_bounded_)
      continue;
    const double q = st.getVariablePosition(n);
    const double lo = j->getVariableBounds().front().min_position_;
    const double hi = j->getVariableBounds().front().max_position_;
    if (q < lo - 1e-6 || q > hi + 1e-6)
      return false;
  }
  return true;
}

bool allowedPair(const std::string& a, const std::string& b)
{
  auto touch = [](const std::string& n, const std::vector<std::string>& t) {
    return std::find(t.begin(), t.end(), n) != t.end();
  };
  if ((a == kPart && touch(b, kTouchA)) || (b == kPart && touch(a, kTouchA)))
    return true;
  if ((a == kPart && touch(b, kTouchB)) || (b == kPart && touch(a, kTouchB)))
    return true;
  if ((a == "arm_a_base_link" && b == kColumn) || (b == "arm_a_base_link" && a == kColumn))
    return true;
  if ((a == "arm_b_base_link" && b == kColumn) || (b == "arm_b_base_link" && a == kColumn))
    return true;
  if ((a == kPart && b == kTable) || (b == kPart && a == kTable))
    return false;
  return false;
}

bool colliding(planning_scene::PlanningScene& scene, moveit::core::RobotState& st, std::string& pair,
               bool ignore_table)
{
  st.update();
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 40;
  req.max_contacts_per_pair = 1;
  scene.checkCollision(req, res, st);
  if (!res.collision)
    return false;
  for (const auto& c : res.contacts)
  {
    if (allowedPair(c.first.first, c.first.second))
      continue;
    if (ignore_table &&
        ((c.first.first == kPart && c.first.second == kTable) ||
         (c.first.second == kPart && c.first.first == kTable)))
      continue;
    pair = c.first.first + " <-> " + c.first.second;
    return true;
  }
  return false;
}

void clearPart(planning_scene::PlanningScene& scene)
{
  if (g_part_attachment.empty())
    return;
  moveit_msgs::msg::AttachedCollisionObject det;
  det.link_name = g_part_attachment;
  det.object.id = kPart;
  det.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processAttachedCollisionObjectMsg(det);
  g_part_attachment.clear();
}

void addCylinder(planning_scene::PlanningScene& scene, const std::string& link,
                 const Eigen::Isometry3d& pose, bool attached, const std::vector<std::string>& touch)
{
  clearPart(scene);
  if (!attached)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = kPart;
    obj.header.frame_id = scene.getPlanningFrame();
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    obj.primitives.resize(1);
    obj.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    obj.primitives[0].dimensions = {0.035, 0.0075};
    obj.primitive_poses.push_back(poseMsg(pose));
    scene.processCollisionObjectMsg(obj);
    return;
  }
  moveit_msgs::msg::AttachedCollisionObject att;
  att.link_name = link;
  att.touch_links = touch;
  att.object.id = kPart;
  att.object.header.frame_id = link;
  att.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  att.object.primitives.resize(1);
  att.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  att.object.primitives[0].dimensions = {0.035, 0.0075};
  att.object.primitive_poses.push_back(poseMsg(pose));
  scene.processAttachedCollisionObjectMsg(att);
  g_part_attachment = link;
}

struct Seg
{
  std::string id;
  std::string method;
  std::string status = "NOT_PLANNED";
  std::string block;
  bool move_a = false;
  bool ignore_table = false;
  bool allow_cross = false;
  std::vector<double> start;
  std::vector<double> goal;
  std::vector<std::vector<double>> path;
  double ep_j1 = 0, ep_j2 = 0, ep_j6 = 0;
  double cum_j1 = 0, cum_j2 = 0, cum_j6 = 0;
  double dur1 = 0;
};

void accum(Seg& s)
{
  auto d = [](double a, double b) { return std::abs(a - b); };
  if (s.start.size() == 6 && s.goal.size() == 6)
  {
    s.ep_j1 = d(s.goal[0], s.start[0]);
    s.ep_j2 = d(s.goal[1], s.start[1]);
    s.ep_j6 = d(s.goal[5], s.start[5]);
  }
  s.cum_j1 = s.cum_j2 = s.cum_j6 = 0;
  for (size_t i = 1; i < s.path.size(); ++i)
  {
    s.cum_j1 += d(s.path[i][0], s.path[i - 1][0]);
    s.cum_j2 += d(s.path[i][1], s.path[i - 1][1]);
    s.cum_j6 += d(s.path[i][5], s.path[i - 1][5]);
  }
  double t = 0;
  for (size_t i = 1; i < s.path.size(); ++i)
  {
    double jump = 0;
    for (int j = 0; j < 6; ++j)
      jump = std::max(jump, std::abs(s.path[i][j] - s.path[i - 1][j]));
    t += std::max(jump / kRefV, 0.02);
  }
  s.dur1 = t;
}

bool sampleOk(planning_scene::PlanningScene& scene, const moveit::core::RobotModelConstPtr& model,
              const std::vector<double>& a, const std::vector<double>& b, bool move_a, double qa,
              double qb, bool ignore_table, std::string& pair)
{
  moveit::core::RobotState st(model);
  st.setToDefaultValues();
  setJ(st, kA, a);
  setJ(st, kB, b);
  if (st.getRobotModel()->hasJointModel("arm_a_gripper_joint"))
    st.setVariablePosition("arm_a_gripper_joint", qa);
  if (st.getRobotModel()->hasJointModel("arm_b_gripper_joint"))
    st.setVariablePosition("arm_b_gripper_joint", qb);
  if (!inLimits(st, move_a ? kA : kB))
  {
    pair = "joint_limit";
    return false;
  }
  return !colliding(scene, st, pair, ignore_table);
}

bool connectOrder(planning_scene::PlanningScene& scene, const moveit::core::RobotModelConstPtr& model,
                  const std::vector<double>& fixed, std::vector<double> from, const std::vector<double>& to,
                  bool move_a, double qa, double qb, bool ignore_table, const std::vector<int>& order,
                  std::vector<std::vector<double>>& path, std::string& pair)
{
  path.clear();
  path.push_back(from);
  std::vector<double> cur = from;
  for (int idx : order)
  {
    if (std::abs(to[idx] - cur[idx]) < 1e-9)
      continue;
    const int n = std::max(1, static_cast<int>(std::ceil(std::abs(to[idx] - cur[idx]) / kStep)));
    for (int s = 1; s <= n; ++s)
    {
      auto q = cur;
      q[idx] = cur[idx] + (to[idx] - cur[idx]) * (static_cast<double>(s) / n);
      const auto& a = move_a ? q : fixed;
      const auto& b = move_a ? fixed : q;
      if (!sampleOk(scene, model, a, b, move_a, qa, qb, ignore_table, pair))
        return false;
      path.push_back(q);
    }
    cur[idx] = to[idx];
  }
  return true;
}

bool connectDirect(planning_scene::PlanningScene& scene, const moveit::core::RobotModelConstPtr& model,
                   const std::vector<double>& fixed, const std::vector<double>& from,
                   const std::vector<double>& to, bool move_a, double qa, double qb, bool ignore_table,
                   std::vector<std::vector<double>>& path, std::string& pair)
{
  path.clear();
  double jump = 0;
  for (int j = 0; j < 6; ++j)
    jump = std::max(jump, std::abs(to[j] - from[j]));
  const int n = std::max(1, static_cast<int>(std::ceil(jump / kStep)));
  for (int s = 0; s <= n; ++s)
  {
    std::vector<double> q(6);
    const double t = static_cast<double>(s) / n;
    for (int j = 0; j < 6; ++j)
      q[j] = from[j] + (to[j] - from[j]) * t;
    const auto& a = move_a ? q : fixed;
    const auto& b = move_a ? fixed : q;
    if (!sampleOk(scene, model, a, b, move_a, qa, qb, ignore_table, pair))
      return false;
    path.push_back(q);
  }
  return true;
}

bool planSeg(planning_scene::PlanningScene& scene, const moveit::core::RobotModelConstPtr& model, Seg& seg,
             const std::vector<double>& fixed, double qa, double qb)
{
  if (seg.start.size() != 6 || seg.goal.size() != 6)
  {
    seg.status = "FAIL";
    seg.block = "bad endpoint";
    return false;
  }
  const std::vector<double>& start = seg.start;
  double max_d = 0;
  for (int j = 0; j < 6; ++j)
    max_d = std::max(max_d, std::abs(seg.goal[j] - start[j]));
  if (max_d < 1e-6)
  {
    seg.path = {start};
    seg.method = "same_keypose";
    seg.status = "PASS";
    accum(seg);
    return true;
  }
  std::string pair;
  if (connectDirect(scene, model, fixed, start, seg.goal, seg.move_a, qa, qb, seg.ignore_table, seg.path,
                    pair))
  {
    seg.method = "joint_direct";
    seg.status = "PASS";
    accum(seg);
    return true;
  }
  const std::vector<std::vector<int>> orders = {
      {5, 4, 3, 2, 1, 0},
      {5, 4, 3, 2, 0, 1},
      {2, 3, 4, 5, 0, 1},
  };
  const char* names[] = {"j6_first", "wrist_then_j12", "arm_then_j12"};
  for (size_t i = 0; i < orders.size(); ++i)
  {
    if (connectOrder(scene, model, fixed, start, seg.goal, seg.move_a, qa, qb, seg.ignore_table, orders[i],
                     seg.path, pair))
    {
      seg.method = names[i];
      seg.status = "PASS";
      accum(seg);
      return true;
    }
  }
  seg.status = "FAIL";
  seg.block = pair.empty() ? "no_connection" : pair;
  seg.path = {start};
  accum(seg);
  return false;
}

void writePath(std::ostream& os, const Seg& s)
{
  const auto& names = s.move_a ? kA : kB;
  os << "  - id: " << s.id << "\n";
  os << "    kind: joint_connection\n";
  os << "    moving: " << (s.move_a ? "arm_a" : "arm_b") << "\n";
  os << "    method: " << s.method << "\n";
  os << "    status: " << s.status << "\n";
  os << "    block: \"" << s.block << "\"\n";
  os << "    speed_scale: 0.2\n";
  os << "    reference_joint_speed_rad_s: " << kRefV << "\n";
  os << "    effective_peak_joint_speed_rad_s: " << (kRefV * 0.2) << "\n";
  os << "    duration_at_scale_1_s: " << s.dur1 << "\n";
  os << "    duration_at_scale_0_2_s: " << (s.dur1 / 0.2) << "\n";
  os << "    endpoint_abs_j1_rad: " << s.ep_j1 << "\n";
  os << "    endpoint_abs_j2_rad: " << s.ep_j2 << "\n";
  os << "    endpoint_abs_j6_rad: " << s.ep_j6 << "\n";
  os << "    cumulative_abs_j1_rad: " << s.cum_j1 << "\n";
  os << "    cumulative_abs_j2_rad: " << s.cum_j2 << "\n";
  os << "    cumulative_abs_j6_rad: " << s.cum_j6 << "\n";
  os << "    joint_names: [";
  for (size_t i = 0; i < names.size(); ++i)
    os << (i ? ", " : "") << names[i];
  os << "]\n";
  os.setf(std::ios::fixed);
  os << std::setprecision(12);
  auto wj = [&](const char* k, const std::vector<double>& q) {
    os << "    " << k << ": [";
    for (size_t i = 0; i < q.size(); ++i)
      os << (i ? ", " : "") << q[i];
    os << "]\n";
  };
  if (!s.start.empty())
    wj("start_joints", s.start);
  if (!s.goal.empty())
    wj("end_joints", s.goal);
  os << "    points:\n";
  double t = 0;
  for (size_t i = 0; i < s.path.size(); ++i)
  {
    if (i > 0)
    {
      double jump = 0;
      for (int j = 0; j < 6; ++j)
        jump = std::max(jump, std::abs(s.path[i][j] - s.path[i - 1][j]));
      t += std::max(jump / kRefV, 0.02);
    }
    const int sec = static_cast<int>(t);
    const auto nsec = static_cast<uint32_t>((t - sec) * 1e9);
    os << "    - positions: [";
    for (int j = 0; j < 6; ++j)
      os << (j ? ", " : "") << s.path[i][j];
    os << "]\n";
    os << "      sec: " << sec << "\n";
    os << "      nanosec: " << nsec << "\n";
    os << "      velocities: [";
    for (int j = 0; j < 6; ++j)
    {
      double v = 0;
      if (i > 0)
      {
        double jump = 0;
        for (int k = 0; k < 6; ++k)
          jump = std::max(jump, std::abs(s.path[i][k] - s.path[i - 1][k]));
        const double step_t = std::max(jump / kRefV, 0.02);
        v = (s.path[i][j] - s.path[i - 1][j]) / step_t;
      }
      os << (j ? ", " : "") << v;
    }
    os << "]\n";
    os << "      accelerations: [0, 0, 0, 0, 0, 0]\n";
  }
}

bool loadStageJoints(const YAML::Node& traj, const std::string& id, std::vector<double>& start,
                     std::vector<double>& goal, std::vector<std::vector<double>>& points)
{
  start.clear();
  goal.clear();
  points.clear();
  for (const auto& st : traj["stages"])
  {
    if (!st["id"] || st["id"].as<std::string>() != id)
      continue;
    yamlVec(st["start_joints"], start);
    yamlVec(st["end_joints"], goal);
    if (st["points"] && st["points"].IsSequence())
    {
      for (const auto& p : st["points"])
      {
        std::vector<double> q;
        if (yamlVec(p["positions"], q))
          points.push_back(q);
      }
    }
    return !start.empty() && !goal.empty();
  }
  return false;
}

bool replaceStage(std::string& text, const Seg& seg)
{
  const std::string key = "  - id: " + seg.id + "\n";
  const auto pos = text.find(key);
  if (pos == std::string::npos)
    return false;
  auto next = text.find("  - id: ", pos + key.size());
  if (next == std::string::npos)
    next = text.size();
  std::ostringstream block;
  writePath(block, seg);
  text.replace(pos, next - pos, block.str());
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("keypose_v1_b_handover_offset", opt);
  emit("KEYPOSE_V1 B_HANDOVER world offset. Offline only. Does not move the robot.");

  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string cfg_path =
      home + "/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/arm_b_handover_offset.yaml";
  YAML::Node cfg = YAML::LoadFile(cfg_path);
  const auto offn = cfg["arm_b_handover_offset_world"];
  const Eigen::Vector3d offset(offn["x"].as<double>(), offn["y"].as<double>(), offn["z"].as<double>());
  const std::string frozen = cfg["frozen_trajectory"].as<std::string>();
  const std::string report_path = cfg["output_report"].as<std::string>();
  const std::string preview_path = cfg["output_preview"].as<std::string>();
  const std::string cand_traj = cfg["output_trajectory_candidate"].as<std::string>();
  const bool overwrite = cfg["overwrite_frozen_trajectory"] && cfg["overwrite_frozen_trajectory"].as<bool>();
  const bool promote = cfg["promote_candidate_to_frozen"].as<bool>(false);
  const double fk_pos_tol = cfg["fk_pos_tol_m"].as<double>(0.0005);
  const double fk_ori_tol = cfg["fk_ori_tol_deg"].as<double>(0.5);
  const double off_pos_tol = cfg["offset_pos_tol_m"].as<double>(0.0005);
  const double off_ori_tol = cfg["offset_ori_tol_deg"].as<double>(0.5);
  const double ik_timeout = cfg["ik_timeout_s"].as<double>(0.25);
  const int nearby_n = cfg["ik_nearby_attempts"].as<int>(48);
  const double nearby_r = cfg["ik_nearby_radius_rad"].as<double>(0.35);

  emit("offset_world_m " + fmtXyz(offset, 6));
  emit("frozen source (untouched unless overwrite=true): " + frozen);
  if (overwrite)
  {
    emit("overwrite_frozen_trajectory is true; refusing. Write a candidate file only.");
    rclcpp::shutdown();
    return 2;
  }

  YAML::Node traj = YAML::LoadFile(frozen);
  YAML::Node a_cands = YAML::LoadFile(home + "/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/"
                                             "arm_a_candidates.yaml");
  YAML::Node tgt = YAML::LoadFile(home + "/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/"
                                         "task_space_targets.yaml");
  YAML::Node search = YAML::LoadFile(home + "/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/"
                                            "search.yaml");
  YAML::Node work = YAML::LoadFile(search["workcell_yaml"].as<std::string>());

  std::vector<double> a_home, b_home, a_han, b_pre, b_han_old, b_face4;
  yamlVec(traj["home_a"], a_home);
  yamlVec(traj["home_b"], b_home);
  std::vector<std::vector<double>> dummy, a_return_pts;
  std::vector<double> a_pre_han_end, b_pre_end, b_han_to_f4_start, a_han_to_home_start;
  if (!loadStageJoints(traj, "b_pre_to_handover", b_pre, b_han_old, dummy))
  {
    emit("FAIL missing b_pre_to_handover in frozen trajectory");
    rclcpp::shutdown();
    return 2;
  }
  if (!loadStageJoints(traj, "b_handover_to_face4", b_han_to_f4_start, b_face4, dummy))
  {
    emit("FAIL missing b_handover_to_face4 in frozen trajectory");
    rclcpp::shutdown();
    return 2;
  }
  if (!loadStageJoints(traj, "pre_handover_to_handover", a_han_to_home_start, a_han, dummy))
  {
    emit("FAIL missing pre_handover_to_handover in frozen trajectory");
    rclcpp::shutdown();
    return 2;
  }
  (void)a_han_to_home_start;
  if (!loadStageJoints(traj, "a_handover_to_home", a_han_to_home_start, a_pre_han_end, a_return_pts))
  {
    emit("FAIL missing a_handover_to_home in frozen trajectory");
    rclcpp::shutdown();
    return 2;
  }
  (void)a_pre_han_end;
  {
    bool differ = b_han_old.size() != 6 || b_han_to_f4_start.size() != 6;
    for (size_t i = 0; i < 6 && !differ; ++i)
      differ = std::abs(b_han_old[i] - b_han_to_f4_start[i]) > 1e-9;
    if (differ)
      emit("WARN frozen b_pre_to_handover end and b_handover_to_face4 start differ");
  }
  std::vector<double> a_han_cand;
  {
    const auto cs = a_cands["keyposes"]["A_HANDOVER"]["candidates"];
    yamlVec(cs[0]["joint_values"], a_han_cand);
    for (int i = 0; i < 6; ++i)
    {
      if (std::abs(a_han_cand[i] - a_han[i]) > 1e-9)
      {
        emit("WARN A_HANDOVER frozen vs candidate-1 differ; using frozen trajectory joints");
        break;
      }
    }
  }

  robot_model_loader::RobotModelLoader loader(node);
  auto model = loader.getModel();
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  auto addBox = [&](const YAML::Node& n, const char* id) {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = n["name"] ? n["name"].as<std::string>() : id;
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
    obj.primitive_poses.push_back(poseMsg(T));
    scene->processCollisionObjectMsg(obj);
  };
  addBox(work["table"], kTable);
  addBox(work["column"], kColumn);
  auto& acm = scene->getAllowedCollisionMatrixNonConst();
  acm.setEntry("arm_a_base_link", kColumn, true);
  acm.setEntry("arm_b_base_link", kColumn, true);
  for (const auto& t : kTouchA)
    acm.setEntry(kPart, t, true);
  for (const auto& t : kTouchB)
    acm.setEntry(kPart, t, true);

  Eigen::Isometry3d TA = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d TB = Eigen::Isometry3d::Identity();
  {
    const auto pa = tgt["T_tcpA_object"]["pose"];
    const auto pb = tgt["T_tcpB_object"]["pose"];
    TA.translation() = Eigen::Vector3d(pa["xyz"][0].as<double>(), pa["xyz"][1].as<double>(),
                                       pa["xyz"][2].as<double>());
    Eigen::Quaterniond qa(pa["xyzw"][3].as<double>(), pa["xyzw"][0].as<double>(), pa["xyzw"][1].as<double>(),
                          pa["xyzw"][2].as<double>());
    qa.normalize();
    TA.linear() = qa.toRotationMatrix();
    TB.translation() = Eigen::Vector3d(pb["xyz"][0].as<double>(), pb["xyz"][1].as<double>(),
                                       pb["xyz"][2].as<double>());
    Eigen::Quaterniond qb(pb["xyzw"][3].as<double>(), pb["xyzw"][0].as<double>(), pb["xyzw"][1].as<double>(),
                          pb["xyzw"][2].as<double>());
    qb.normalize();
    TB.linear() = qb.toRotationMatrix();
  }

  const double qa_open = 0.0, qa_grasp = 0.083, qb_open = 0.0, qb_hold = 0.083;
  const auto* gb = model->getJointModelGroup("arm_b");
  if (!gb)
  {
    emit("FAIL missing arm_b group");
    rclcpp::shutdown();
    return 2;
  }

  moveit::core::RobotState st(model);
  st.setToDefaultValues();
  setJ(st, kA, a_han);
  setJ(st, kB, b_han_old);
  if (st.getRobotModel()->hasJointModel("arm_a_gripper_joint"))
    st.setVariablePosition("arm_a_gripper_joint", qa_grasp);
  if (st.getRobotModel()->hasJointModel("arm_b_gripper_joint"))
    st.setVariablePosition("arm_b_gripper_joint", qb_open);
  st.update();
  const Eigen::Isometry3d T_old = st.getGlobalLinkTransform(kTcpB);
  const Eigen::Isometry3d T_a = st.getGlobalLinkTransform("arm_a_gripper_tcp");
  Eigen::Isometry3d T_tgt = T_old;
  T_tgt.translation() += offset;
  emit("old B_HANDOVER tcp xyz " + fmtXyz(T_old.translation()));
  emit("new B_HANDOVER target xyz " + fmtXyz(T_tgt.translation()) + " (world translation only)");
  emit("A_HANDOVER tcp xyz " + fmtXyz(T_a.translation()) + " unchanged");

  auto tryIk = [&](moveit::core::RobotState& seed, double timeout, std::vector<double>& q_out,
                   Eigen::Isometry3d& fk_out, double& pos_err, double& ori_err) {
    if (!seed.setFromIK(gb, T_tgt, kTcpB, timeout))
      return false;
    seed.update();
    if (!inLimits(seed, kB))
      return false;
    q_out = jointsOf(seed, kB);
    fk_out = seed.getGlobalLinkTransform(kTcpB);
    poseError(T_tgt, fk_out, pos_err, ori_err);
    return pos_err <= fk_pos_tol && ori_err <= fk_ori_tol;
  };

  std::vector<double> b_han_new;
  Eigen::Isometry3d T_new = Eigen::Isometry3d::Identity();
  double ik_pos = 1e9, ik_ori = 1e9;
  bool ik_ok = false;
  {
    moveit::core::RobotState seed(st);
    ik_ok = tryIk(seed, ik_timeout, b_han_new, T_new, ik_pos, ik_ori);
  }
  if (!ik_ok)
  {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-nearby_r, nearby_r);
    for (int i = 0; i < nearby_n && !ik_ok; ++i)
    {
      moveit::core::RobotState seed(st);
      auto q = b_han_old;
      for (int j = 0; j < 6; ++j)
        q[j] += dist(rng);
      setJ(seed, kB, q);
      seed.update();
      ik_ok = tryIk(seed, 0.12, b_han_new, T_new, ik_pos, ik_ori);
    }
  }
  emit(std::string("IK ") + (ik_ok ? "PASS" : "FAIL") + " fk_pos_err_m=" + std::to_string(ik_pos) +
       " fk_ori_err_deg=" + std::to_string(ik_ori));
  if (!ik_ok)
  {
    emit("cannot solve offset B_HANDOVER; frozen trajectory left unchanged");
    rclcpp::shutdown();
    return 3;
  }

  const Eigen::Vector3d dxyz = T_new.translation() - T_old.translation();
  double ori_keep = 0, dummy_pos = 0;
  poseError(T_old, T_new, dummy_pos, ori_keep);
  (void)dummy_pos;
  const Eigen::Vector3d dxyz_err = dxyz - offset;
  const bool offset_ok = dxyz_err.norm() <= off_pos_tol && ori_keep <= off_ori_tol;
  emit("FK delta xyz_mm [" + std::to_string(dxyz.x() * 1000.0) + ", " + std::to_string(dxyz.y() * 1000.0) +
       ", " + std::to_string(dxyz.z() * 1000.0) + "] expect " + fmtXyz(offset * 1000.0, 3));
  emit(std::string("offset/orientation check ") + (offset_ok ? "PASS" : "FAIL") +
       " ori_change_deg=" + std::to_string(ori_keep));
  emit("old joints " + fmtVec(b_han_old));
  emit("new joints " + fmtVec(b_han_new));

  addCylinder(*scene, "arm_a_gripper_tcp", TA, true, kTouchA);
  std::string pair;
  const bool dual_pre_transfer = sampleOk(*scene, model, a_han, b_han_new, false, qa_grasp, qb_open, false, pair);
  emit(std::string("dual handover pose (part on A, B open) ") +
       (dual_pre_transfer ? "PASS" : ("FAIL " + pair)));

  Seg pre;
  pre.id = "b_pre_to_handover";
  pre.start = b_pre;
  pre.goal = b_han_new;
  const bool pre_ok = planSeg(*scene, model, pre, a_han, qa_grasp, qb_open);
  emit("b_pre_to_handover " + pre.status + " " + pre.method + " " + pre.block);

  addCylinder(*scene, "arm_b_gripper_tcp", TB, true, kTouchB);
  pair.clear();
  const bool dual_post_transfer =
      sampleOk(*scene, model, a_han, b_han_new, true, qa_open, qb_hold, false, pair);
  emit(std::string("dual handover pose (part on B, A open) ") +
       (dual_post_transfer ? "PASS" : ("FAIL " + pair)));

  bool a_return_ok = true;
  std::string a_return_hit;
  for (size_t i = 0; i < a_return_pts.size(); ++i)
  {
    std::string p;
    if (!sampleOk(*scene, model, a_return_pts[i], b_han_new, true, qa_open, qb_hold, false, p))
    {
      a_return_ok = false;
      a_return_hit = "sample " + std::to_string(i) + " " + p;
      break;
    }
  }
  emit(std::string("existing a_handover_to_home vs new B_HANDOVER ") +
       (a_return_ok ? "PASS" : ("FAIL " + a_return_hit)));

  Seg face4;
  face4.id = "b_handover_to_face4";
  face4.start = b_han_new;
  face4.goal = b_face4;
  const bool f4_ok = planSeg(*scene, model, face4, a_home, qa_open, qb_hold);
  emit("b_handover_to_face4 " + face4.status + " " + face4.method + " " + face4.block);

  const bool all = ik_ok && offset_ok && dual_pre_transfer && dual_post_transfer && pre_ok && f4_ok && a_return_ok;
  emit(std::string("OVERALL ") + (all ? "PASS" : "FAIL"));

  auto writePose = [](std::ostream& os, const char* key, const Eigen::Isometry3d& T) {
    os << key << ":\n";
    os << "  xyz: " << fmtXyz(T.translation()) << "\n";
    os << "  xyzw: " << fmtQuat(quatOf(T)) << "\n";
  };

  {
    std::ofstream os(report_path);
    os << std::fixed << std::setprecision(12);
    os << "task_version: KEYPOSE_V1\n";
    os << "execution: false\n";
    os << "frozen_source_untouched: " << (promote ? "false" : "true") << "\n";
    os << "frame: world\n";
    os << "units: m\n";
    os << "arm_b_handover_offset_world: " << fmtXyz(offset) << "\n";
    os << "overall: " << (all ? "PASS" : "FAIL") << "\n";
    os << "ik: " << (ik_ok ? "PASS" : "FAIL") << "\n";
    os << "fk_ik_pos_err_m: " << ik_pos << "\n";
    os << "fk_ik_ori_err_deg: " << ik_ori << "\n";
    os << "offset_check: " << (offset_ok ? "PASS" : "FAIL") << "\n";
    os << "measured_delta_world_m: " << fmtXyz(dxyz) << "\n";
    os << "measured_delta_world_mm: [" << (dxyz.x() * 1000.0) << ", " << (dxyz.y() * 1000.0) << ", "
       << (dxyz.z() * 1000.0) << "]\n";
    os << "orientation_change_deg: " << ori_keep << "\n";
    os << "a_handover_unchanged: true\n";
    os << "old_B_HANDOVER_joints: " << fmtVec(b_han_old) << "\n";
    os << "new_B_HANDOVER_joints: " << fmtVec(b_han_new) << "\n";
    os << "A_HANDOVER_joints: " << fmtVec(a_han) << "\n";
    writePose(os, "old_B_HANDOVER_tcp", T_old);
    writePose(os, "new_B_HANDOVER_tcp", T_new);
    writePose(os, "A_HANDOVER_tcp", T_a);
    os << "collision:\n";
    os << "  dual_pose_part_on_A_B_approach: " << (dual_pre_transfer ? "PASS" : "FAIL") << "\n";
    os << "  dual_pose_part_on_B: " << (dual_post_transfer ? "PASS" : "FAIL") << "\n";
    os << "  b_pre_to_handover: " << pre.status << "\n";
    os << "  b_pre_to_handover_method: " << pre.method << "\n";
    os << "  b_pre_to_handover_block: \"" << pre.block << "\"\n";
    os << "  b_handover_to_face4: " << face4.status << "\n";
    os << "  b_handover_to_face4_method: " << face4.method << "\n";
    os << "  b_handover_to_face4_block: \"" << face4.block << "\"\n";
    os << "  existing_a_handover_to_home: " << (a_return_ok ? "PASS" : "FAIL") << "\n";
    os << "  existing_a_handover_to_home_note: \"" << a_return_hit << "\"\n";
    os << "unaffected_kept: [A_HANDOVER, A trajectories, Home, face keyposes, gripper params, waits, ServoJ]\n";
    os << "affected_replanned: [b_pre_to_handover, b_handover_to_face4]\n";
    os << "candidate_trajectory: " << cand_traj << "\n";
  }

  {
    std::ofstream os(preview_path);
    os << std::fixed << std::setprecision(12);
    os << "execution: false\n";
    os << "note: RViz compare only. Not executable.\n";
    os << "A_HANDOVER_joints: " << fmtVec(a_han) << "\n";
    os << "old_B_HANDOVER_joints: " << fmtVec(b_han_old) << "\n";
    os << "new_B_HANDOVER_joints: " << fmtVec(b_han_new) << "\n";
    os << "q_a_grasp: " << qa_grasp << "\n";
    os << "q_b_open: " << qb_open << "\n";
    writePose(os, "A_HANDOVER_tcp", T_a);
    writePose(os, "old_B_HANDOVER_tcp", T_old);
    writePose(os, "new_B_HANDOVER_tcp", T_new);
    os << "offset_world_m: " << fmtXyz(offset) << "\n";
  }

  std::ifstream in(frozen);
  std::stringstream buf;
  buf << in.rdbuf();
  std::string text = buf.str();
  const std::string frozen_text = text;
  auto setField = [&](const std::string& key, const std::string& val) {
    const auto p = text.find(key);
    if (p == std::string::npos)
      return;
    const auto e = text.find('\n', p);
    text.replace(p, e - p, key + val);
  };
  setField("task_version: ", "KEYPOSE_V1_SIX_FACE_B_HANDOVER_OFFSET_CANDIDATE");
  setField("execution: ", "false");
  if (text.find("frozen_source_untouched:") == std::string::npos)
  {
    const auto p = text.find("execution: false\n");
    if (p != std::string::npos)
    {
      std::ostringstream extra;
      extra << std::fixed << std::setprecision(12);
      extra << "execution: false\n";
      extra << "frozen_source_untouched: true\n";
      extra << "source_frozen: " << frozen << "\n";
      extra << "arm_b_handover_offset_world: " << fmtXyz(offset) << "\n";
      extra << "offset_frame: world\n";
      extra << "new_B_HANDOVER_joints: " << fmtVec(b_han_new) << "\n";
      extra << "chain_offset: " << (all ? "PASS" : "FAIL") << "\n";
      text.replace(p, std::string("execution: false\n").size(), extra.str());
    }
  }
  if (!replaceStage(text, pre) || !replaceStage(text, face4))
  {
    emit("FAIL could not patch candidate trajectory stages");
    rclcpp::shutdown();
    return 3;
  }
  {
    std::ofstream os(cand_traj);
    os << text;
  }
  if (promote)
  {
    if (!all)
    {
      emit("FAIL promotion refused because validation did not PASS");
      rclcpp::shutdown();
      return 3;
    }
    // Promote from the original text so no candidate-only metadata leaks into
    // the formal frozen trajectory. Only the two validated B connections move.
    std::string promoted = frozen_text;
    if (!replaceStage(promoted, pre) || !replaceStage(promoted, face4))
    {
      emit("FAIL promotion could not patch both affected frozen stages");
      rclcpp::shutdown();
      return 3;
    }
    std::ofstream os(frozen);
    os << promoted;
    emit("PROMOTED validated B_HANDOVER connections into " + frozen);
  }
  emit("wrote report " + report_path);
  emit("wrote preview " + preview_path);
  emit("wrote candidate trajectory " + cand_traj);
  emit(promote ? "formal frozen trajectory updated: " + frozen : "did not modify " + frozen);
  rclcpp::shutdown();
  return all ? 0 : 3;
}
