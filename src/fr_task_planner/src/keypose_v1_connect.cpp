// Connect confirmed KEYPOSE_V1 candidates. Does not search new keyposes.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit_msgs/msg/motion_plan_request.hpp>
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
constexpr double kStep = 0.02;
constexpr double kRefV = 1.0;

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

std::vector<double> cand1(const YAML::Node& root, const char* name)
{
  std::vector<double> q;
  const auto block = root["keyposes"][name];
  if (!block || !block["legal"] || !block["legal"].as<bool>())
    return {};
  const auto cs = block["candidates"];
  if (!cs || !cs.IsSequence() || cs.size() == 0)
    return {};
  if (!yamlVec(cs[0]["joint_values"], q))
    return {};
  return q;
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

void setJ(moveit::core::RobotState& st, const std::vector<std::string>& n, const std::vector<double>& q)
{
  for (size_t i = 0; i < n.size() && i < q.size(); ++i)
    st.setVariablePosition(n[i], q[i]);
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

bool g_allow_gripper_cross = false;

bool allowedPair(const std::string& a, const std::string& b)
{
  auto touch = [](const std::string& n, const std::vector<std::string>& t) {
    return std::find(t.begin(), t.end(), n) != t.end();
  };
  if ((a == kPart && touch(b, kTouchA)) || (b == kPart && touch(a, kTouchA)))
    return true;
  if ((a == kPart && touch(b, kTouchB)) || (b == kPart && touch(a, kTouchB)))
    return true;
  if (g_allow_gripper_cross &&
      ((touch(a, kTouchA) && touch(b, kTouchB)) || (touch(b, kTouchA) && touch(a, kTouchB))))
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
  moveit_msgs::msg::AttachedCollisionObject det;
  det.object.id = kPart;
  det.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processAttachedCollisionObjectMsg(det);
  moveit_msgs::msg::CollisionObject rem;
  rem.id = kPart;
  rem.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processCollisionObjectMsg(rem);
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
}

struct Seg
{
  std::string id;
  std::string moving;
  std::string method;
  std::string status = "NOT_PLANNED";
  std::string block;
  bool move_a = true;
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
  g_allow_gripper_cross = seg.allow_cross;
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
        double dt = std::max(t - (t - std::max(std::abs(s.path[i][j] - s.path[i - 1][j]) / kRefV, 0.0)), 1e-6);
        (void)dt;
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

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("keypose_v1_connect", opt);
  const bool only_a_return =
      node->has_parameter("only_a_return") && node->get_parameter("only_a_return").as_bool();
  const std::string dir = std::getenv("HOME")
                              ? std::string(std::getenv("HOME")) +
                                    "/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1"
                              : ".";
  YAML::Node a = YAML::LoadFile(dir + "/arm_a_candidates.yaml");
  YAML::Node b = YAML::LoadFile(dir + "/arm_b_candidates.yaml");
  YAML::Node tgt = YAML::LoadFile(dir + "/task_space_targets.yaml");
  YAML::Node search = YAML::LoadFile(dir + "/search.yaml");
  YAML::Node work = YAML::LoadFile(search["workcell_yaml"].as<std::string>());

  const char* need_a[] = {"A_HOME", "A_PREGRASP", "A_GRASP", "A_LIFT", "A_FACE1",
                          "A_FACE2", "A_FACE3", "A_PRE_HANDOVER", "A_HANDOVER"};
  const char* need_b[] = {"B_HOME", "B_PRE_HANDOVER", "B_HANDOVER", "B_FACE4", "B_FACE5", "B_FACE6"};
  std::map<std::string, std::vector<double>> Q;
  for (const char* n : need_a)
  {
    Q[n] = cand1(a, n);
    if (Q[n].empty())
    {
      emit(std::string("MISSING confirmed candidate ") + n);
      rclcpp::shutdown();
      return 2;
    }
  }
  for (const char* n : need_b)
  {
    Q[n] = cand1(b, n);
    if (Q[n].empty())
    {
      emit(std::string("MISSING confirmed candidate ") + n);
      rclcpp::shutdown();
      return 2;
    }
  }
  emit("confirmed keyposes present: A 9, B 6, candidate index 1, legal true");
  emit("no place/release keypose in the confirmed set or in DUAL-7 stage list; not inventing one");

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
  moveit::core::RobotState gst(model);
  gst.setToDefaultValues();
  setJ(gst, kA, Q["A_GRASP"]);
  gst.update();
  const Eigen::Isometry3d grasp_obj = gst.getGlobalLinkTransform("arm_a_gripper_tcp") * TA;
  addCylinder(*scene, "", grasp_obj, false, {});

  const double qa_open = 0.0, qa_grasp = 0.083, qb_open = 0.0, qb_hold = 0.083;
  if (only_a_return)
  {
    emit("ONLY A Handover -> A Home. Other segments are not replanned.");
    emit("Previous run did NOT call OMPL RRTConnect. It only checked joint-space interpolation.");
    addCylinder(*scene, "arm_b_gripper_tcp", TB, true, kTouchB);
    g_allow_gripper_cross = false;
    std::string han_pair, home_pair;
    const bool han_raw =
        sampleOk(*scene, model, Q["A_HANDOVER"], Q["B_HANDOVER"], true, qa_open, qb_hold, false, han_pair);
    const bool home_raw =
        sampleOk(*scene, model, Q["A_HOME"], Q["B_HANDOVER"], true, qa_open, qb_hold, false, home_pair);
    emit(std::string("A_HANDOVER static without gripper-cross ") + (han_raw ? "PASS" : ("FAIL " + han_pair)));
    emit(std::string("A_HOME static without gripper-cross ") + (home_raw ? "PASS" : ("FAIL " + home_pair)));
    g_allow_gripper_cross = true;
    for (const auto& ta : kTouchA)
      for (const auto& tb : kTouchB)
        scene->getAllowedCollisionMatrixNonConst().setEntry(ta, tb, true);
    std::string pair;
    const bool han_ok =
        sampleOk(*scene, model, Q["A_HANDOVER"], Q["B_HANDOVER"], true, qa_open, qb_hold, false, pair);
    emit(std::string("A_HANDOVER static ") + (han_ok ? "PASS" : ("FAIL " + pair)));
    emit("A_HANDOVER rad will be printed by deg line");
    const bool home_ok =
        sampleOk(*scene, model, Q["A_HOME"], Q["B_HANDOVER"], true, qa_open, qb_hold, false, pair);
    emit(std::string("A_HOME static ") + (home_ok ? "PASS" : ("FAIL " + pair)));
    emit("scene: q_a=0 open, q_b=0.083 hold, part attached to arm_b_gripper_tcp, B joints = B_HANDOVER");
    double jump = 0;
    for (int j = 0; j < 6; ++j)
      jump = std::max(jump, std::abs(Q["A_HOME"][j] - Q["A_HANDOVER"][j]));
    const int n = std::max(1, static_cast<int>(std::ceil(jump / kStep)));
    bool hit = false;
    for (int s = 0; s <= n; ++s)
    {
      const double t = static_cast<double>(s) / n;
      std::vector<double> q(6);
      for (int j = 0; j < 6; ++j)
        q[j] = Q["A_HANDOVER"][j] + (Q["A_HOME"][j] - Q["A_HANDOVER"][j]) * t;
      std::string p;
      if (!sampleOk(*scene, model, q, Q["B_HANDOVER"], true, qa_open, qb_hold, false, p))
      {
        emit("direct first collision t=" + std::to_string(t) + " pair=" + p);
        std::ostringstream o;
        o << std::fixed << std::setprecision(6);
        o << "direct joints rad [";
        for (int j = 0; j < 6; ++j)
          o << (j ? ", " : "") << q[j];
        o << "]";
        emit(o.str());
        o.str("");
        o << std::fixed << std::setprecision(2) << "direct joints deg [";
        for (int j = 0; j < 6; ++j)
          o << (j ? ", " : "") << (q[j] * 180.0 / M_PI);
        o << "]";
        emit(o.str());
        hit = true;
        break;
      }
    }
    if (!hit)
      emit("direct interpolation has no collision");
    auto dumpq = [&](const char* name, const std::vector<double>& q) {
      std::ostringstream o;
      o << std::fixed << std::setprecision(6) << name << " rad [";
      for (int j = 0; j < 6; ++j)
        o << (j ? ", " : "") << q[j];
      o << "]";
      emit(o.str());
      o.str("");
      o << std::fixed << std::setprecision(2) << name << " deg [";
      for (int j = 0; j < 6; ++j)
        o << (j ? ", " : "") << (q[j] * 180.0 / M_PI);
      o << "]";
      emit(o.str());
    };
    dumpq("A_HANDOVER", Q["A_HANDOVER"]);
    dumpq("A_HOME", Q["A_HOME"]);
    dumpq("B_HANDOVER", Q["B_HANDOVER"]);
    if (!han_ok || !home_ok)
    {
      emit("endpoint not independently legal; OMPL not started");
      rclcpp::shutdown();
      return 3;
    }
    if (!node->has_parameter("planning_plugin"))
      node->declare_parameter("planning_plugin", std::string("ompl_interface/OMPLPlanner"));
    if (!node->has_parameter("planner_configs.RRTConnectkConfigDefault.type"))
    {
      node->declare_parameter("planner_configs.RRTConnectkConfigDefault.type",
                              std::string("geometric::RRTConnect"));
      node->declare_parameter("planner_configs.RRTConnectkConfigDefault.range", 0.0);
      node->declare_parameter("arm_a.default_planner_config", std::string("RRTConnectkConfigDefault"));
      node->declare_parameter("arm_a.planner_configs", std::vector<std::string>{"RRTConnectkConfigDefault"});
    }
    planning_pipeline::PlanningPipeline pipe(
        model, node, "", "ompl_interface/OMPLPlanner",
        std::vector<std::string>{"default_planner_request_adapters/AddTimeOptimalParameterization",
                                 "default_planner_request_adapters/FixWorkspaceBounds"});
    pipe.displayComputedMotionPlans(false);
    pipe.checkSolutionPaths(false);
    Seg best;
    best.status = "FAIL";
    best.block = "no_ompl_plan";
    best.id = "a_handover_to_home";
    best.move_a = true;
    best.start = Q["A_HANDOVER"];
    best.goal = Q["A_HOME"];
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
      moveit::core::RobotState start(model);
      start.setToDefaultValues();
      setJ(start, kA, Q["A_HANDOVER"]);
      setJ(start, kB, Q["B_HANDOVER"]);
      start.setVariablePosition("arm_a_gripper_joint", qa_open);
      start.setVariablePosition("arm_b_gripper_joint", qb_hold);
      start.update();
      planning_interface::MotionPlanRequest req;
      req.group_name = "arm_a";
      req.planner_id = "RRTConnectkConfigDefault";
      req.allowed_planning_time = 8.0;
      req.num_planning_attempts = 1;
      req.max_velocity_scaling_factor = 1.0;
      req.max_acceleration_scaling_factor = 1.0;
      moveit::core::robotStateToRobotStateMsg(start, req.start_state);
      req.start_state.is_diff = false;
      moveit_msgs::msg::Constraints goal;
      for (int j = 0; j < 6; ++j)
      {
        moveit_msgs::msg::JointConstraint jc;
        jc.joint_name = kA[j];
        jc.position = Q["A_HOME"][j];
        jc.tolerance_above = 1e-3;
        jc.tolerance_below = 1e-3;
        jc.weight = 1.0;
        goal.joint_constraints.push_back(jc);
      }
      req.goal_constraints.push_back(goal);
      planning_interface::MotionPlanResponse res;
      const bool called = pipe.generatePlan(scene, req, res);
      emit("OMPL RRTConnect attempt " + std::to_string(attempt) + " called=" +
           std::to_string(called) + " code=" + std::to_string(res.error_code_.val) +
           " time=" + std::to_string(res.planning_time_));
      if (!called || !res.trajectory_ || res.trajectory_->getWayPointCount() < 2)
        continue;
      Seg s = best;
      s.path.clear();
      s.method = "ompl_rrtconnect";
      s.block.clear();
      bool bad = false;
      for (size_t i = 0; i < res.trajectory_->getWayPointCount(); ++i)
      {
        std::vector<double> q(6);
        const auto& wp = res.trajectory_->getWayPoint(i);
        for (int j = 0; j < 6; ++j)
          q[j] = wp.getVariablePosition(kA[j]);
        std::string p;
        if (!sampleOk(*scene, model, q, Q["B_HANDOVER"], true, qa_open, qb_hold, false, p))
        {
          s.block = "postcheck " + p;
          bad = true;
          break;
        }
        s.path.push_back(q);
      }
      if (bad)
      {
        emit("  postcheck fail " + s.block);
        continue;
      }
      double ds = 0, de = 0;
      for (int j = 0; j < 6; ++j)
      {
        ds = std::max(ds, std::abs(s.path.front()[j] - Q["A_HANDOVER"][j]));
        de = std::max(de, std::abs(s.path.back()[j] - Q["A_HOME"][j]));
      }
      if (ds > 1e-3 || de > 1e-3)
      {
        emit("  rejected: endpoints moved ds=" + std::to_string(ds) + " de=" + std::to_string(de));
        continue;
      }
      s.start = Q["A_HANDOVER"];
      s.goal = Q["A_HOME"];
      s.status = "PASS";
      accum(s);
      emit("  cumJ1=" + std::to_string(s.cum_j1) + " cumJ2=" + std::to_string(s.cum_j2) +
           " cumJ6=" + std::to_string(s.cum_j6) + " points=" + std::to_string(s.path.size()));
      if (best.status != "PASS" || (s.cum_j1 + s.cum_j2) < (best.cum_j1 + best.cum_j2))
        best = s;
    }
    const std::string out =
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
        "/fr_task_ws/src/fr_task_planner/config/keypose_v1_six_face_trajectory.yaml";
    if (best.status == "PASS")
    {
      std::ifstream in(out);
      std::stringstream buf;
      buf << in.rdbuf();
      std::string text = buf.str();
      const auto pos = text.find("  - id: a_handover_to_home\n");
      const auto next = text.find("  - id: ", pos + 10);
      if (pos != std::string::npos && next != std::string::npos)
      {
        std::ostringstream block;
        writePath(block, best);
        text.replace(pos, next - pos, block.str());
        if (text.find("status: FAIL") == std::string::npos)
        {
          auto cpos = text.find("chain: ");
          if (cpos != std::string::npos)
            text.replace(cpos, std::string("chain: FAIL").size(), "chain: PASS");
        }
        std::ofstream os(out);
        os << text;
        emit("updated only a_handover_to_home");
      }
    }
    emit(std::string("A return OMPL ") + best.status + " " + best.block);
    rclcpp::shutdown();
    return best.status == "PASS" ? 0 : 3;
  }
  std::vector<Seg> segs;
  auto run = [&](const std::string& id, bool move_a, const std::vector<double>& start,
                 const std::vector<double>& goal, const std::vector<double>& fixed, double qa, double qb,
                 bool ignore_table, bool allow_cross = false) {
    Seg s;
    s.id = id;
    s.move_a = move_a;
    s.ignore_table = ignore_table;
    s.allow_cross = allow_cross;
    s.start = start;
    s.goal = goal;
    const bool ok = planSeg(*scene, model, s, fixed, qa, qb);
    emit(id + " " + s.status + " " + s.method + " " + s.block + " cumJ1=" + std::to_string(s.cum_j1) +
         " cumJ2=" + std::to_string(s.cum_j2) + " cumJ6=" + std::to_string(s.cum_j6));
    segs.push_back(s);
    return ok;
  };

  bool all = true;
  all &= run("home_to_pregrasp", true, Q["A_HOME"], Q["A_PREGRASP"], Q["B_HOME"], qa_open, qb_open, false);
  all &= run("pregrasp_to_grasp", true, Q["A_PREGRASP"], Q["A_GRASP"], Q["B_HOME"], qa_open, qb_open, false);
  addCylinder(*scene, "arm_a_gripper_tcp", TA, true, kTouchA);
  all &= run("grasp_to_lift", true, Q["A_GRASP"], Q["A_LIFT"], Q["B_HOME"], qa_grasp, qb_open, true);
  all &= run("lift_to_face1", true, Q["A_LIFT"], Q["A_FACE1"], Q["B_HOME"], qa_grasp, qb_open, false);
  all &= run("face1_to_face2", true, Q["A_FACE1"], Q["A_FACE2"], Q["B_HOME"], qa_grasp, qb_open, false);
  all &= run("face2_to_face3", true, Q["A_FACE2"], Q["A_FACE3"], Q["B_HOME"], qa_grasp, qb_open, false);
  all &= run("b_home_to_pre_handover", false, Q["B_HOME"], Q["B_PRE_HANDOVER"], Q["A_FACE3"], qa_grasp, qb_open,
             false);
  all &= run("face3_to_pre_handover", true, Q["A_FACE3"], Q["A_PRE_HANDOVER"], Q["B_PRE_HANDOVER"], qa_grasp,
             qb_open, false);
  all &= run("pre_handover_to_handover", true, Q["A_PRE_HANDOVER"], Q["A_HANDOVER"], Q["B_PRE_HANDOVER"],
             qa_grasp, qb_open, false);
  all &= run("b_pre_to_handover", false, Q["B_PRE_HANDOVER"], Q["B_HANDOVER"], Q["A_HANDOVER"], qa_grasp, qb_open,
             false);
  addCylinder(*scene, "arm_b_gripper_tcp", TB, true, kTouchB);
  all &= run("a_handover_to_home", true, Q["A_HANDOVER"], Q["A_HOME"], Q["B_HANDOVER"], qa_open, qb_hold, false,
             true);
  all &= run("b_handover_to_face4", false, Q["B_HANDOVER"], Q["B_FACE4"], Q["A_HOME"], qa_open, qb_hold, false);
  all &= run("face4_to_face5", false, Q["B_FACE4"], Q["B_FACE5"], Q["A_HOME"], qa_open, qb_hold, false);
  all &= run("face5_to_face6", false, Q["B_FACE5"], Q["B_FACE6"], Q["A_HOME"], qa_open, qb_hold, false);

  const std::string out =
      std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
      "/fr_task_ws/src/fr_task_planner/config/keypose_v1_six_face_trajectory.yaml";
  std::ofstream os(out);
  os << "task_version: KEYPOSE_V1_SIX_FACE\n";
  os << "execution: false\n";
  os << "baseline_overwritten: false\n";
  os << "keypose_dir: " << dir << "\n";
  os << "selected_candidate: 1\n";
  os << "speed_scale: 0.2\n";
  os << "scale_applied_by: dual7 scaleSegment(trajectory_speed_scale)\n";
  os << "grasp_post_close_wait_sec: 2.0\n";
  os << "handover_post_close_wait_sec: 2.0\n";
  os << "place_stage: absent\n";
  os << "place_note: \"DUAL-7 ends at B +Z / B_FACE6. No confirmed place keypose. None added.\"\n";
  os << "chain: " << (all ? "PASS" : "FAIL") << "\n";
  os.setf(std::ios::fixed);
  os << std::setprecision(12);
  os << "home_a: [";
  for (int i = 0; i < 6; ++i)
    os << (i ? ", " : "") << Q["A_HOME"][i];
  os << "]\nhome_b: [";
  for (int i = 0; i < 6; ++i)
    os << (i ? ", " : "") << Q["B_HOME"][i];
  os << "]\nstages:\n";
  os << "  - id: gripper_activate_a\n    kind: gripper_activate\n    moving: arm_a\n    startup: true\n";
  os << "  - id: gripper_open_a_startup\n    kind: gripper_open\n    moving: arm_a\n    startup: true\n";
  os << "  - id: gripper_activate_b\n    kind: gripper_activate\n    moving: arm_b\n    startup: true\n";
  os << "  - id: gripper_open_b_startup\n    kind: gripper_open\n    moving: arm_b\n    startup: true\n";
  auto dumpMotion = [&](const std::string& id) {
    for (const auto& s : segs)
      if (s.id == id)
        writePath(os, s);
  };
  auto grip = [&](const std::string& id, const char* kind, const char* arm) {
    os << "  - id: " << id << "\n    kind: " << kind << "\n    moving: " << arm << "\n";
  };
  dumpMotion("home_to_pregrasp");
  dumpMotion("pregrasp_to_grasp");
  grip("gripper_close_a", "gripper_close", "arm_a");
  os << "    wait_after_success_param: grasp_post_close_wait_sec\n";
  dumpMotion("grasp_to_lift");
  dumpMotion("lift_to_face1");
  dumpMotion("face1_to_face2");
  dumpMotion("face2_to_face3");
  dumpMotion("b_home_to_pre_handover");
  dumpMotion("face3_to_pre_handover");
  dumpMotion("pre_handover_to_handover");
  dumpMotion("b_pre_to_handover");
  grip("gripper_close_b", "gripper_close", "arm_b");
  os << "    wait_after_success_param: handover_post_close_wait_sec\n";
  grip("gripper_open_a", "gripper_open", "arm_a");
  os << "  - id: attachment_transfer\n    kind: attachment_transfer\n    moving: arm_b\n";
  dumpMotion("a_handover_to_home");
  dumpMotion("b_handover_to_face4");
  dumpMotion("face4_to_face5");
  dumpMotion("face5_to_face6");
  os.close();
  emit(std::string("wrote ") + out + " chain=" + (all ? "PASS" : "FAIL"));
  rclcpp::shutdown();
  return all ? 0 : 3;
}
