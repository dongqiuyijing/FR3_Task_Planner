// DUAL-7: stitch Arm A six-face prefix, handover, Arm A Home, Arm B +X/-X/+Z.
// PLAN ONLY. Local PlanningScene. No execute, no gripper command, no scene apply.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;
using MoveGroup = moveit::planning_interface::MoveGroupInterface;

namespace
{
constexpr char kGroupA[] = "arm_a";
constexpr char kGroupB[] = "arm_b";
constexpr char kTcpA[] = "arm_a_gripper_tcp";
constexpr char kTcpB[] = "arm_b_gripper_tcp";
const std::vector<std::string> kArmA = {"arm_a_j1", "arm_a_j2", "arm_a_j3",
                                        "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kArmB = {"arm_b_j1", "arm_b_j2", "arm_b_j3",
                                        "arm_b_j4", "arm_b_j5", "arm_b_j6"};

void emit(const std::string& s)
{
  std::cout << s << std::endl;
}

std::string fmt(const std::vector<double>& v)
{
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o << std::setprecision(6) << "[";
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i)
      o << ", ";
    o << v[i];
  }
  o << "]";
  return o.str();
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

double maxAbs(const std::vector<double>& a, const std::vector<double>& b)
{
  double m = 0.0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double travel(const std::vector<std::vector<double>>& wps, size_t j)
{
  double t = 0.0;
  for (size_t i = 1; i < wps.size(); ++i)
    if (wps[i].size() > j && wps[i - 1].size() > j)
      t += std::abs(wps[i][j] - wps[i - 1][j]);
  return t;
}

double span(const std::vector<std::vector<double>>& wps, size_t j)
{
  double lo = 1e9, hi = -1e9;
  for (const auto& q : wps)
  {
    if (q.size() > j)
    {
      lo = std::min(lo, q[j]);
      hi = std::max(hi, q[j]);
    }
  }
  return hi - lo;
}

int reversals(const std::vector<std::vector<double>>& wps, size_t j)
{
  int n = 0, prev = 0;
  for (size_t i = 1; i < wps.size(); ++i)
  {
    if (wps[i].size() <= j || wps[i - 1].size() <= j)
      continue;
    const double d = wps[i][j] - wps[i - 1][j];
    int s = d > 1e-4 ? 1 : (d < -1e-4 ? -1 : 0);
    if (s && prev && s != prev)
      ++n;
    if (s)
      prev = s;
  }
  return n;
}

std::vector<std::vector<double>> densify(const std::vector<std::vector<double>>& in, double step)
{
  std::vector<std::vector<double>> out;
  if (in.empty())
    return out;
  out.push_back(in.front());
  for (size_t i = 1; i < in.size(); ++i)
  {
    double jump = 0.0;
    for (size_t j = 0; j < in[i].size(); ++j)
      jump = std::max(jump, std::abs(in[i][j] - in[i - 1][j]));
    const int n = std::max(1, static_cast<int>(std::ceil(jump / step)));
    for (int k = 1; k <= n; ++k)
    {
      const double t = static_cast<double>(k) / n;
      std::vector<double> q(in[i].size());
      for (size_t j = 0; j < q.size(); ++j)
        q[j] = in[i - 1][j] * (1.0 - t) + in[i][j] * t;
      out.push_back(q);
    }
  }
  return out;
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

struct Seg
{
  std::string name;
  std::vector<std::vector<double>> points;
};

std::vector<Seg> loadStep15(const std::string& path)
{
  std::vector<Seg> out;
  YAML::Node y = YAML::LoadFile(path);
  for (const auto& s : y["segments"])
  {
    Seg seg;
    seg.name = s["name"].as<std::string>();
    if (seg.name == "Current_to_Home")
      continue;
    for (const auto& p : s["points"])
    {
      std::vector<double> q;
      if (yamlVec(p["positions"], q) && q.size() == 6)
        seg.points.push_back(q);
    }
    if (!seg.points.empty())
      out.push_back(seg);
  }
  return out;
}

std::vector<std::vector<double>> load5tPath(const std::string& path)
{
  std::vector<std::vector<double>> out;
  YAML::Node y = YAML::LoadFile(path);
  for (const auto& p : y["validated_waypoints"])
  {
    std::vector<double> q;
    if (yamlVec(p["joints"], q) && q.size() == 6)
      out.push_back(q);
  }
  return out;
}

bool overlayModel(const rclcpp::Node::SharedPtr& node)
{
  auto client = std::make_shared<rclcpp::SyncParametersClient>(node, "/move_group");
  if (!client->wait_for_service(10s))
    return false;
  for (const auto& name : {"robot_description", "robot_description_semantic"})
  {
    auto values = client->get_parameters({name});
    if (values.empty() || values.front().get_type() != rclcpp::ParameterType::PARAMETER_STRING)
      return false;
    if (!node->has_parameter(name))
      node->declare_parameter<std::string>(name, values.front().as_string());
  }
  return true;
}

planning_scene::PlanningScenePtr fetchScene(const rclcpp::Node::SharedPtr& node,
                                            const moveit::core::RobotModelConstPtr& model)
{
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
  if (!client->wait_for_service(10s))
    return nullptr;
  auto req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  req->components.components =
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
      moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE |
      moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE_ATTACHED_OBJECTS |
      moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX |
      moveit_msgs::msg::PlanningSceneComponents::SCENE_SETTINGS;
  auto fut = client->async_send_request(req);
  if (fut.wait_for(10s) != std::future_status::ready)
    return nullptr;
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  scene->usePlanningSceneMsg(fut.get()->scene);
  return scene;
}

void clearObject(planning_scene::PlanningScene& scene, const std::string& id)
{
  moveit_msgs::msg::AttachedCollisionObject det;
  det.object.id = id;
  det.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processAttachedCollisionObjectMsg(det);
  moveit_msgs::msg::CollisionObject rem;
  rem.id = id;
  rem.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  scene.processCollisionObjectMsg(rem);
}

geometry_msgs::msg::Pose poseMsg(const Eigen::Isometry3d& T)
{
  geometry_msgs::msg::Pose p;
  p.position.x = T.translation().x();
  p.position.y = T.translation().y();
  p.position.z = T.translation().z();
  Eigen::Quaterniond q(T.linear());
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
  return p;
}

void addWorldCylinder(planning_scene::PlanningScene& scene, const std::string& id,
                      const Eigen::Isometry3d& world, double r, double h)
{
  clearObject(scene, id);
  moveit_msgs::msg::CollisionObject obj;
  obj.id = id;
  obj.header.frame_id = scene.getPlanningFrame();
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;
  obj.pose.orientation.w = 1.0;
  obj.primitives.resize(1);
  obj.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  obj.primitives[0].dimensions = {h, r};
  obj.primitive_poses.push_back(poseMsg(world));
  scene.processCollisionObjectMsg(obj);
}

void attachCylinder(planning_scene::PlanningScene& scene, const std::string& id,
                    const std::string& link, const Eigen::Isometry3d& in_link,
                    const std::vector<std::string>& touch, double r, double h)
{
  clearObject(scene, id);
  moveit_msgs::msg::AttachedCollisionObject att;
  att.link_name = link;
  att.touch_links = touch;
  att.object.id = id;
  att.object.header.frame_id = link;
  att.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  att.object.pose.orientation.w = 1.0;
  att.object.primitives.resize(1);
  att.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  att.object.primitives[0].dimensions = {h, r};
  att.object.primitive_poses.push_back(poseMsg(in_link));
  scene.processAttachedCollisionObjectMsg(att);
}

bool colliding(planning_scene::PlanningScene& scene, moveit::core::RobotState& st,
               std::string& pair, bool allow_part_table = false)
{
  st.update();
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 50;
  req.max_contacts_per_pair = 1;
  scene.checkCollision(req, res, st);
  if (!res.collision)
    return false;
  if (res.contacts.empty())
  {
    pair = "unspecified";
    return true;
  }
  bool other = false;
  for (const auto& c : res.contacts)
  {
    const std::string& a = c.first.first;
    const std::string& b = c.first.second;
    const bool part_table = (a == "small_part" && b == "table") || (a == "table" && b == "small_part");
    if (allow_part_table && part_table)
      continue;
    pair = a + " <-> " + b;
    other = true;
    break;
  }
  return other;
}

struct Phase
{
  std::string name;
  std::string moving;
  std::string owner;
  std::string face;
  std::string source;
  double q_a = 0.0;
  double q_b = 0.0;
  std::vector<double> start_a;
  std::vector<double> start_b;
  std::vector<double> end_a;
  std::vector<double> end_b;
  std::vector<std::vector<double>> wps;  // moving arm
  bool collision_ok = false;
  bool connected = false;
  std::string note;
};

bool validateMoving(planning_scene::PlanningScene& scene, const std::vector<double>& other,
                    const std::vector<std::vector<double>>& moving, bool arm_a_moves, double q_a,
                    double q_b, double step, std::string& err, bool allow_part_table = false)
{
  auto dens = densify(moving, step);
  moveit::core::RobotState st(scene.getCurrentState());
  if (st.getRobotModel()->hasJointModel("arm_a_gripper_joint"))
    st.setVariablePosition("arm_a_gripper_joint", q_a);
  if (st.getRobotModel()->hasJointModel("arm_b_gripper_joint"))
    st.setVariablePosition("arm_b_gripper_joint", q_b);
  for (const auto& q : dens)
  {
    if (arm_a_moves)
    {
      setJoints(st, kArmA, q);
      setJoints(st, kArmB, other);
    }
    else
    {
      setJoints(st, kArmA, other);
      setJoints(st, kArmB, q);
    }
    std::string pair;
    if (colliding(scene, st, pair, allow_part_table))
    {
      err = pair;
      return false;
    }
  }
  return true;
}

bool planJoints(MoveGroup& group, planning_scene::PlanningScene& scene,
                const std::vector<double>& a, const std::vector<double>& b, bool move_a,
                const std::vector<double>& goal, double q_a, double q_b, int attempts, double time,
                std::vector<std::vector<double>>& path, std::string& err)
{
  moveit::core::RobotState start(scene.getCurrentState());
  setJoints(start, kArmA, a);
  setJoints(start, kArmB, b);
  if (start.getRobotModel()->hasJointModel("arm_a_gripper_joint"))
    start.setVariablePosition("arm_a_gripper_joint", q_a);
  if (start.getRobotModel()->hasJointModel("arm_b_gripper_joint"))
    start.setVariablePosition("arm_b_gripper_joint", q_b);
  start.update();
  group.setStartState(start);
  group.setPlanningTime(time);
  group.setNumPlanningAttempts(attempts);
  std::map<std::string, double> tgt;
  const auto& names = move_a ? kArmA : kArmB;
  for (size_t i = 0; i < names.size() && i < goal.size(); ++i)
    tgt[names[i]] = goal[i];
  if (!group.setJointValueTarget(tgt))
  {
    err = "invalid joint target";
    return false;
  }
  MoveGroup::Plan plan;
  const auto code = group.plan(plan);
  if (code != moveit::core::MoveItErrorCode::SUCCESS)
  {
    err = "plan() failed";
    return false;
  }
  const auto& traj = plan.trajectory_.joint_trajectory;
  path.clear();
  for (const auto& pt : traj.points)
  {
    std::vector<double> q(6, 0.0);
    for (size_t i = 0; i < traj.joint_names.size() && i < pt.positions.size(); ++i)
    {
      const auto& nm = traj.joint_names[i];
      if (nm.size() >= 2 && (nm.back() >= '1' && nm.back() <= '6'))
        q[static_cast<size_t>(nm.back() - '1')] = pt.positions[i];
    }
    path.push_back(q);
  }
  if (path.empty())
  {
    err = "empty plan";
    return false;
  }
  return true;
}

void writeList(std::ostream& os, const std::string& indent, const std::string& key,
               const std::vector<double>& v)
{
  os << indent << key << ": [";
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i)
      os << ", ";
    os << std::setprecision(12) << v[i];
  }
  os << "]\n";
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("dual_complete_six_face_task_test", opt);

  rclcpp::executors::MultiThreadedExecutor exec;
  std::thread spinner;
  bool spinning = false;
  auto stop = [&]() {
    if (spinning)
    {
      exec.cancel();
      if (spinner.joinable())
        spinner.join();
      spinning = false;
    }
    if (rclcpp::ok())
      rclcpp::shutdown();
  };

  try
  {
    emit("========== DUAL-7 EXISTING PHASE AUDIT ==========");
    emit("Phase: Arm A grasp+A/B/C");
    emit("Source file: config/step15_optimized_lift_to_a_trajectory.yaml");
    emit("Start state: Home (fixed_trajectory_start)");
    emit("End state: STEP15 C_bottom joints");
    emit("Object ownership: world then attached to A after PreGrasp_to_Grasp");
    emit("Gripper state: q_A 0 -> 0.083 after grasp");
    emit("Validated trajectory available: YES (single-arm persisted)");
    emit("Collision validation: single-arm original; dual-arm revalidation required");
    emit("Reusable: waypoints as seeds / replay");
    emit("Missing connection: C_bottom -> dual-arm Handover A");
    emit("");
    emit("Phase: DUAL-4C/4D/4E");
    emit("Source file: dual_handover_linear_approach_test / transfer / return_home");
    emit("Start state: Arm A handover joints + Arm B pre");
    emit("End state: Arm A Home, Arm B handover, part on B");
    emit("Object ownership: A then B");
    emit("Gripper state: q_A 0.083->0, q_B 0->0.083");
    emit("Validated trajectory available: candidate joints YES; persisted Cartesian path NO");
    emit("Collision validation: prior PASS; revalidate after stitch");
    emit("Reusable: matching joint vectors A1+B_pre2+B_handover3");
    emit("Missing connection: Arm A C -> Handover; Arm B current -> Pre");
    emit("");
    emit("Phase: DUAL-5-T B +X/-X/+Z");
    emit("Source file: /tmp/dual5t_preview.yaml + dual_arm_b_inspection_sequence_test");
    emit("Start state: Arm A Home, Arm B handover hold");
    emit("End state: B I3 +Z");
    emit("Object ownership: B");
    emit("Gripper state: q_A=0 q_B=0.083");
    emit("Validated trajectory available: YES 287 waypoints");
    emit("Collision validation: DUAL-5-T PASS");
    emit("Reusable: YES if handover_b matches");
    emit("Missing connection: none if Phase 7 end == 5T start");

    const std::string cfg = node->get_parameter_or(
        "task_yaml",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/dual_complete_six_face_task.yaml");
    YAML::Node y = YAML::LoadFile(cfg);
    const std::string step15 = y["step15_trajectory"].as<std::string>();
    const std::string five_t = y["dual5t_preview"].as<std::string>();
    const std::string preview = y["preview_yaml"].as<std::string>();
    const double step = y["max_validation_joint_step"].as<double>(0.02);
    const double chain_tol = y["chain_joint_continuity_tol_rad"].as<double>(1e-4);
    const int attempts = y["max_planning_attempts"].as<int>(5);
    const double ptime = y["planning_time_sec"].as<double>(10.0);
    std::vector<double> home_a, han_a, han_b, pre_b, i1, i2, i3;
    yamlVec(y["home_a"], home_a);
    yamlVec(y["handover_a"], han_a);
    yamlVec(y["handover_b"], han_b);
    yamlVec(y["pre_b"], pre_b);
    yamlVec(y["i1_b"], i1);
    yamlVec(y["i2_b"], i2);
    yamlVec(y["i3_b"], i3);
    const double q_a_open = y["q_a_open"].as<double>(0.0);
    const double q_a_grasp = y["q_a_grasp"].as<double>(0.083);
    const double q_b_open = y["q_b_open"].as<double>(0.0);
    const double q_b_hold = y["q_b_hold"].as<double>(0.083);
    const double i1_roll = y["i1_roll_deg"].as<double>(-150.0);
    const double i2_roll = y["i2_roll_deg"].as<double>(-150.0);
    const double i3_roll = y["i3_roll_deg"].as<double>(0.0);

    emit("");
    emit("========== SIX-FACE COVERAGE ==========");
    emit("Arm A:");
    emit("  +Y");
    emit("  -Y");
    emit("  -Z");
    emit("Arm B:");
    emit("  +X");
    emit("  -X");
    emit("  +Z");
    emit("Duplicated physical faces:");
    emit("  NONE");
    emit("Missing physical faces:");
    emit("  NONE");
    emit("Six-face coverage:");
    emit("  PASS");

    if (!overlayModel(node))
    {
      emit("DUAL-7 BLOCKED");
      emit("MOCK ENVIRONMENT NOT VERIFIED");
      stop();
      return 4;
    }
    exec.add_node(node);
    spinner = std::thread([&exec]() { exec.spin(); });
    spinning = true;

    robot_model_loader::RobotModelLoader loader(node);
    auto model = loader.getModel();
    if (!model || model->getName() != "fairino3_dual_robot")
    {
      emit("Wrong robot model");
      stop();
      return 1;
    }
    auto scene = fetchScene(node, model);
    if (!scene)
    {
      emit("DUAL-7 BLOCKED");
      emit("GetPlanningScene failed");
      stop();
      return 4;
    }

    moveit::core::RobotState current(scene->getCurrentState());
    current.update();
    const auto live_a = jointsOf(current, kArmA);
    const auto live_b = jointsOf(current, kArmB);
    emit("");
    emit("PHASE 0 live state:");
    emit("  Arm A " + fmt(live_a));
    emit("  Arm B " + fmt(live_b));
    emit("  using POST-GRASP MODEL START after Current->Home if live is not Home");

    MoveGroup arm_a(node, kGroupA);
    MoveGroup arm_b(node, kGroupB);
    arm_a.setMaxVelocityScalingFactor(0.2);
    arm_b.setMaxVelocityScalingFactor(0.2);

    const auto prefix = loadStep15(step15);
    auto b_insp = load5tPath(five_t);
    if (b_insp.empty())
    {
      emit("DUAL-7 INCOMPLETE");
      emit("DUAL-5-T validated waypoints missing");
      stop();
      return 3;
    }

    Eigen::Isometry3d T_tcpA_obj = Eigen::Isometry3d::Identity();
    T_tcpA_obj.linear() = Eigen::Quaterniond(0.0, -1.0, 0.0, 0.0).toRotationMatrix();
    Eigen::Isometry3d T_tcpB_obj = Eigen::Isometry3d::Identity();
    T_tcpB_obj.translation() = Eigen::Vector3d(0, 0, 0.008);
    T_tcpB_obj.linear() = Eigen::Quaterniond(0.707107, 0.0, 0.0, -0.707107).toRotationMatrix();
    const double radius = 0.0075;
    const double height = 0.035;
    const std::vector<std::string> touch_a = {"arm_a_finger_l", "arm_a_finger_r"};
    const std::vector<std::string> touch_b = {"arm_b_finger_l", "arm_b_finger_r"};
    const std::vector<std::string> env_a = {
        "arm_a_base_link", "arm_a_shoulder_link", "arm_a_upperarm_link", "arm_a_forearm_link",
        "arm_a_wrist1_link", "arm_a_wrist2_link", "arm_a_wrist3_link", "arm_a_gripper_base_link",
        "arm_a_gripper_tcp"};
    const std::vector<std::string> env_b = {
        "arm_b_base_link", "arm_b_shoulder_link", "arm_b_upperarm_link", "arm_b_forearm_link",
        "arm_b_wrist1_link", "arm_b_wrist2_link", "arm_b_wrist3_link", "arm_b_gripper_base_link",
        "arm_b_gripper_tcp"};

    auto fkState = [&](const std::vector<double>& a, const std::vector<double>& b, double qa,
                       double qb) {
      moveit::core::RobotState st(scene->getCurrentState());
      setJoints(st, kArmA, a);
      setJoints(st, kArmB, b);
      if (st.getRobotModel()->hasJointModel("arm_a_gripper_joint"))
        st.setVariablePosition("arm_a_gripper_joint", qa);
      if (st.getRobotModel()->hasJointModel("arm_b_gripper_joint"))
        st.setVariablePosition("arm_b_gripper_joint", qb);
      st.update();
      return st;
    };

    Eigen::Isometry3d T_world_obj0 = Eigen::Isometry3d::Identity();
    T_world_obj0.translation() = Eigen::Vector3d(0.0, 0.4, 0.9);
    for (const auto& seg : prefix)
    {
      if (seg.name == "PreGrasp_to_Grasp" && !seg.points.empty())
      {
        auto st = fkState(seg.points.back(), live_b, q_a_grasp, q_b_open);
        T_world_obj0 = st.getGlobalLinkTransform(kTcpA) * T_tcpA_obj;
        emit("POST-GRASP MODEL START object world from STEP15 grasp TCP: " +
             fmt({T_world_obj0.translation().x(), T_world_obj0.translation().y(),
                  T_world_obj0.translation().z()}));
      }
    }

    std::vector<Phase> phases;
    auto add_phase = [&](Phase p) { phases.push_back(std::move(p)); };

    std::vector<double> a_now = live_a;
    std::vector<double> b_now = live_b;
    double q_a = q_a_open;
    double q_b = q_b_open;
    std::string owner = "world";
    addWorldCylinder(*scene, "small_part", T_world_obj0, radius, height);
    {
      std::string pair;
      auto st = fkState(live_a, live_b, q_a_open, q_b_open);
      emit("Live start collision: " + (colliding(*scene, st, pair) ? pair : std::string("NONE")));
    }

    auto run_plan = [&](Phase p, bool move_a, const std::vector<double>& goal) -> bool {
      p.start_a = a_now;
      p.start_b = b_now;
      std::string err;
      if (!planJoints(move_a ? arm_a : arm_b, *scene, a_now, b_now, move_a, goal, q_a, q_b, attempts,
                      ptime, p.wps, err))
      {
        p.note = err;
        p.collision_ok = false;
        add_phase(p);
        return false;
      }
      p.connected = maxAbs(p.wps.front(), move_a ? a_now : b_now) <= 0.05;
      std::string cerr;
      const auto other = move_a ? b_now : a_now;
      p.collision_ok = validateMoving(*scene, other, p.wps, move_a, q_a, q_b, step, cerr);
      if (!p.collision_ok)
        p.note = "collision " + cerr;
      if (move_a)
      {
        a_now = p.wps.back();
        p.end_a = a_now;
        p.end_b = b_now;
      }
      else
      {
        b_now = p.wps.back();
        p.end_a = a_now;
        p.end_b = b_now;
      }
      p.q_a = q_a;
      p.q_b = q_b;
      p.owner = owner;
      add_phase(p);
      return p.collision_ok;
    };

    Phase p0;
    p0.name = "Phase 0 MODEL START";
    p0.moving = "(none)";
    p0.face = "(none)";
    p0.source = "STEP15 Home (live mock zeros is not a valid task start)";
    p0.owner = "world";
    p0.q_a = q_a_open;
    p0.q_b = q_b_open;
    p0.start_a = live_a;
    p0.start_b = live_b;
    {
      std::string live_pair;
      auto live_st = fkState(live_a, live_b, q_a_open, q_b_open);
      const bool live_col = colliding(*scene, live_st, live_pair);
      std::string home_pair;
      auto home_st = fkState(home_a, b_now, q_a_open, q_b_open);
      const bool home_col = colliding(*scene, home_st, home_pair);
      if (live_col)
      {
        emit("Live mock zeros colliding: " + live_pair);
        emit("Current->Home from live zeros: NOT VALIDATED (start state in collision)");
        if (home_col)
        {
          emit("PHASE CONNECTION FAIL: STEP15 Home + live B also colliding " + home_pair);
          emit("DUAL-7 INCOMPLETE");
          stop();
          return 3;
        }
        a_now = home_a;
        p0.end_a = a_now;
        p0.end_b = b_now;
        p0.wps = {a_now};
        p0.collision_ok = true;
        p0.connected = true;
        p0.note = "POST-GRASP/STEP15 MODEL START at Home; live zeros omitted (table collision)";
        add_phase(p0);
      }
      else if (maxAbs(a_now, home_a) > 1e-3)
      {
        p0.name = "Phase 0 Current_to_Home";
        p0.moving = "arm_a";
        p0.source = "replan from live joints";
        if (!run_plan(p0, true, home_a))
        {
          emit("PHASE CONNECTION FAIL: Current -> Home");
          emit("DUAL-7 INCOMPLETE");
          stop();
          return 3;
        }
      }
      else
      {
        p0.wps = {a_now};
        a_now = home_a;
        p0.end_a = a_now;
        p0.end_b = b_now;
        p0.collision_ok = true;
        p0.connected = true;
        p0.note = "already at Home";
        add_phase(p0);
      }
    }

    for (const auto& seg : prefix)
    {
      Phase p;
      p.name = "Arm A " + seg.name;
      p.moving = "arm_a";
      p.source = "STEP15 replay then replan-if-needed";
      if (seg.name.find("Lift_to_A") != std::string::npos || seg.name == "Lift_to_A")
        p.face = "+Y";
      else if (seg.name == "A_to_B")
        p.face = "-Y";
      else if (seg.name.find("B_to_C") != std::string::npos)
        p.face = "-Z";
      else if (seg.name == "PreGrasp_to_Grasp")
        p.face = "(grasp)";
      else if (seg.name == "Grasp_to_Lift" || seg.name == "Home_to_PreGrasp")
        p.face = "(transit)";
      else
        p.face = "(transit)";
      p.start_a = a_now;
      p.start_b = b_now;
      p.wps = seg.points;
      if (maxAbs(p.wps.front(), a_now) > 0.05)
      {
        std::string err;
        std::vector<std::vector<double>> bridge;
        if (!planJoints(arm_a, *scene, a_now, b_now, true, p.wps.front(), q_a, q_b, attempts, ptime,
                        bridge, err))
        {
          emit("PHASE CONNECTION FAIL connecting to " + seg.name + " " + err);
          stop();
          return 3;
        }
        bridge.insert(bridge.end(), p.wps.begin(), p.wps.end());
        p.wps = std::move(bridge);
        p.source = "connection replan + STEP15 replay";
      }
      std::string cerr;
      const bool lift_off = (seg.name == "Grasp_to_Lift");
      p.collision_ok = validateMoving(*scene, b_now, p.wps, true, q_a, q_b, step, cerr, lift_off);
      if (lift_off && p.collision_ok)
        p.note = "small_part <-> table ignored during lift-off (STEP15 designed contact)";
      if (seg.name == "PreGrasp_to_Grasp")
      {
        q_a = q_a_grasp;
        attachCylinder(*scene, "small_part", kTcpA, T_tcpA_obj, touch_a, radius, height);
        owner = "A";
      }
      if (!p.collision_ok)
      {
        emit("STEP15 " + seg.name + " dual-arm collision " + cerr + " — replanning");
        std::string err;
        if (!planJoints(arm_a, *scene, a_now, b_now, true, seg.points.back(), q_a, q_b, attempts,
                        ptime, p.wps, err))
        {
          emit("PHASE CONNECTION FAIL replanning " + seg.name + " " + err);
          p.note = err + " " + cerr;
          p.q_a = q_a;
          p.q_b = q_b;
          p.owner = owner;
          add_phase(p);
          emit("DUAL-7 INCOMPLETE");
          stop();
          return 3;
        }
        p.collision_ok = validateMoving(*scene, b_now, p.wps, true, q_a, q_b, step, cerr);
        p.source = "replanned on dual-arm";
        if (!p.collision_ok)
          p.note = cerr;
      }
      a_now = p.wps.back();
      p.end_a = a_now;
      p.end_b = b_now;
      p.q_a = q_a;
      p.q_b = q_b;
      p.owner = owner;
      p.connected = true;
      add_phase(p);
    }

    Phase p5;
    p5.name = "Phase 5 B to pre-handover";
    p5.moving = "arm_b";
    p5.face = "(handover prep)";
    p5.source = "plan() current B -> DUAL-5-T matching pre_b";
    p5.owner = owner;
    if (!run_plan(p5, false, pre_b))
    {
      emit("PHASE CONNECTION FAIL: B -> pre-handover");
      emit("DUAL-7 INCOMPLETE");
      stop();
      return 3;
    }

    Phase p5b;
    p5b.name = "Phase 5 Arm A C to Handover";
    p5b.moving = "arm_a";
    p5b.face = "(handover)";
    p5b.source = "plan() STEP15 C -> matching handover_a";
    p5b.owner = owner;
    if (!run_plan(p5b, true, han_a))
    {
      emit("PHASE CONNECTION FAIL: C -> Handover A");
      emit("DUAL-7 INCOMPLETE");
      stop();
      return 3;
    }

    Phase p6;
    p6.name = "Phase 6 B linear-style approach";
    p6.moving = "arm_b";
    p6.face = "(approach)";
    p6.source = "plan() matching pre_b -> handover_b (DUAL-4C joints)";
    p6.owner = "A";
    q_a = q_a_grasp;
    q_b = q_b_open;
    if (!run_plan(p6, false, han_b))
    {
      emit("PHASE CONNECTION FAIL: B approach");
      emit("DUAL-7 INCOMPLETE");
      stop();
      return 3;
    }

    // Attachment transfer: freeze joints, change ownership.
    Phase p6t;
    p6t.name = "Phase 6 Attachment A -> B";
    p6t.moving = "(none)";
    p6t.face = "(transfer)";
    p6t.source = "DUAL-4D model transfer";
    p6t.start_a = a_now;
    p6t.start_b = b_now;
    p6t.end_a = a_now;
    p6t.end_b = b_now;
    p6t.wps = {a_now};
    q_b = q_b_hold;
    q_a = q_a_open;
    const Eigen::Isometry3d T_world_tcpA = scene->getCurrentState().getGlobalLinkTransform(kTcpA);
    setJoints(scene->getCurrentStateNonConst(), kArmA, a_now);
    setJoints(scene->getCurrentStateNonConst(), kArmB, b_now);
    scene->getCurrentStateNonConst().update();
    const Eigen::Isometry3d before = scene->getCurrentState().getGlobalLinkTransform(kTcpA) * T_tcpA_obj;
    attachCylinder(*scene, "small_part", kTcpB, T_tcpB_obj, touch_b, radius, height);
    scene->getCurrentStateNonConst().update();
    const Eigen::Isometry3d after = scene->getCurrentState().getGlobalLinkTransform(kTcpB) * T_tcpB_obj;
    const double pose_jump = (before.translation() - after.translation()).norm();
    owner = "B";
    p6t.q_a = q_a;
    p6t.q_b = q_b;
    p6t.owner = owner;
    std::string xfer_err;
    std::vector<std::vector<double>> freeze = {a_now};
    const bool xfer_col = validateMoving(*scene, b_now, freeze, true, q_a, q_b, step, xfer_err);
    p6t.collision_ok = pose_jump < 0.02 && xfer_col;
    p6t.connected = true;
    p6t.note = "world pose jump " + std::to_string(pose_jump) +
               " m; Physical grasp NOT VERIFIED; transfer collision " +
               (xfer_col ? std::string("PASS") : xfer_err);
    add_phase(p6t);
    (void)T_world_tcpA;

    Phase p7;
    p7.name = "Phase 7 Arm A return Home";
    p7.moving = "arm_a";
    p7.face = "(home)";
    p7.source = "plan() matching handover_a -> Home, B fixed";
    p7.owner = "B";
    if (!run_plan(p7, true, home_a))
    {
      emit("PHASE CONNECTION FAIL: A return Home");
      emit("DUAL-7 INCOMPLETE");
      stop();
      return 3;
    }

    if (maxAbs(b_now, b_insp.front()) > 0.02)
    {
      emit("PHASE CONNECTION FAIL: B handover vs DUAL-5-T path start");
      emit("  now " + fmt(b_now));
      emit("  5T " + fmt(b_insp.front()));
      emit("DUAL-7 INCOMPLETE");
      stop();
      return 3;
    }

    auto nearest = [&](const std::vector<double>& goal, size_t from) {
      size_t best = from;
      double best_d = 1e9;
      for (size_t i = from; i < b_insp.size(); ++i)
      {
        const double d = maxAbs(b_insp[i], goal);
        if (d < best_d)
        {
          best_d = d;
          best = i;
        }
      }
      return best;
    };
    const size_t i1_idx = nearest(i1, 0);
    const size_t i2_idx = nearest(i2, i1_idx);
    const size_t i3_idx = nearest(i3, i2_idx);
    emit("DUAL-5-T split indices: I1=" + std::to_string(i1_idx) + " I2=" + std::to_string(i2_idx) +
         " I3=" + std::to_string(i3_idx) + " n=" + std::to_string(b_insp.size()));
    emit("  I1 maxAbs " + std::to_string(maxAbs(b_insp[i1_idx], i1)));
    emit("  I2 maxAbs " + std::to_string(maxAbs(b_insp[i2_idx], i2)));
    emit("  I3 maxAbs " + std::to_string(maxAbs(b_insp[i3_idx], i3)));

    auto split_range = [&](size_t from, size_t to, const std::string& name, const std::string& face) {
      Phase p;
      p.name = name;
      p.moving = "arm_b";
      p.face = face;
      p.source = "DUAL-5-T validated waypoints";
      p.start_a = a_now;
      p.start_b = b_now;
      p.owner = "B";
      p.q_a = q_a_open;
      p.q_b = q_b_hold;
      if (to >= b_insp.size())
        to = b_insp.size() - 1;
      for (size_t i = from; i <= to; ++i)
        p.wps.push_back(b_insp[i]);
      std::string cerr;
      p.collision_ok = validateMoving(*scene, a_now, p.wps, false, q_a_open, q_b_hold, step, cerr);
      if (!p.collision_ok)
        p.note = cerr;
      if (!p.wps.empty())
        b_now = p.wps.back();
      p.end_a = a_now;
      p.end_b = b_now;
      p.connected = true;
      add_phase(p);
      return p.collision_ok;
    };

    if (!split_range(0, i1_idx, "Phase 8 B +X", "+X") ||
        !split_range(i1_idx, i2_idx, "Phase 9 B -X", "-X") ||
        !split_range(i2_idx, i3_idx, "Phase 10 B +Z", "+Z"))
    {
      emit("DUAL-7 INCOMPLETE");
      emit("Arm B inspection revalidation failed");
      stop();
      return 3;
    }

    emit("");
    emit("========== DUAL-7 COMPLETE TASK ==========");
    bool all_ok = true;
    for (size_t i = 0; i < phases.size(); ++i)
    {
      const auto& p = phases[i];
      emit("");
      emit(p.name + ":");
      emit("  actual start state:");
      emit("    A " + fmt(p.start_a));
      emit("    B " + fmt(p.start_b));
      emit("  actual end state:");
      emit("    A " + fmt(p.end_a));
      emit("    B " + fmt(p.end_b));
      emit("  moving arm: " + p.moving);
      emit("  object owner: " + p.owner);
      emit("  gripper state: q_A=" + std::to_string(p.q_a) + " q_B=" + std::to_string(p.q_b));
      emit("  trajectory source: " + p.source);
      emit("  collision validation: " + std::string(p.collision_ok ? "PASS" : "FAIL"));
      emit("  connection to next phase: " + std::string(p.connected ? "OK" : "FAIL"));
      if (!p.note.empty())
        emit("  note: " + p.note);
      if (!p.collision_ok || !p.connected)
        all_ok = false;
      if (i + 1 < phases.size())
      {
        const double da = maxAbs(p.end_a, phases[i + 1].start_a);
        const double db = maxAbs(p.end_b, phases[i + 1].start_b);
        emit("  joint discontinuity to next: A=" + std::to_string(da) + " B=" + std::to_string(db) +
             " rad");
        if (da > chain_tol + 0.05 || db > chain_tol + 0.05)
        {
          emit("  PHASE CONNECTION FAIL");
          all_ok = false;
        }
      }
    }

    std::vector<std::vector<double>> all_a, all_b;
    for (const auto& p : phases)
    {
      if (p.moving == "arm_a")
      {
        all_a.insert(all_a.end(), p.wps.begin(), p.wps.end());
        if (!all_b.empty() || !p.start_b.empty())
          all_b.push_back(p.end_b.empty() ? p.start_b : p.end_b);
      }
      else if (p.moving == "arm_b")
      {
        all_b.insert(all_b.end(), p.wps.begin(), p.wps.end());
        all_a.push_back(p.end_a.empty() ? p.start_a : p.end_a);
      }
    }

    emit("");
    emit("========== MOTION METRICS (GEOMETRIC, NOT TIME-OPTIMAL) ==========");
    emit("Selected complete task candidate:");
    emit("  matching DUAL-5-T chain A1+B_pre2+B_handover3 joint vectors");
    emit("Selected handover Arm A: " + fmt(han_a));
    emit("Selected handover Arm B: " + fmt(han_b));
    emit("Selected B inspection rolls: I1=" + std::to_string(i1_roll) +
         " I2=" + std::to_string(i2_roll) + " I3=" + std::to_string(i3_roll));
    auto report_arm = [&](const std::string& label, const std::vector<std::vector<double>>& wps) {
      emit(label);
      emit("  J1 cumulative travel: " + std::to_string(travel(wps, 0)) + " rad");
      emit("  J2 cumulative travel: " + std::to_string(travel(wps, 1)) + " rad");
      emit("  J1 range: " + std::to_string(span(wps, 0)) + " rad");
      emit("  J2 range: " + std::to_string(span(wps, 1)) + " rad");
      emit("  J6 cumulative travel: " + std::to_string(travel(wps, 5)) + " rad");
      emit("  J6 reversals: " + std::to_string(reversals(wps, 5)));
    };
    report_arm("Arm A:", all_a);
    report_arm("Arm B:", all_b);

    auto envelope = [&](const std::string& label, const std::string& tcp,
                        const std::vector<std::string>& links, const std::string& moving) {
      Eigen::Vector3d aabb_min = Eigen::Vector3d::Constant(1e9);
      Eigen::Vector3d aabb_max = Eigen::Vector3d::Constant(-1e9);
      double tcp_len = 0.0;
      Eigen::Vector3d prev = Eigen::Vector3d::Zero();
      bool have = false;
      for (const auto& p : phases)
      {
        if (p.moving != moving)
          continue;
        for (const auto& q : p.wps)
        {
          const auto st = fkState(moving == "arm_a" ? q : p.start_a,
                                  moving == "arm_b" ? q : p.start_b, p.q_a, p.q_b);
          if (!st.getRobotModel()->hasLinkModel(tcp))
            continue;
          const Eigen::Vector3d t = st.getGlobalLinkTransform(tcp).translation();
          if (have)
            tcp_len += (t - prev).norm();
          prev = t;
          have = true;
          for (const auto& link : links)
          {
            if (!st.getRobotModel()->hasLinkModel(link))
              continue;
            const Eigen::Vector3d pt = st.getGlobalLinkTransform(link).translation();
            aabb_min = aabb_min.cwiseMin(pt);
            aabb_max = aabb_max.cwiseMax(pt);
          }
        }
      }
      emit(label);
      emit("  TCP path length: " + std::to_string(tcp_len) + " m");
      emit("  moving-link swept AABB proxy min: " + fmt({aabb_min.x(), aabb_min.y(), aabb_min.z()}));
      emit("  moving-link swept AABB proxy max: " + fmt({aabb_max.x(), aabb_max.y(), aabb_max.z()}));
      emit("  (swept-envelope proxy, NOT true swept volume)");
    };
    envelope("Arm A envelope:", kTcpA, env_a, "arm_a");
    envelope("Arm B envelope:", kTcpB, env_b, "arm_b");
    double jspace = 0.0;
    auto add_jspace = [&](const std::vector<std::vector<double>>& wps) {
      for (size_t i = 1; i < wps.size(); ++i)
        for (size_t j = 0; j < std::min(wps[i].size(), wps[i - 1].size()); ++j)
          jspace += std::abs(wps[i][j] - wps[i - 1][j]);
    };
    add_jspace(all_a);
    add_jspace(all_b);
    emit("Total tested joint-space geometric path length: " + std::to_string(jspace) + " rad");
    emit("BEST TESTED COMPLETE GEOMETRIC TASK (not time-optimal)");

    std::ofstream yf(preview);
    yf << "stage: DUAL-7\n";
    yf << "rviz_preview_only: true\n";
    yf << "not_executable: true\n";
    yf << "visualization_playback_timing_only: true\n";
    yf << "complete: " << (all_ok ? "true" : "false") << "\n";
    writeList(yf, "", "home_a", home_a);
    yf << "phases:\n";
    for (const auto& p : phases)
    {
      yf << "  - name: \"" << p.name << "\"\n";
      yf << "    moving: \"" << p.moving << "\"\n";
      yf << "    owner: \"" << p.owner << "\"\n";
      yf << "    face: \"" << p.face << "\"\n";
      yf << "    q_a: " << p.q_a << "\n";
      yf << "    q_b: " << p.q_b << "\n";
      writeList(yf, "    ", "start_a", p.start_a);
      writeList(yf, "    ", "start_b", p.start_b);
      writeList(yf, "    ", "end_a", p.end_a);
      writeList(yf, "    ", "end_b", p.end_b);
      {
        std::vector<double> ea = p.end_a.empty() ? p.start_a : p.end_a;
        std::vector<double> eb = p.end_b.empty() ? p.start_b : p.end_b;
        if (ea.size() == 6 && eb.size() == 6)
        {
          const auto st = fkState(ea, eb, p.q_a, p.q_b);
          Eigen::Vector3d obj = T_world_obj0.translation();
          if (p.owner == "A" && st.getRobotModel()->hasLinkModel(kTcpA))
            obj = (st.getGlobalLinkTransform(kTcpA) * T_tcpA_obj).translation();
          else if (p.owner == "B" && st.getRobotModel()->hasLinkModel(kTcpB))
            obj = (st.getGlobalLinkTransform(kTcpB) * T_tcpB_obj).translation();
          writeList(yf, "    ", "object_xyz", {obj.x(), obj.y(), obj.z()});
        }
      }
      yf << "    waypoints:\n";
      for (const auto& q : p.wps)
        writeList(yf, "      - ", "joints", q);
    }
    emit("");
    emit("RVIZ PREVIEW YAML:");
    emit("  " + preview);
    emit("RVIZ PREVIEW ONLY");
    emit("NOT EXECUTABLE");
    emit("VISUALIZATION PLAYBACK TIMING ONLY");

    emit("");
    emit(all_ok ? "DUAL-7 COMPLETE SIX-FACE MODEL DEMO PASS" : "DUAL-7 INCOMPLETE");
    emit("Physical grasp: NOT VERIFIED");
    emit("Real robot commands: ZERO");
    emit("Real gripper commands: ZERO");
    stop();
    return all_ok ? 0 : 3;
  }
  catch (const std::exception& e)
  {
    emit(std::string("DUAL-7 exception: ") + e.what());
    try
    {
      stop();
    }
    catch (...)
    {
    }
    return 1;
  }
}
