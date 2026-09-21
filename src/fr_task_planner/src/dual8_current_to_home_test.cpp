// DUAL-8-HOME: plan dual-arm Current→Home from live named joints.
// PLAN ONLY by default. Does not send FollowJointTrajectory or gripper commands.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;
using MoveGroup = moveit::planning_interface::MoveGroupInterface;

namespace
{
constexpr char kGroupA[] = "arm_a";
constexpr char kGroupB[] = "arm_b";
constexpr char kGroupDual[] = "dual_arms";
const std::vector<std::string> kJa = {"arm_a_j1", "arm_a_j2", "arm_a_j3",
                                      "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kJb = {"arm_b_j1", "arm_b_j2", "arm_b_j3",
                                      "arm_b_j4", "arm_b_j5", "arm_b_j6"};

void emit(const std::string& s)
{
  std::cout << s << std::endl;
}

std::string fmt(const std::vector<double>& v)
{
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o << std::setprecision(9) << "[";
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (i)
      o << ", ";
    o << v[i];
  }
  o << "]";
  return o.str();
}

bool yamlVec(const YAML::Node& n, std::vector<double>& out)
{
  if (!n || !n.IsSequence())
    return false;
  out.clear();
  for (const auto& v : n)
    out.push_back(v.as<double>());
  return out.size() == 6;
}

std::vector<double> named(const sensor_msgs::msg::JointState& msg,
                          const std::vector<std::string>& names, bool& ok)
{
  std::vector<double> q(names.size(), 0.0);
  ok = true;
  for (size_t i = 0; i < names.size(); ++i)
  {
    auto it = std::find(msg.name.begin(), msg.name.end(), names[i]);
    if (it == msg.name.end())
    {
      ok = false;
      continue;
    }
    const size_t idx = static_cast<size_t>(std::distance(msg.name.begin(), it));
    if (idx >= msg.position.size())
    {
      ok = false;
      continue;
    }
    q[i] = msg.position[idx];
  }
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

bool colliding(planning_scene::PlanningScene& scene, moveit::core::RobotState& st, std::string& pair)
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
  if (res.contacts.empty())
  {
    pair = "unspecified";
    return true;
  }
  const auto& c = *res.contacts.begin();
  pair = c.first.first + " <-> " + c.first.second;
  return true;
}

bool inLimits(const moveit::core::RobotModelConstPtr& model, const std::string& name, double q,
              std::string& err)
{
  const auto* jm = model->getJointModel(name);
  if (!jm)
  {
    err = "missing joint " + name;
    return false;
  }
  const auto& b = jm->getVariableBounds();
  if (b.empty())
    return true;
  if (b.front().position_bounded_ &&
      (q < b.front().min_position_ - 1e-9 || q > b.front().max_position_ + 1e-9))
  {
    std::ostringstream o;
    o << name << " q=" << q << " outside [" << b.front().min_position_ << ", "
      << b.front().max_position_ << "]";
    err = o.str();
    return false;
  }
  return true;
}

bool allLimits(const moveit::core::RobotModelConstPtr& model, const std::vector<std::string>& names,
               const std::vector<double>& q, std::string& err)
{
  for (size_t i = 0; i < names.size() && i < q.size(); ++i)
  {
    if (!inLimits(model, names[i], q[i], err))
      return false;
  }
  return true;
}

bool interpolateValid(planning_scene::PlanningScene& scene, moveit::core::RobotState start,
                      const std::vector<std::string>& moving_names,
                      const std::vector<std::vector<double>>& wps, double step, std::string& err)
{
  if (wps.size() < 2)
    return true;
  for (size_t i = 1; i < wps.size(); ++i)
  {
    double jump = 0.0;
    for (size_t j = 0; j < std::min(wps[i].size(), wps[i - 1].size()); ++j)
      jump = std::max(jump, std::abs(wps[i][j] - wps[i - 1][j]));
    const int n = std::max(1, static_cast<int>(std::ceil(jump / std::max(step, 1e-6))));
    for (int k = 0; k <= n; ++k)
    {
      const double s = static_cast<double>(k) / static_cast<double>(n);
      for (size_t j = 0; j < moving_names.size() && j < wps[i].size(); ++j)
      {
        const double a = wps[i - 1][j];
        const double b = wps[i][j];
        start.setVariablePosition(moving_names[j], a + s * (b - a));
      }
      std::string pair;
      if (colliding(scene, start, pair))
      {
        err = "interpolated collision " + pair + " at waypoint " + std::to_string(i);
        return false;
      }
    }
  }
  return true;
}

std::vector<std::vector<double>> extractArm(const moveit_msgs::msg::RobotTrajectory& traj,
                                            const std::vector<std::string>& names)
{
  std::vector<std::vector<double>> path;
  for (const auto& pt : traj.joint_trajectory.points)
  {
    std::vector<double> q(names.size(), 0.0);
    for (size_t i = 0; i < names.size(); ++i)
    {
      auto it = std::find(traj.joint_trajectory.joint_names.begin(),
                          traj.joint_trajectory.joint_names.end(), names[i]);
      if (it == traj.joint_trajectory.joint_names.end())
        continue;
      const size_t idx = static_cast<size_t>(std::distance(traj.joint_trajectory.joint_names.begin(), it));
      if (idx < pt.positions.size())
        q[i] = pt.positions[idx];
    }
    path.push_back(q);
  }
  return path;
}

bool planNamed(MoveGroup& group, planning_scene::PlanningScene& /*scene*/,
               const moveit::core::RobotState& start, const std::map<std::string, double>& tgt,
               int attempts, double time, moveit_msgs::msg::RobotTrajectory& traj, std::string& err)
{
  group.setStartState(start);
  group.setPlanningTime(time);
  group.setNumPlanningAttempts(attempts);
  if (!group.setJointValueTarget(tgt))
  {
    err = "invalid joint target";
    return false;
  }
  MoveGroup::Plan plan;
  if (group.plan(plan) != moveit::core::MoveItErrorCode::SUCCESS)
  {
    err = "plan() failed";
    return false;
  }
  traj = plan.trajectory_;
  if (traj.joint_trajectory.points.empty())
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
  auto node = rclcpp::Node::make_shared("dual8_current_to_home_test", opt);

  emit("========== DUAL-8-HOME ==========");
  emit("PLAN ONLY. No FollowJointTrajectory. No gripper command.");
  emit("Does not auto-continue into DUAL-7 grasp.");

  const std::string cfg_path = node->get_parameter_or(
      "home_yaml",
      std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
          "/fr_task_ws/src/fr_task_planner/config/dual8_home.yaml");
  YAML::Node y = YAML::LoadFile(cfg_path);
  std::vector<double> home_a, home_b, home_b_deg, pre_b;
  if (!yamlVec(y["home_a"], home_a) || !yamlVec(y["home_b"], home_b) ||
      !yamlVec(y["home_b_deg"], home_b_deg) || !yamlVec(y["pre_b"], pre_b))
  {
    emit("FAIL: dual8_home.yaml missing home vectors");
    rclcpp::shutdown();
    return 1;
  }
  const double step = y["max_validation_joint_step"].as<double>(0.02);
  const int attempts = y["max_planning_attempts"].as<int>(5);
  const double ptime = y["planning_time_sec"].as<double>(10.0);
  const std::string preview = y["preview_yaml"].as<std::string>("/tmp/dual8_home_preview.yaml");
  const std::string conn = y["connection_yaml"].as<std::string>();
  const std::string report = y["report_yaml"].as<std::string>("/tmp/dual8_home_report.yaml");

  emit("Arm A Home rad " + fmt(home_a));
  emit("Arm B Home deg " + fmt(home_b_deg));
  emit("Arm B Home rad " + fmt(home_b));
  for (size_t i = 0; i < 6; ++i)
  {
    const double expect = home_b_deg[i] * M_PI / 180.0;
    if (std::abs(expect - home_b[i]) > 1e-9)
    {
      emit("FAIL: home_b rad does not match home_b_deg; not silently rewriting degree");
      rclcpp::shutdown();
      return 1;
    }
  }

  if (!overlayModel(node))
  {
    emit("FAIL: move_group robot_description unavailable");
    rclcpp::shutdown();
    return 4;
  }
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  std::thread spinner([&exec]() { exec.spin(); });

  auto cleanup = [&]() {
    exec.cancel();
    spinner.join();
  };

  robot_model_loader::RobotModelLoader loader(node);
  auto model = loader.getModel();
  if (!model || model->getName() != "fairino3_dual_robot")
  {
    emit("FAIL: unexpected robot model");
    cleanup();
    rclcpp::shutdown();
    return 1;
  }
  if (!model->hasJointModelGroup(kGroupDual) || !model->hasJointModelGroup(kGroupA) ||
      !model->hasJointModelGroup(kGroupB))
  {
    emit("FAIL: missing arm_a / arm_b / dual_arms");
    cleanup();
    rclcpp::shutdown();
    return 1;
  }
  const auto* dual_g = model->getJointModelGroup(kGroupDual);
  emit("dual_arms active joints: " + std::to_string(dual_g->getActiveJointModelNames().size()));

  std::string lim_err;
  bool limits_ok = allLimits(model, kJa, home_a, lim_err) && allLimits(model, kJb, home_b, lim_err);
  emit(std::string("Home joint limits: ") + (limits_ok ? "PASS" : "FAIL " + lim_err));
  if (!limits_ok)
  {
    emit("User-specified Arm B angles were NOT rewritten.");
    cleanup();
    rclcpp::shutdown();
    return 1;
  }

  auto scene = fetchScene(node, model);
  if (!scene)
  {
    emit("FAIL: GetPlanningScene");
    cleanup();
    rclcpp::shutdown();
    return 4;
  }
  std::vector<const moveit::core::AttachedBody*> attached;
  scene->getCurrentState().getAttachedBodies(attached);
  emit("Attached bodies: " +
       (attached.empty() ? std::string("(none)") : std::string(attached.front()->getName())));

  moveit::core::RobotState home_st(scene->getCurrentState());
  setJoints(home_st, kJa, home_a);
  setJoints(home_st, kJb, home_b);
  std::string home_pair;
  const bool home_col = colliding(*scene, home_st, home_pair);
  emit(std::string("A Home + B Home static collision: ") +
       (home_col ? "FAIL " + home_pair : "PASS"));
  if (home_col)
  {
    emit("User-specified Arm B Home was NOT rewritten.");
    cleanup();
    rclcpp::shutdown();
    return 1;
  }

  sensor_msgs::msg::JointState::SharedPtr js;
  std::mutex mu;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mu);
        js = msg;
      });
  const auto t0 = std::chrono::steady_clock::now();
  while (rclcpp::ok() && std::chrono::steady_clock::now() - t0 < 5s)
  {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (js)
        break;
    }
    std::this_thread::sleep_for(50ms);
  }
  sensor_msgs::msg::JointState live_msg;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!js)
    {
      emit("FAIL: no live /joint_states; cannot plan from real current");
      cleanup();
      rclcpp::shutdown();
      return 1;
    }
    live_msg = *js;
  }
  bool ok_a = false, ok_b = false;
  const auto live_a = named(live_msg, kJa, ok_a);
  const auto live_b = named(live_msg, kJb, ok_b);
  emit("live Arm A (by name) " + fmt(live_a) + (ok_a ? " names=ok" : " names=MISSING"));
  emit("live Arm B (by name) " + fmt(live_b) + (ok_b ? " names=ok" : " names=MISSING"));
  if (!ok_a || !ok_b)
  {
    emit("FAIL: live joint names missing arm_a_j* or arm_b_j*");
    cleanup();
    rclcpp::shutdown();
    return 1;
  }
  if (maxAbs(live_a, std::vector<double>(6, 0.0)) < 1e-3 &&
      maxAbs(live_b, std::vector<double>(6, 0.0)) < 1e-3)
  {
    emit("FAIL: live joints look like mock zeros; not a real Current→Home start");
    cleanup();
    rclcpp::shutdown();
    return 1;
  }
  std::string live_lim;
  if (!allLimits(model, kJa, live_a, live_lim) || !allLimits(model, kJb, live_b, live_lim))
  {
    emit("FAIL live limits " + live_lim);
    cleanup();
    rclcpp::shutdown();
    return 1;
  }
  moveit::core::RobotState live_st(scene->getCurrentState());
  setJoints(live_st, kJa, live_a);
  setJoints(live_st, kJb, live_b);
  std::string live_pair;
  const bool live_col = colliding(*scene, live_st, live_pair);
  emit(std::string("live dual-arm collision: ") + (live_col ? "FAIL " + live_pair : "PASS"));
  if (live_col)
  {
    emit("REFUSING to plan Current→Home from a colliding start");
    cleanup();
    rclcpp::shutdown();
    return 1;
  }

  MoveGroup dual(node, kGroupDual);
  MoveGroup arm_a(node, kGroupA);
  MoveGroup arm_b(node, kGroupB);
  dual.setMaxVelocityScalingFactor(0.2);
  arm_a.setMaxVelocityScalingFactor(0.2);
  arm_b.setMaxVelocityScalingFactor(0.2);

  std::map<std::string, double> tgt_dual, tgt_a, tgt_b;
  for (size_t i = 0; i < 6; ++i)
  {
    tgt_dual[kJa[i]] = home_a[i];
    tgt_dual[kJb[i]] = home_b[i];
    tgt_a[kJa[i]] = home_a[i];
    tgt_b[kJb[i]] = home_b[i];
  }

  moveit_msgs::msg::RobotTrajectory dual_traj;
  std::string dual_err;
  const bool dual_plan =
      planNamed(dual, *scene, live_st, tgt_dual, attempts, ptime, dual_traj, dual_err);
  bool dual_valid = false;
  std::vector<std::vector<double>> dual_a, dual_b;
  if (dual_plan)
  {
    dual_a = extractArm(dual_traj, kJa);
    dual_b = extractArm(dual_traj, kJb);
    std::string ierr;
    std::vector<std::string> dual_names = kJa;
    dual_names.insert(dual_names.end(), kJb.begin(), kJb.end());
    std::vector<std::vector<double>> dual_wps;
    const size_t nwp = std::min(dual_a.size(), dual_b.size());
    dual_wps.reserve(nwp);
    for (size_t i = 0; i < nwp; ++i)
    {
      std::vector<double> q = dual_a[i];
      q.insert(q.end(), dual_b[i].begin(), dual_b[i].end());
      dual_wps.push_back(q);
    }
    dual_valid = interpolateValid(*scene, live_st, dual_names, dual_wps, step, ierr);
    if (!dual_valid)
      emit("12-DOF plan interpolation: FAIL " + ierr);
  }
  emit(std::string("12-DOF dual_arms plan: ") + (dual_plan && dual_valid ? "PASS" : "FAIL " + dual_err));
  emit("SIMULTANEOUS HOME EXECUTION: NOT GUARANTEED (two independent FollowJointTrajectory "
       "controllers; common stamp is best-effort only)");

  auto seqPlan = [&](bool a_first, std::vector<std::vector<double>>& first_wps,
                     std::vector<std::vector<double>>& second_wps, std::string& err) {
    moveit::core::RobotState st = live_st;
    std::map<std::string, double> first = a_first ? tgt_a : tgt_b;
    MoveGroup& g1 = a_first ? arm_a : arm_b;
    moveit_msgs::msg::RobotTrajectory t1;
    if (!planNamed(g1, *scene, st, first, attempts, ptime, t1, err))
      return false;
    first_wps = extractArm(t1, a_first ? kJa : kJb);
    if (!interpolateValid(*scene, st, a_first ? kJa : kJb, first_wps, step, err))
      return false;
    setJoints(st, a_first ? kJa : kJb, a_first ? home_a : home_b);
    std::map<std::string, double> second = a_first ? tgt_b : tgt_a;
    MoveGroup& g2 = a_first ? arm_b : arm_a;
    moveit_msgs::msg::RobotTrajectory t2;
    if (!planNamed(g2, *scene, st, second, attempts, ptime, t2, err))
      return false;
    second_wps = extractArm(t2, a_first ? kJb : kJa);
    return interpolateValid(*scene, st, a_first ? kJb : kJa, second_wps, step, err);
  };

  std::vector<std::vector<double>> ab_a, ab_b, ba_b, ba_a;
  std::string seq_err;
  const bool seq_ab = seqPlan(true, ab_a, ab_b, seq_err);
  emit(std::string("SEQUENTIAL A then B: ") + (seq_ab ? "PASS" : "FAIL " + seq_err));
  const bool seq_ba = seqPlan(false, ba_b, ba_a, seq_err);
  emit(std::string("SEQUENTIAL B then A: ") + (seq_ba ? "PASS" : "FAIL " + seq_err));
  emit(std::string("SIMULTANEOUS HOME: ") +
       (dual_plan && dual_valid ? "PLAN PASS / EXECUTION NOT HARDWARE-SYNCED" : "NOT AVAILABLE"));
  emit(std::string("SEQUENTIAL HOME: ") + (seq_ab || seq_ba ? "PASS" : "FAIL"));

  moveit::core::RobotState at_home = home_st;
  std::map<std::string, double> tgt_pre;
  for (size_t i = 0; i < 6; ++i)
    tgt_pre[kJb[i]] = pre_b[i];
  moveit_msgs::msg::RobotTrajectory pre_traj;
  std::string pre_err;
  const bool pre_plan =
      planNamed(arm_b, *scene, at_home, tgt_pre, attempts, ptime, pre_traj, pre_err);
  bool pre_ok = false;
  std::vector<std::vector<double>> pre_wps;
  if (pre_plan)
  {
    pre_wps = extractArm(pre_traj, kJb);
    pre_ok = interpolateValid(*scene, at_home, kJb, pre_wps, step, pre_err);
    if (pre_wps.empty() || maxAbs(pre_wps.front(), home_b) > 0.05)
    {
      pre_ok = false;
      pre_err = "B Home→Pre start mismatch";
    }
  }
  emit(std::string("Arm B new Home → Pre-handover: ") + (pre_ok ? "PASS" : "FAIL " + pre_err));

  if (pre_ok && !conn.empty())
  {
    std::ofstream cf(conn);
    cf << "task_version: DUAL-8-HOME\n";
    cf << "id: b_home_to_pre_handover\n";
    cf << "replaces: dual7 b_to_pre_handover from model-zero\n";
    cf << "not_cartesian_linear: true\n";
    cf << "joint_names: [arm_b_j1, arm_b_j2, arm_b_j3, arm_b_j4, arm_b_j5, arm_b_j6]\n";
    writeList(cf, "", "start_joints", home_b);
    writeList(cf, "", "end_joints", pre_b);
    cf << "points:\n";
    for (const auto& q : pre_wps)
    {
      cf << "  - positions: [";
      for (size_t i = 0; i < q.size(); ++i)
      {
        if (i)
          cf << ", ";
        cf << std::setprecision(12) << q[i];
      }
      cf << "]\n";
    }
    emit("wrote " + conn);
  }

  std::ofstream pf(preview);
  pf << "stage: DUAL-8-HOME\n";
  pf << "rviz_preview_only: true\n";
  pf << "not_executable: true\n";
  pf << "complete: " << ((seq_ab || seq_ba || (dual_plan && dual_valid)) && pre_ok && !home_col ? "true" : "false")
     << "\n";
  writeList(pf, "", "home_a", home_a);
  writeList(pf, "", "home_b", home_b);
  pf << "phases:\n";
  auto dumpPhase = [&](const std::string& name, const std::string& moving,
                       const std::vector<double>& sa, const std::vector<double>& sb,
                       const std::vector<double>& ea, const std::vector<double>& eb,
                       const std::vector<std::vector<double>>& wps) {
    pf << "  - name: \"" << name << "\"\n";
    pf << "    moving: \"" << moving << "\"\n";
    pf << "    owner: \"world\"\n";
    pf << "    face: \"(home)\"\n";
    pf << "    q_a: 0.0\n";
    pf << "    q_b: 0.0\n";
    writeList(pf, "    ", "start_a", sa);
    writeList(pf, "    ", "start_b", sb);
    writeList(pf, "    ", "end_a", ea);
    writeList(pf, "    ", "end_b", eb);
    pf << "    waypoints:\n";
    for (const auto& q : wps)
    {
      pf << "      - [";
      for (size_t i = 0; i < q.size(); ++i)
      {
        if (i)
          pf << ", ";
        pf << std::setprecision(12) << q[i];
      }
      pf << "]\n";
    }
  };
  if (seq_ab)
  {
    dumpPhase("Current A to Home", "arm_a", live_a, live_b, home_a, live_b, ab_a);
    dumpPhase("Current B to Home", "arm_b", home_a, live_b, home_a, home_b, ab_b);
  }
  else if (seq_ba)
  {
    dumpPhase("Current B to Home", "arm_b", live_a, live_b, live_a, home_b, ba_b);
    dumpPhase("Current A to Home", "arm_a", live_a, home_b, home_a, home_b, ba_a);
  }
  if (pre_ok)
    dumpPhase("B Home to Pre-handover", "arm_b", home_a, home_b, home_a, pre_b, pre_wps);
  emit("wrote " + preview);

  auto pub = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/dual8/preview/display_planned_path", 1);
  if (dual_plan)
  {
    moveit_msgs::msg::DisplayTrajectory disp;
    disp.model_id = model->getName();
    moveit_msgs::msg::RobotState rs;
    moveit::core::robotStateToRobotStateMsg(live_st, rs);
    disp.trajectory_start = rs;
    disp.trajectory.push_back(dual_traj);
    pub->publish(disp);
    emit("published /dual8/preview/display_planned_path");
  }

  std::ofstream rf(report);
  rf << "task_version: DUAL-8-HOME\n";
  rf << "limits_ok: " << (limits_ok ? "true" : "false") << "\n";
  rf << "home_static_collision: " << (home_col ? "true" : "false") << "\n";
  rf << "dual_arms_plan: " << (dual_plan && dual_valid ? "true" : "false") << "\n";
  rf << "simultaneous_execution_guaranteed: false\n";
  rf << "sequential_ab: " << (seq_ab ? "true" : "false") << "\n";
  rf << "sequential_ba: " << (seq_ba ? "true" : "false") << "\n";
  rf << "b_home_to_pre: " << (pre_ok ? "true" : "false") << "\n";
  emit("wrote " + report);

  const int rc = (limits_ok && !home_col && (seq_ab || seq_ba || (dual_plan && dual_valid)) && pre_ok)
                     ? 0
                     : 3;
  emit(std::string("DUAL-8-HOME: ") + (rc == 0 ? "MODEL/PLAN PASS" : "FAIL"));
  cleanup();
  rclcpp::shutdown();
  return rc;
}
