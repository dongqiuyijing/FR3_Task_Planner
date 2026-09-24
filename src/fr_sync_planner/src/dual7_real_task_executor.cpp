// DUAL-7 dual-arm real executor. Reuses STEP13 frozen-executor helpers.
// Default DRY RUN. Does not rewrite the single-arm executor.
#include "fr_task_planner/frozen_executor.hpp"
#include "fr_task_planner/keypose_v1_gates.hpp"
#include "fr_task_planner/winner_trajectory_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <fairino_msgs/srv/gripper_bridge.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <yaml-cpp/yaml.h>

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using MoveGroup = moveit_msgs::action::MoveGroup;
using fr_task_planner::ExecutorConfig;
using fr_task_planner::GripperBridgeRequestFields;
using fr_task_planner::GripperCompletionKind;
using fr_task_planner::JointSnapshot;
using fr_task_planner::TrajectorySegmentRecord;
using fr_task_planner::actionTimeoutSec;
using fr_task_planner::classifyGripperCall;
using fr_task_planner::classifyGripperCompletion;
using fr_task_planner::formatGripperCompletionFailure;
using fr_task_planner::formatGripperRequestLog;
using fr_task_planner::formatGripperResponseLog;
using fr_task_planner::gripperBridgeMotionDone;
using fr_task_planner::gripperCompletionName;
using fr_task_planner::kRealRobotConfirmation;
using fr_task_planner::makeGripperCloseRequest;
using fr_task_planner::makeGripperOpenRequest;
using fr_task_planner::makeGripperPingRequest;
using fr_task_planner::motionAllowed;
using fr_task_planner::readTrajectoryYaml;
using fr_task_planner::remapSegmentByJointName;
using fr_task_planner::scaleSegment;
using fr_task_planner::segmentDuration;
using fr_task_planner::setTimeFromStart;
using fr_task_planner::timeFromStartSec;

namespace
{
const std::vector<std::string> kJa = {"arm_a_j1", "arm_a_j2", "arm_a_j3",
                                      "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kJb = {"arm_b_j1", "arm_b_j2", "arm_b_j3",
                                      "arm_b_j4", "arm_b_j5", "arm_b_j6"};

std::string getString(const rclcpp::Node::SharedPtr& n, const std::string& k, const std::string& d)
{
  return n->has_parameter(k) ? n->get_parameter(k).as_string() : d;
}
double getDouble(const rclcpp::Node::SharedPtr& n, const std::string& k, double d)
{
  return n->has_parameter(k) ? n->get_parameter(k).as_double() : d;
}
bool getBool(const rclcpp::Node::SharedPtr& n, const std::string& k, bool d)
{
  return n->has_parameter(k) ? n->get_parameter(k).as_bool() : d;
}
int getInt(const rclcpp::Node::SharedPtr& n, const std::string& k, int d)
{
  return n->has_parameter(k) ? static_cast<int>(n->get_parameter(k).as_int()) : d;
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

bool nearZero(const std::vector<double>& q, double tol = 1e-3)
{
  if (q.size() < 6)
    return false;
  for (size_t i = 0; i < 6; ++i)
    if (std::abs(q[i]) > tol)
      return false;
  return true;
}

std::vector<double> named(const JointSnapshot& snap, const std::vector<std::string>& names)
{
  std::vector<double> q(names.size(), 0.0);
  for (size_t i = 0; i < names.size(); ++i)
  {
    const auto it = snap.joints.find(names[i]);
    if (it != snap.joints.end())
      q[i] = it->second;
  }
  return q;
}

bool hasNames(const JointSnapshot& snap, const std::vector<std::string>& names)
{
  for (const auto& n : names)
  {
    if (snap.joints.find(n) == snap.joints.end())
      return false;
  }
  return true;
}

struct DualCheck
{
  bool ok = false;
  std::string error;
  double max_error = 0.0;
};

DualCheck checkVec(const std::vector<double>& actual, const std::vector<double>& expected,
                   const std::vector<std::string>& names, double tol, const std::string& code)
{
  DualCheck r;
  if (actual.size() != expected.size() || expected.size() < 6)
  {
    r.error = "joint name/value size mismatch";
    return r;
  }
  double m = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < expected.size(); ++i)
  {
    if (!std::isfinite(actual[i]))
    {
      r.error = std::string("MISSING_JOINT joint=") + (i < names.size() ? names[i] : "?");
      return r;
    }
    const double e = std::abs(actual[i] - expected[i]);
    if (e >= m)
    {
      m = e;
      worst = i;
    }
  }
  r.max_error = m;
  if (m > tol)
  {
    std::ostringstream o;
    o << code << " joint=" << (worst < names.size() ? names[worst] : "?")
      << " predicted=" << actual[worst] << " stage_start=" << expected[worst] << " err=" << m;
    r.error = o.str();
    return r;
  }
  r.ok = true;
  return r;
}

DualCheck checkDual(const JointSnapshot& snap, const std::vector<std::string>& names,
                    const std::vector<double>& expected, double tol, const std::string& code)
{
  if (!hasNames(snap, names))
  {
    DualCheck r;
    r.error = "MISSING_JOINT";
    return r;
  }
  return checkVec(named(snap, names), expected, names, tol, code);
}

GripperBridgeRequestFields makeActivateRequest(const ExecutorConfig& cfg)
{
  GripperBridgeRequestFields req = makeGripperCloseRequest(cfg);
  req.command = "activate";
  req.position = 0;
  return req;
}

void prefixJ(TrajectorySegmentRecord& seg, const std::string& arm)
{
  const std::string p = (arm == "arm_b") ? "arm_b_" : "arm_a_";
  if (seg.joint_names.size() == 6 && (seg.joint_names[0] == "j1" || seg.joint_names[0] == "arm_a_j1" ||
                                      seg.joint_names[0] == "arm_b_j1"))
  {
    if (seg.joint_names[0] == "j1")
    {
      for (auto& n : seg.joint_names)
        n = p + n;
    }
  }
  else if (seg.joint_names.empty())
    seg.joint_names = (arm == "arm_b") ? kJb : kJa;
}

void assignConservativeTimes(TrajectorySegmentRecord& seg, double vmax)
{
  if (seg.points.empty())
    return;
  bool has_time = false;
  for (const auto& pt : seg.points)
    if (timeFromStartSec(pt) > 1e-9)
      has_time = true;
  if (has_time)
    return;
  double t = 0.0;
  setTimeFromStart(seg.points.front(), 0.0);
  for (size_t i = 1; i < seg.points.size(); ++i)
  {
    double jump = 0.0;
    const auto& a = seg.points[i - 1].positions;
    const auto& b = seg.points[i].positions;
    for (size_t j = 0; j < std::min(a.size(), b.size()); ++j)
      jump = std::max(jump, std::abs(b[j] - a[j]));
    t += std::max(jump / std::max(vmax, 1e-3), 0.05);
    setTimeFromStart(seg.points[i], t);
  }
  seg.duration = segmentDuration(seg);
}

trajectory_msgs::msg::JointTrajectory toTraj(const TrajectorySegmentRecord& seg)
{
  trajectory_msgs::msg::JointTrajectory msg;
  msg.header.frame_id = "world";
  msg.joint_names = seg.joint_names;
  for (const auto& pt : seg.points)
  {
    trajectory_msgs::msg::JointTrajectoryPoint out;
    out.positions = pt.positions;
    out.velocities = pt.velocities;
    out.accelerations = pt.accelerations;
    out.time_from_start.sec = pt.sec;
    out.time_from_start.nanosec = pt.nanosec;
    msg.points.push_back(out);
  }
  return msg;
}

struct Stage
{
  std::string id;
  std::string kind;
  std::string moving;
  std::string logical;
  std::string source;
  bool not_cartesian = false;
  bool requires_approach_confirm = false;
  bool requires_b_grasp_confirm = false;
  std::vector<double> object_center_world;
  double object_bottom_z = 0.0;
  TrajectorySegmentRecord seg;
  // Populated only by an isolated sync-handover candidate.  The legacy
  // single-arm stages continue to use seg unchanged.
  TrajectorySegmentRecord seg_a;
  TrajectorySegmentRecord seg_b;
};

class Dual7RealExecutor
{
public:
  explicit Dual7RealExecutor(rclcpp::Node::SharedPtr node) : node_(std::move(node)) {}

  int run()
  {
    load();
    std::string gate;
    const bool motion = motionAllowed(cfg_, gate);
    RCLCPP_INFO(node_->get_logger(), "========== DUAL-7 REAL TASK EXECUTOR ==========");
    RCLCPP_INFO(node_->get_logger(), "MODE: %s", motion ? "EXECUTE" : "DRY_RUN");
    RCLCPP_INFO(node_->get_logger(), "REAL ROBOT MOTION ENABLED: %s", motion ? "YES" : "NO");
    RCLCPP_INFO(node_->get_logger(), "auto_continue: %s", auto_continue_ ? "true" : "false");
    RCLCPP_INFO(node_->get_logger(), "segment: %s", segment_.empty() ? "(none)" : segment_.c_str());
    RCLCPP_INFO(node_->get_logger(), "resume_from: %s",
                resume_from_.empty() ? "(none)" : resume_from_.c_str());
    RCLCPP_INFO(node_->get_logger(), "task_mode: %s", task_mode_.c_str());
    RCLCPP_INFO(node_->get_logger(), "AUTO_HANDOVER_RELEASE: %s",
                auto_handover_release_ ? "ENABLED" : "DISABLED");
    RCLCPP_INFO(node_->get_logger(), "HANDOVER_WAIT: %.3f sec (used only after B CLOSE SUCCESS)",
                handover_release_delay_sec_);
    RCLCPP_INFO(node_->get_logger(), "speed scale: %.6f (STEP13 scaler, NOT RViz playback)",
                cfg_.trajectory_speed_scale);
    RCLCPP_INFO(node_->get_logger(), "execute confirmation token: %s", kRealRobotConfirmation);
    RCLCPP_INFO(node_->get_logger(), "Arm A action: %s", action_a_.c_str());
    RCLCPP_INFO(node_->get_logger(), "Arm B action: %s", action_b_.c_str());
    RCLCPP_INFO(node_->get_logger(), "Arm A gripper: %s", grip_a_.c_str());
    RCLCPP_INFO(node_->get_logger(), "Arm B gripper: %s", grip_b_.c_str());
    RCLCPP_INFO(node_->get_logger(),
                "timeout clocks: executor gripper_timeout_sec=%.3f hardware max_time_ms=%d "
                "hardware_service_wait_est_sec=%.3f (max_time+servo_restart_margin)",
                cfg_.gripper_timeout_sec, cfg_.gripper_max_time_ms, hw_wait_sec_);
    RCLCPP_INFO(node_->get_logger(),
                "activate is explicit ActGripper(id,1); reset is NEVER sent by this executor");
    RCLCPP_INFO(node_->get_logger(),
                "service-available / ping / activate / MoveGripper / motion-done / "
                "ServoJ-restore / physical-grasp are DISTINCT states");
    RCLCPP_INFO(node_->get_logger(), "motion gate: %s", gate.c_str());
    RCLCPP_INFO(node_->get_logger(),
                "OMPL handover approach is NOT a verified Cartesian linear path");
    RCLCPP_INFO(node_->get_logger(),
                "Physical B grasp is NOT implied by model Attachment");
    RCLCPP_INFO(node_->get_logger(),
                "DUAL-8 Home: sequential A then B; simultaneous execute is NOT hardware-synced");
    RCLCPP_INFO(node_->get_logger(),
                "continue_after_home=%s (Home-only mode never auto-starts gripper/grasp; "
                "full_task continues after real Home arrival)",
                continue_after_home_ ? "true" : "false");
    RCLCPP_INFO(node_->get_logger(),
                "segment:=ID runs ONE stage; resume_from:=ID continues FROM that stage");

    if (step15_path_.find("dual7_preview.yaml") != std::string::npos ||
        segs_path_.find("dual7_preview.yaml") != std::string::npos)
    {
      RCLCPP_ERROR(node_->get_logger(), "refusing /tmp/dual7_preview.yaml as executable trajectory");
      logOutcome(false, false, false, false, "FORBIDDEN_PREVIEW_YAML");
      return 1;
    }
    if (!segment_.empty() && !resume_from_.empty())
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "segment and resume_from cannot be set together; segment=one stage, "
                   "resume_from=from that stage onward");
      logOutcome(false, false, false, false, "PARAM_CONFLICT");
      return 1;
    }
    if (inject_failure_ == "YAML_LOAD")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=YAML_LOAD");
      logOutcome(false, false, false, false, "YAML_LOAD");
      return 1;
    }
    if (inject_failure_ == "UNSAFE_RESUME")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=UNSAFE_RESUME: refusing to skip safety");
      logOutcome(false, false, false, false, "UNSAFE_RESUME");
      return 1;
    }

    std::string err;
    if (task_mode_ == "keypose_v1")
    {
      if (!buildKeyposeStages(err))
      {
        RCLCPP_ERROR(node_->get_logger(), "keypose_v1 load failed: %s", err.c_str());
        logOutcome(false, false, false, false, err);
        return 1;
      }
    }
    else if (!buildStages(err))
    {
      RCLCPP_ERROR(node_->get_logger(), "task load failed: %s", err.c_str());
      logOutcome(false, false, false, false, err);
      return 1;
    }
    file_load_ok_ = true;
    RCLCPP_INFO(node_->get_logger(), "FILE_LOAD_PASS");

    joint_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) { onJs(msg); });
    client_a_ = rclcpp_action::create_client<FollowJointTrajectory>(node_, action_a_);
    client_b_ = rclcpp_action::create_client<FollowJointTrajectory>(node_, action_b_);
    grip_client_a_ = node_->create_client<fairino_msgs::srv::GripperBridge>(grip_a_);
    grip_client_b_ = node_->create_client<fairino_msgs::srv::GripperBridge>(grip_b_);
    move_client_ = rclcpp_action::create_client<MoveGroup>(node_, "move_action");
    confirm_srv_ = node_->create_service<std_srvs::srv::SetBool>(
        "confirm_b_grasp",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
          if (req->data)
          {
            confirm_vote_.store(1);
            res->success = true;
            res->message = "B_GRASP_CONFIRMED_BY_OPERATOR";
          }
          else
          {
            confirm_vote_.store(-1);
            res->success = false;
            res->message = "B_GRASP_REJECTED_BY_OPERATOR";
          }
        });

    waitLive(motion ? cfg_.startup_ready_timeout_sec : 3.0);
    inspect();
    if (inject_failure_ == "JOINT_STATES")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=JOINT_STATES");
      logOutcome(true, false, false, false, "JOINT_STATES");
      return 1;
    }

    if (motion && mock_zero_)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "MOCK_ZERO_NOT_REAL_START: live Arm A/B joints are ~0; refusing execute");
      logOutcome(true, false, false, false, "MOCK_ZERO");
      return 1;
    }
    if (motion)
    {
      auto s = snap();
      if (!s)
      {
        RCLCPP_ERROR(node_->get_logger(), "no live /joint_states; refusing execute");
        logOutcome(true, false, false, false, "NO_JOINT_STATES");
        return 1;
      }
      if (!hasNames(*s, kJa) || !hasNames(*s, kJb))
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "live /joint_states missing arm_a_j1..j6 or arm_b_j1..j6; refusing execute");
        logOutcome(true, false, false, false, "MISSING_JOINT_NAMES");
        return 1;
      }
      if (s->age_sec > max_joint_age_sec_)
      {
        RCLCPP_ERROR(node_->get_logger(), "joint_states stale age=%.3f s > %.3f", s->age_sec,
                     max_joint_age_sec_);
        logOutcome(true, false, false, false, "STALE_JOINT_STATES");
        return 1;
      }
    }
    if (motion && cfg_.gripper_timeout_sec + 1e-6 < hw_wait_sec_)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "executor gripper_timeout_sec=%.3f is shorter than hardware wait %.3f s; "
                   "refusing execute so a ServoJ restart is not mistaken for a finished move",
                   cfg_.gripper_timeout_sec, hw_wait_sec_);
      logOutcome(true, false, false, false, "GRIPPER_TIMEOUT_CLOCK");
      return 1;
    }
    if (motion && use_sim_time_)
    {
      RCLCPP_ERROR(node_->get_logger(), "REAL_EXECUTION_SIM_TIME_INVALID");
      logOutcome(true, false, false, false, "SIM_TIME");
      return 1;
    }
    if (motion && !auto_continue_ && segment_.empty() && resume_from_.empty())
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "default does not auto-run the full task; pass segment:=ID, resume_from:=ID, "
                   "task_mode:=full_task, or auto_continue:=true");
      printIndex();
      logOutcome(true, false, false, false, "NO_SCHEDULE");
      return 1;
    }

    initPredFromLive();
    if (!resume_from_.empty())
    {
      std::string rerr;
      if (!reconstructPredUntil(resume_from_, rerr))
      {
        RCLCPP_ERROR(node_->get_logger(), "resume reconstruct failed: %s", rerr.c_str());
        logOutcome(true, false, false, false, rerr);
        return 1;
      }
      if (motion && !resumeLiveMatchesPred())
      {
        logOutcome(true, false, false, false, "RESUME_LIVE_MISMATCH");
        return 1;
      }
    }

    int rc = 0;
    bool ran_any = false;
    bool seen_resume = resume_from_.empty();
    std::string fail_stage;
    for (auto& st : stages_)
    {
      if (!segment_.empty() && st.id != segment_)
        continue;
      if (!resume_from_.empty())
      {
        if (st.id == resume_from_)
          seen_resume = true;
        if (!seen_resume)
          continue;
      }
      ran_any = true;
      last_stage_id_ = st.id;
      if (inject_failure_ == "START_MISMATCH" &&
          (st.kind == "frozen_step15" || st.kind == "ompl_connection" ||
           st.kind == "dual5t_validated"))
      {
        RCLCPP_ERROR(node_->get_logger(), "inject_failure=START_MISMATCH at %s", st.id.c_str());
        fail_stage = st.id;
        rc = 1;
        break;
      }
      if (!runStage(st, motion))
      {
        fail_stage = st.id;
        rc = 1;
        break;
      }
      if (motion && st.id == "dual_current_to_home" && !continue_after_home_)
      {
        RCLCPP_INFO(node_->get_logger(),
                    "DUAL-8 Home finished; Home-only mode does not continue into gripper/grasp. "
                    "Use task_mode:=full_task for the complete six-face task.");
        break;
      }
      if (!auto_continue_ && motion && !segment_.empty())
        break;
    }
    if ((motion || !resume_from_.empty() || !segment_.empty()) && !ran_any)
    {
      RCLCPP_ERROR(node_->get_logger(), "unknown stage '%s'",
                   !segment_.empty() ? segment_.c_str() : resume_from_.c_str());
      printIndex();
      rc = 1;
      fail_stage = "unknown";
    }
    if (rc != 0)
    {
      RCLCPP_ERROR(node_->get_logger(), "STOP at stage=%s; not scheduling later stages; "
                                        "no automatic gripper reset/open",
                   fail_stage.c_str());
    }
    RCLCPP_INFO(node_->get_logger(), "commands sent to robot: %d", motion_sent_);
    RCLCPP_INFO(node_->get_logger(), "commands sent to gripper: %d", gripper_sent_);
    if (!motion && (motion_sent_ || gripper_sent_))
    {
      RCLCPP_ERROR(node_->get_logger(), "dry-run leaked a motion/gripper command");
      logOutcome(true, false, false, false, "DRY_RUN_LEAK");
      return 2;
    }
    const bool chain = (rc == 0) && file_load_ok_;
    const bool real_ok = motion && rc == 0 && fail_stage.empty() && segment_.empty() &&
                         resume_from_.empty() && task_mode_ == "full_task" && physical_b_grasp_ &&
                         continue_after_home_;
    if (chain && !motion)
      RCLCPP_INFO(node_->get_logger(), "TRAJECTORY_CHAIN_PASS");
    else if (!motion)
      RCLCPP_ERROR(node_->get_logger(), "TRAJECTORY_CHAIN_FAIL");
    RCLCPP_INFO(node_->get_logger(),
                "object_logic_owner=%s physical_B_grasp=%s (logic is NOT hardware proof)",
                owner_logic_.c_str(), physical_b_grasp_ ? "YES" : "NO");
    logOutcome(file_load_ok_, chain && !motion, home_plan_pass_, real_ok, fail_stage);
    return rc;
  }

private:
  void load()
  {
    cfg_.execute = getBool(node_, "execute", false);
    cfg_.real_robot_confirmation = getString(node_, "real_robot_confirmation", "");
    cfg_.plan_current_to_home = getBool(node_, "plan_current_to_home", false);
    cfg_.trajectory_speed_scale = getDouble(node_, "trajectory_speed_scale", 0.3);
    cfg_.home_tolerance_rad = getDouble(node_, "home_tolerance_rad", 0.02);
    cfg_.segment_start_tolerance_rad = getDouble(node_, "segment_start_tolerance_rad", 0.02);
    cfg_.segment_end_tolerance_rad = getDouble(node_, "segment_end_tolerance_rad", 0.02);
    cfg_.joint_settle_timeout_sec = getDouble(node_, "joint_settle_timeout_sec", 10.0);
    cfg_.joint_settle_poll_period_sec = getDouble(node_, "joint_settle_poll_period_sec", 0.10);
    cfg_.joint_settle_required_samples = getInt(node_, "joint_settle_required_samples", 3);
    cfg_.timeout_factor = getDouble(node_, "timeout_factor", 1.5);
    cfg_.timeout_margin_sec = getDouble(node_, "timeout_margin_sec", 10.0);
    cfg_.gripper_timeout_sec = getDouble(node_, "gripper_timeout_sec", 15.0);
    cfg_.gripper_post_close_wait_sec = getDouble(node_, "gripper_post_close_wait_sec", 0.5);
    cfg_.gripper_id = getInt(node_, "gripper_id", 1);
    cfg_.gripper_open_position = getInt(node_, "gripper_open_position", 0);
    cfg_.gripper_close_position = getInt(node_, "gripper_close_position", 85);
    cfg_.gripper_velocity = getInt(node_, "gripper_velocity", 80);
    cfg_.gripper_force = getInt(node_, "gripper_force", 20);
    cfg_.gripper_max_time_ms = getInt(node_, "gripper_max_time_ms", 5000);
    cfg_.gripper_block = getInt(node_, "gripper_block", 1);
    auto_continue_ = getBool(node_, "auto_continue", false);
    continue_after_home_ = getBool(node_, "continue_after_home", false);
    simultaneous_home_ = getBool(node_, "simultaneous_home", false);
    plan_home_ = getBool(node_, "plan_current_to_home", false);
    confirm_approach_ = getBool(node_, "confirm_handover_approach", false);
    confirm_grasp_b_ = getBool(node_, "handover_b_grasp_confirmed", false);
    auto_handover_release_ = getBool(node_, "auto_handover_release", false);
    handover_release_delay_sec_ = getDouble(node_, "handover_release_delay_sec", 2.0);
    if (handover_release_delay_sec_ < 0.0)
      handover_release_delay_sec_ = 0.0;
    grasp_post_close_wait_sec_ = getDouble(node_, "grasp_post_close_wait_sec", -1.0);
    handover_post_close_wait_sec_ = getDouble(node_, "handover_post_close_wait_sec", -1.0);
    place_post_open_wait_sec_ = getDouble(node_, "place_post_open_wait_sec", 1.5);
    if (handover_post_close_wait_sec_ >= 0.0)
      handover_release_delay_sec_ = handover_post_close_wait_sec_;
    already_act_a_ = getBool(node_, "gripper_a_already_activated", false);
    already_act_b_ = getBool(node_, "gripper_b_already_activated", false);
    hw_margin_ms_ = getInt(node_, "hardware_servo_restart_margin_ms", 4000);
    segment_ = getString(node_, "segment", "");
    resume_from_ = getString(node_, "resume_from", "");
    task_mode_ = getString(node_, "task_mode", "home_only");
    inject_failure_ = getString(node_, "inject_failure", "");
    empty_gripper_confirmation_ = getString(node_, "empty_gripper_confirmation", "");
    max_joint_age_sec_ = getDouble(node_, "max_joint_age_sec", 2.0);
    b_grasp_confirm_timeout_sec_ = getDouble(node_, "b_grasp_confirm_timeout_sec", 600.0);
    if (task_mode_ == "keypose_v1")
    {
      auto_continue_ = true;
      continue_after_home_ = true;
      plan_home_ = true;
      home_plan_velocity_scale_ = 1.0;
    }
    if (task_mode_ == "full_task")
    {
      auto_continue_ = true;
      continue_after_home_ = true;
      plan_home_ = true;
    }
    step15_path_ = getString(node_, "step15_trajectory", "");
    segs_path_ = getString(node_, "segments_file", "");
    home_yaml_ = getString(
        node_, "home_yaml",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/dual8_home.yaml");
    b_home_to_pre_file_ = getString(
        node_, "b_home_to_pre_file",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/dual8_b_home_to_pre_handover.yaml");
    action_a_ = getString(node_, "arm_a_trajectory_action_name",
                          "/arm_a_controller/follow_joint_trajectory");
    action_b_ = getString(node_, "arm_b_trajectory_action_name",
                          "/arm_b_controller/follow_joint_trajectory");
    grip_a_ = getString(node_, "arm_a_gripper_service_name", "/arm_a/fairino_gripper/command");
    grip_b_ = getString(node_, "arm_b_gripper_service_name", "/arm_b/fairino_gripper/command");
    group_a_ = getString(node_, "arm_a_planning_group", "arm_a");
    vmax_ = getDouble(node_, "conservative_joint_vmax_rad_s", 0.3);
    use_sim_time_ = getBool(node_, "use_sim_time", false);
    cfg_.planning_group = group_a_;
    const int motion_ms = std::max(cfg_.gripper_max_time_ms, 1000);
    hw_wait_sec_ = (static_cast<double>(motion_ms) + static_cast<double>(hw_margin_ms_)) / 1000.0;
    const std::string cfg_path = getString(
        node_, "config_file",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/dual7_real_executor.yaml");
    try
    {
      YAML::Node y = YAML::LoadFile(cfg_path);
      yamlVec(y["home_a"], home_a_);
      yamlVec(y["home_b"], home_b_);
      yamlVec(y["pre_b"], pre_b_);
      yamlVec(y["handover_a"], handover_a_);
      yamlVec(y["handover_b"], handover_b_);
    }
    catch (const std::exception& e)
    {
      RCLCPP_WARN(node_->get_logger(), "failed to load home/pre from %s: %s", cfg_path.c_str(),
                  e.what());
    }
    try
    {
      YAML::Node hy = YAML::LoadFile(home_yaml_);
      yamlVec(hy["home_a"], home_a_);
      yamlVec(hy["home_b"], home_b_);
      yamlVec(hy["home_b_deg"], home_b_deg_);
      yamlVec(hy["pre_b"], pre_b_);
      RCLCPP_INFO(node_->get_logger(), "loaded Dual-8 Home authority %s", home_yaml_.c_str());
    }
    catch (const std::exception& e)
    {
      RCLCPP_WARN(node_->get_logger(), "failed to load dual8_home.yaml %s: %s", home_yaml_.c_str(),
                  e.what());
    }
    if (home_b_.size() == 6 && home_b_deg_.size() == 6)
    {
      for (size_t i = 0; i < 6; ++i)
      {
        const double expect = home_b_deg_[i] * M_PI / 180.0;
        if (std::abs(expect - home_b_[i]) > 1e-8)
        {
          RCLCPP_ERROR(node_->get_logger(),
                       "home_b rad does not match home_b_deg; refusing to rewrite user degrees");
          home_b_.clear();
          break;
        }
      }
    }
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

  TrajectorySegmentRecord extraSeg(const std::string& id)
  {
    TrajectorySegmentRecord rec;
    rec.logical_segment = id;
    rec.name = id;
    try
    {
      YAML::Node y = YAML::LoadFile(segs_path_);
      for (const auto& s : y["segments"])
      {
        if (!s["id"] || s["id"].as<std::string>() != id)
          continue;
        rec.joint_names.clear();
        for (const auto& n : s["joint_names"])
          rec.joint_names.push_back(n.as<std::string>());
        rec.points.clear();
        for (const auto& p : s["points"])
        {
          fr_task_planner::TrajectoryPointRecord pt;
          yamlVec(p["positions"], pt.positions);
          rec.points.push_back(pt);
        }
        yamlVec(s["start_joints"], rec.start_joints);
        yamlVec(s["end_joints"], rec.end_joints);
        if (rec.start_joints.empty() && !rec.points.empty())
          rec.start_joints = rec.points.front().positions;
        if (rec.end_joints.empty() && !rec.points.empty())
          rec.end_joints = rec.points.back().positions;
        rec.deployable = true;
        return rec;
      }
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(node_->get_logger(), "segments_file load failed for %s: %s", id.c_str(),
                   e.what());
    }
    return rec;
  }

  TrajectorySegmentRecord loadBHomeToPre()
  {
    TrajectorySegmentRecord rec;
    rec.logical_segment = "B_Home_to_PreHandover";
    rec.name = "b_home_to_pre_handover";
    try
    {
      std::ifstream in(b_home_to_pre_file_);
      if (in.good())
      {
        YAML::Node y = YAML::LoadFile(b_home_to_pre_file_);
        rec.joint_names.clear();
        if (y["joint_names"])
        {
          for (const auto& n : y["joint_names"])
            rec.joint_names.push_back(n.as<std::string>());
        }
        else
          rec.joint_names = kJb;
        rec.points.clear();
        for (const auto& p : y["points"])
        {
          fr_task_planner::TrajectoryPointRecord pt;
          yamlVec(p["positions"], pt.positions);
          rec.points.push_back(pt);
        }
        yamlVec(y["start_joints"], rec.start_joints);
        yamlVec(y["end_joints"], rec.end_joints);
        if (rec.start_joints.empty() && !rec.points.empty())
          rec.start_joints = rec.points.front().positions;
        if (rec.end_joints.empty() && !rec.points.empty())
          rec.end_joints = rec.points.back().positions;
        rec.deployable = !rec.points.empty();
        if (rec.deployable)
        {
          RCLCPP_INFO(node_->get_logger(),
                      "using Dual-8 B Home→Pre %s points=%zu (not the model-zero DUAL-7 segment)",
                      b_home_to_pre_file_.c_str(), rec.points.size());
          return rec;
        }
      }
    }
    catch (const std::exception& e)
    {
      RCLCPP_WARN(node_->get_logger(), "dual8 B Home→Pre load failed: %s", e.what());
    }
    RCLCPP_WARN(node_->get_logger(),
                "Dual-8 B Home→Pre file missing or empty at %s; refusing model-zero fallback",
                b_home_to_pre_file_.c_str());
    rec.deployable = false;
    return rec;
  }

  bool buildKeyposeStages(std::string& err)
  {
    const std::string path = getString(
        node_, "keypose_v1_trajectory",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/keypose_v1_six_face_trajectory.yaml");
    YAML::Node y;
    try
    {
      y = YAML::LoadFile(path);
    }
    catch (const std::exception& e)
    {
      err = std::string("keypose trajectory: ") + e.what();
      return false;
    }
    if (y["keypose_dir"])
      keypose_dir_ = y["keypose_dir"].as<std::string>();
    const std::string deployed_config_dir = getString(node_, "keypose_v1_config_dir", "");
    if (!deployed_config_dir.empty())
    {
      keypose_dir_ = deployed_config_dir;
      RCLCPP_INFO(node_->get_logger(), "using deployment keypose config directory %s",
                  keypose_dir_.c_str());
    }
    yamlVec(y["home_a"], home_a_);
    yamlVec(y["home_b"], home_b_);
    home_b_deg_.clear();
    if (!y["stages"] || !y["stages"].IsSequence())
    {
      err = "keypose trajectory has no stages";
      return false;
    }
    Stage home;
    home.id = "dual_current_to_home";
    home.kind = "runtime_replan";
    home.moving = "dual";
    home.logical = "Dual_Current_to_Home";
    home.source = "live /joint_states → KEYPOSE homes; zero gripper; not stored in the six-face file";
    stages_.push_back(home);
    const double file_scale = y["speed_scale"] ? y["speed_scale"].as<double>() : 0.2;
    if (std::abs(cfg_.trajectory_speed_scale - file_scale) > 1e-9)
    {
      RCLCPP_WARN(node_->get_logger(),
                  "trajectory_speed_scale %.3f != file speed_scale %.3f; execution uses the parameter",
                  cfg_.trajectory_speed_scale, file_scale);
    }
    // Isolated sync candidate is explicitly time-parameterized for scale 1.0.
    // This limit applies only to dual7_sync_task_executor built in this worktree.
    if (cfg_.trajectory_speed_scale <= 0.0 || cfg_.trajectory_speed_scale > 1.0)
    {
      err = "keypose_v1 trajectory_speed_scale must be in (0, 1.0]";
      return false;
    }
    for (const auto& s : y["stages"])
    {
      Stage st;
      const std::string loaded_id = s["id"].as<std::string>();
      st.id = loaded_id;
      st.kind = s["kind"].as<std::string>();
      st.moving = s["moving"] ? s["moving"].as<std::string>() : "";
      st.logical = st.id;
      if (s["object_center_world"])
        yamlVec(s["object_center_world"], st.object_center_world);
      if (s["object_bottom_z"])
        st.object_bottom_z = s["object_bottom_z"].as<double>();
      auto load_segment = [&](const YAML::Node& in, TrajectorySegmentRecord& out,
                              const std::string& arm_prefix) -> bool {
        out.logical_segment = st.id + "_" + arm_prefix;
        out.deployable = true;
        if (in["joint_names"])
          for (const auto& n : in["joint_names"])
            out.joint_names.push_back(n.as<std::string>());
        yamlVec(in["start_joints"], out.start_joints);
        yamlVec(in["end_joints"], out.end_joints);
        if (!in["points"] || !in["points"].IsSequence())
          return false;
        for (const auto& p : in["points"])
        {
          fr_task_planner::TrajectoryPointRecord pt;
          yamlVec(p["positions"], pt.positions);
          if (p["velocities"])
            yamlVec(p["velocities"], pt.velocities);
          if (p["accelerations"])
            yamlVec(p["accelerations"], pt.accelerations);
          pt.sec = p["sec"] ? p["sec"].as<int>() : 0;
          pt.nanosec = p["nanosec"] ? static_cast<uint32_t>(p["nanosec"].as<int>()) : 0;
          out.points.push_back(pt);
        }
        return !out.points.empty() && out.start_joints.size() == 6 && out.end_joints.size() == 6;
      };
      if (st.kind == "dual_joint_connection")
      {
        if (!s["status"] || s["status"].as<std::string>() != "PASS" ||
            st.moving != "dual" || !load_segment(s["arm_a"], st.seg_a, "arm_a") ||
            !load_segment(s["arm_b"], st.seg_b, "arm_b"))
        {
          err = "invalid dual_joint_connection " + st.id;
          return false;
        }
        const double da = segmentDuration(st.seg_a), db = segmentDuration(st.seg_b);
        if (std::abs(da - db) > 1e-6)
        {
          err = "dual_joint_connection time bases differ " + st.id;
          return false;
        }
        st.not_cartesian = true;
        if (st.id == "dual_pre_handover_to_handover" ||
            st.id == "dual_face3_and_b_home_to_handover")
          expected_b_handover_ = st.seg_b.end_joints;
      }
      if (st.kind == "joint_connection")
      {
        if (!s["status"] || s["status"].as<std::string>() != "PASS")
        {
          err = "refusing keypose stage " + st.id + " status=" +
                (s["status"] ? s["status"].as<std::string>() : "missing");
          return false;
        }
        st.not_cartesian = true;
        st.seg.logical_segment = st.id;
        st.seg.deployable = true;
        if (s["joint_names"])
          for (const auto& n : s["joint_names"])
            st.seg.joint_names.push_back(n.as<std::string>());
        yamlVec(s["start_joints"], st.seg.start_joints);
        yamlVec(s["end_joints"], st.seg.end_joints);
        for (const auto& p : s["points"])
        {
          fr_task_planner::TrajectoryPointRecord pt;
          yamlVec(p["positions"], pt.positions);
          if (p["velocities"])
            yamlVec(p["velocities"], pt.velocities);
          if (p["accelerations"])
            yamlVec(p["accelerations"], pt.accelerations);
          pt.sec = p["sec"] ? p["sec"].as<int>() : 0;
          pt.nanosec = p["nanosec"] ? static_cast<uint32_t>(p["nanosec"].as<int>()) : 0;
          st.seg.points.push_back(pt);
        }
        if (st.seg.points.empty())
        {
          err = "empty path " + st.id;
          return false;
        }
        if (loaded_id == "b_pre_to_handover")
          expected_b_handover_ = st.seg.end_joints;
        if (loaded_id == "a_handover_to_home")
          a_exit_points_ = st.seg.points;
      }
      stages_.push_back(std::move(st));
      if (loaded_id == "gripper_close_b")
      {
        Stage confirm;
        confirm.id = "physical_grasp_confirm_b";
        confirm.kind = "confirm_b_grasp";
        confirm.moving = "arm_b";
        confirm.requires_b_grasp_confirm = true;
        stages_.push_back(std::move(confirm));
      }
    }
    RCLCPP_INFO(node_->get_logger(),
                "KEYPOSE_V1 stages=%zu speed_scale=%.3f grasp_wait=%.3f handover_wait=%.3f file=%s",
                stages_.size(), cfg_.trajectory_speed_scale, grasp_post_close_wait_sec_,
                handover_release_delay_sec_, path.c_str());
    printIndex();
    return true;
  }

  bool buildStages(std::string& err)
  {
    fr_task_planner::PersistedTrajectory step15;
    if (!readTrajectoryYaml(step15_path_, step15, err))
      return false;
    auto take15 = [&](const std::string& logical) {
      TrajectorySegmentRecord found;
      for (const auto& s : step15.segments)
      {
        if (s.logical_segment == logical && s.deployable)
          found = s;
      }
      prefixJ(found, "arm_a");
      found.logical_segment = logical;
      return found;
    };
    auto addMotion = [&](const std::string& id, const std::string& kind, const std::string& moving,
                         const std::string& logical, TrajectorySegmentRecord seg, bool ompl) {
      Stage st;
      st.id = id;
      st.kind = kind;
      st.moving = moving;
      st.logical = logical;
      st.not_cartesian = ompl;
      st.requires_approach_confirm = (id == "b_handover_approach");
      st.seg = std::move(seg);
      st.seg.logical_segment = logical;
      stages_.push_back(std::move(st));
    };

    Stage home;
    home.id = "dual_current_to_home";
    home.kind = "runtime_replan";
    home.moving = "dual";
    home.logical = "Dual_Current_to_Home";
    home.source = "live named joints → dual8 Home; ZERO gripper";
    stages_.push_back(home);

    addMotion("home_to_pregrasp", "frozen_step15", "arm_a", "Home_to_PreGrasp",
              take15("Home_to_PreGrasp"), false);
    addMotion("pregrasp_to_grasp", "frozen_step15", "arm_a", "PreGrasp_to_Grasp",
              take15("PreGrasp_to_Grasp"), false);
    Stage act_a;
    act_a.id = "gripper_activate_a";
    act_a.kind = "gripper_activate";
    act_a.moving = "arm_a";
    stages_.push_back(act_a);
    Stage close_a;
    close_a.id = "gripper_close_a";
    close_a.kind = "gripper_close";
    close_a.moving = "arm_a";
    stages_.push_back(close_a);
    addMotion("grasp_to_lift", "frozen_step15", "arm_a", "Grasp_to_Lift", take15("Grasp_to_Lift"),
              false);
    addMotion("lift_to_a", "frozen_step15", "arm_a", "Lift_to_A", take15("Lift_to_A"), false);
    addMotion("a_to_b", "frozen_step15", "arm_a", "A_to_B", take15("A_to_B"), false);
    addMotion("b_to_c", "frozen_step15", "arm_a", "B_to_C_original_bottom",
              take15("B_to_C_original_bottom"), false);
    addMotion("b_to_pre_handover", "ompl_connection", "arm_b", "B_to_PreHandover",
              loadBHomeToPre(), true);
    addMotion("a_c_to_handover", "ompl_connection", "arm_a", "C_to_Handover",
              extraSeg("a_c_to_handover"), true);
    addMotion("b_handover_approach", "ompl_connection", "arm_b", "B_HandoverApproach",
              extraSeg("b_handover_approach"), true);
    Stage act_b;
    act_b.id = "gripper_activate_b";
    act_b.kind = "gripper_activate";
    act_b.moving = "arm_b";
    stages_.push_back(act_b);
    Stage close_b;
    close_b.id = "gripper_close_b";
    close_b.kind = "gripper_close";
    close_b.moving = "arm_b";
    stages_.push_back(close_b);
    Stage confirm;
    confirm.id = "physical_grasp_confirm_b";
    confirm.kind = "confirm_b_grasp";
    confirm.requires_b_grasp_confirm = true;
    stages_.push_back(confirm);
    Stage open_a;
    open_a.id = "gripper_open_a";
    open_a.kind = "gripper_open";
    open_a.moving = "arm_a";
    stages_.push_back(open_a);
    Stage attach;
    attach.id = "attachment_transfer";
    attach.kind = "attachment_transfer";
    attach.moving = "arm_b";
    stages_.push_back(attach);
    addMotion("a_return_home", "ompl_connection", "arm_a", "A_ReturnHome", extraSeg("a_return_home"),
              true);
    addMotion("b_plus_x", "dual5t_validated", "arm_b", "B_PlusX", extraSeg("b_plus_x"), false);
    addMotion("b_minus_x", "dual5t_validated", "arm_b", "B_MinusX", extraSeg("b_minus_x"), false);
    addMotion("b_plus_z", "dual5t_validated", "arm_b", "B_PlusZ", extraSeg("b_plus_z"), false);

    for (auto& st : stages_)
    {
      if (st.seg.points.empty() && (st.kind == "frozen_step15" || st.kind == "ompl_connection" ||
                                    st.kind == "dual5t_validated"))
      {
        err = "missing waypoints for " + st.id;
        return false;
      }
    }
    printIndex();
    return true;
  }

  void printIndex()
  {
    RCLCPP_INFO(node_->get_logger(), "stage index (default does not auto-run all):");
    for (const auto& st : stages_)
      RCLCPP_INFO(node_->get_logger(), "  %-28s kind=%-18s moving=%-6s cartesian=%s points=%zu",
                  st.id.c_str(), st.kind.c_str(), st.moving.c_str(),
                  st.not_cartesian ? "NO-OMPL" : "n/a-or-frozen", st.seg.points.size());
  }

  void onJs(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mu_);
    latest_.joints.clear();
    latest_.velocities.clear();
    const size_t n = std::min(msg->name.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i)
    {
      latest_.joints[msg->name[i]] = msg->position[i];
      if (i < msg->velocity.size())
        latest_.velocities[msg->name[i]] = msg->velocity[i];
    }
    latest_.stamp_valid = true;
    last_stamp_ = rclcpp::Time(msg->header.stamp);
    latest_.age_sec = std::max(0.0, (node_->get_clock()->now() - last_stamp_).seconds());
    have_js_ = true;
  }

  std::optional<JointSnapshot> snap()
  {
    for (int i = 0; i < 20 && rclcpp::ok(); ++i)
    {
      rclcpp::spin_some(node_);
      std::lock_guard<std::mutex> lock(mu_);
      if (have_js_)
      {
        auto s = latest_;
        s.age_sec = std::max(0.0, (node_->get_clock()->now() - last_stamp_).seconds());
        return s;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!have_js_)
      return std::nullopt;
    auto s = latest_;
    s.age_sec = std::max(0.0, (node_->get_clock()->now() - last_stamp_).seconds());
    return s;
  }

  void waitLive(double timeout_sec)
  {
    const auto t0 = std::chrono::steady_clock::now();
    while (rclcpp::ok())
    {
      rclcpp::spin_some(node_);
      const double w =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      bool js = false;
      {
        std::lock_guard<std::mutex> lock(mu_);
        js = have_js_;
      }
      const bool a = client_a_ && client_a_->action_server_is_ready();
      const bool b = client_b_ && client_b_->action_server_is_ready();
      if ((js && a && b) || w >= timeout_sec)
      {
        live_js_ = js;
        live_a_ = a;
        live_b_ = b;
        live_ga_ = grip_client_a_ && grip_client_a_->service_is_ready();
        live_gb_ = grip_client_b_ && grip_client_b_->service_is_ready();
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  void inspect()
  {
    auto s = snap();
    if (!s)
    {
      RCLCPP_WARN(node_->get_logger(),
                  "live /joint_states: not available (ok for offline dry-run)");
      return;
    }
    const auto qa = named(*s, kJa);
    const auto qb = named(*s, kJb);
    const bool have_a = hasNames(*s, kJa);
    const bool have_b = hasNames(*s, kJb);
    mock_zero_ = have_a && have_b && nearZero(qa) && nearZero(qb);
    RCLCPP_INFO(node_->get_logger(), "live Arm A %s names=%s", fmt(qa).c_str(),
                have_a ? "ok" : "MISSING");
    RCLCPP_INFO(node_->get_logger(), "live Arm B %s names=%s", fmt(qb).c_str(),
                have_b ? "ok" : "MISSING");
    if (home_a_.size() == 6)
      RCLCPP_INFO(node_->get_logger(), "Arm A Home target rad %s", fmt(home_a_).c_str());
    if (home_b_deg_.size() == 6)
      RCLCPP_INFO(node_->get_logger(), "Arm B Home target deg %s", fmt(home_b_deg_).c_str());
    if (home_b_.size() == 6)
      RCLCPP_INFO(node_->get_logger(), "Arm B Home target rad %s", fmt(home_b_).c_str());
    if (pre_b_.size() == 6)
      RCLCPP_INFO(node_->get_logger(), "Arm B pre-handover target %s", fmt(pre_b_).c_str());
    RCLCPP_INFO(node_->get_logger(), "joint age=%.3f s mock_zero=%s", s->age_sec,
                mock_zero_ ? "YES" : "NO");
    if (mock_zero_)
      RCLCPP_WARN(node_->get_logger(),
                  "live joints look like mock zeros; not a valid real-task start");
    RCLCPP_INFO(node_->get_logger(), "action A %s: %s", action_a_.c_str(), live_a_ ? "available" : "offline");
    RCLCPP_INFO(node_->get_logger(), "action B %s: %s", action_b_.c_str(), live_b_ ? "available" : "offline");
    RCLCPP_INFO(node_->get_logger(), "gripper A %s: %s", grip_a_.c_str(), live_ga_ ? "available" : "offline");
    RCLCPP_INFO(node_->get_logger(), "gripper B %s: %s", grip_b_.c_str(), live_gb_ ? "available" : "offline");
  }

  bool partOwnership(bool& on_a, bool& on_b, std::string& err)
  {
    on_a = false;
    on_b = false;
    auto client = node_->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
    if (!client->wait_for_service(std::chrono::seconds(2)))
    {
      err = "get_planning_scene unavailable";
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    req->components.components = moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE_ATTACHED_OBJECTS;
    auto fut = client->async_send_request(req);
    if (rclcpp::spin_until_future_complete(node_, fut, std::chrono::seconds(3)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      err = "get_planning_scene timed out";
      return false;
    }
    const auto resp = fut.get();
    if (!resp)
    {
      err = "empty planning scene";
      return false;
    }
    for (const auto& att : resp->scene.robot_state.attached_collision_objects)
    {
      if (att.object.id != "small_part")
        continue;
      if (att.object.operation == moveit_msgs::msg::CollisionObject::REMOVE)
        continue;
      if (att.link_name.find("arm_a_") == 0)
        on_a = true;
      if (att.link_name.find("arm_b_") == 0)
        on_b = true;
    }
    return true;
  }

  bool exitPathClear(const JointSnapshot& live)
  {
    if (a_exit_points_.empty() || expected_b_handover_.size() != 6)
      return false;
    auto client = node_->create_client<moveit_msgs::srv::GetStateValidity>("/check_state_validity");
    if (!client->wait_for_service(std::chrono::seconds(2)))
      return false;
    const auto live_b = named(live, kJb);
    const int stride = std::max(1, static_cast<int>(a_exit_points_.size() / 12));
    for (size_t i = 0; i < a_exit_points_.size(); i += static_cast<size_t>(stride))
    {
      auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
      req->group_name = "arm_a";
      req->robot_state.is_diff = false;
      req->robot_state.joint_state.name = kJa;
      req->robot_state.joint_state.name.insert(req->robot_state.joint_state.name.end(), kJb.begin(),
                                               kJb.end());
      req->robot_state.joint_state.position = a_exit_points_[i].positions;
      req->robot_state.joint_state.position.insert(req->robot_state.joint_state.position.end(),
                                                   live_b.begin(), live_b.end());
      auto fut = client->async_send_request(req);
      if (rclcpp::spin_until_future_complete(node_, fut, std::chrono::seconds(2)) !=
          rclcpp::FutureReturnCode::SUCCESS)
        return false;
      const auto resp = fut.get();
      if (!resp || !resp->valid)
        return false;
    }
    return true;
  }

  bool aExitReady(const JointSnapshot& live)
  {
    fr_task_planner::KeyposeExitGate gate;
    gate.feedback_fresh = live.age_sec <= max_joint_age_sec_;
    gate.b_close_finished = b_close_success_;
    gate.handover_confirmed = physical_b_grasp_;
    gate.a_open = handover_a_opened_;
    if (expected_b_handover_.size() == 6)
    {
      const auto chk = checkDual(live, kJb, expected_b_handover_, cfg_.segment_start_tolerance_rad,
                                 "B_NOT_AT_HANDOVER");
      gate.b_at_handover = chk.ok;
    }
    std::string scene_err;
    if (!partOwnership(gate.part_on_a, gate.part_on_b, scene_err))
    {
      RCLCPP_ERROR(node_->get_logger(), "a_handover_to_home scene check failed: %s", scene_err.c_str());
      return false;
    }
    gate.exit_path_clear = exitPathClear(live);
    const std::string why = fr_task_planner::keyposeExitGateError(gate);
    if (!why.empty())
    {
      RCLCPP_ERROR(node_->get_logger(), "a_handover_to_home refused: %s", why.c_str());
      return false;
    }
    RCLCPP_INFO(node_->get_logger(),
                "a_handover_to_home gate PASS: B at handover, close+confirm done, A open, part on B");
    return true;
  }

  bool attachPartToB()
  {
    if (keypose_dir_.empty())
    {
      RCLCPP_ERROR(node_->get_logger(), "keypose_dir missing; cannot attach part to B");
      return false;
    }
    YAML::Node tgt;
    try
    {
      tgt = YAML::LoadFile(keypose_dir_ + "/task_space_targets.yaml");
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(node_->get_logger(), "grasp pose load failed: %s", e.what());
      return false;
    }
    const auto pose = tgt["T_tcpB_object"]["pose"];
    moveit_msgs::msg::AttachedCollisionObject att;
    att.link_name = "arm_b_gripper_tcp";
    att.touch_links = {"arm_b_gripper_base_link", "arm_b_finger_l", "arm_b_finger_r", "arm_b_gripper_tcp"};
    att.object.id = "small_part";
    att.object.header.frame_id = att.link_name;
    att.object.operation = moveit_msgs::msg::CollisionObject::ADD;
    att.object.primitives.resize(1);
    att.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    att.object.primitives[0].dimensions = {0.035, 0.0075};
    geometry_msgs::msg::Pose p;
    p.position.x = pose["xyz"][0].as<double>();
    p.position.y = pose["xyz"][1].as<double>();
    p.position.z = pose["xyz"][2].as<double>();
    p.orientation.x = pose["xyzw"][0].as<double>();
    p.orientation.y = pose["xyzw"][1].as<double>();
    p.orientation.z = pose["xyzw"][2].as<double>();
    p.orientation.w = pose["xyzw"][3].as<double>();
    att.object.primitive_poses.push_back(p);
    moveit_msgs::msg::AttachedCollisionObject drop_a;
    drop_a.link_name = "arm_a_gripper_tcp";
    drop_a.object.id = "small_part";
    drop_a.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    auto client = node_->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");
    if (!client->wait_for_service(std::chrono::seconds(2)))
    {
      RCLCPP_ERROR(node_->get_logger(), "apply_planning_scene unavailable; part stays unassigned");
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    req->scene.is_diff = true;
    req->scene.robot_state.is_diff = true;
    req->scene.robot_state.attached_collision_objects = {drop_a, att};
    auto fut = client->async_send_request(req);
    if (rclcpp::spin_until_future_complete(node_, fut, std::chrono::seconds(3)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "apply_planning_scene timed out");
      return false;
    }
    const auto resp = fut.get();
    if (!resp || !resp->success)
    {
      RCLCPP_ERROR(node_->get_logger(), "apply_planning_scene rejected the B attachment");
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "planning scene: small_part attached to arm_b_gripper_tcp");
    return true;
  }

  bool releasePartAtPlace(const Stage& st)
  {
    if (st.object_center_world.size() != 3 || st.object_bottom_z <= 0.0)
    {
      RCLCPP_ERROR(node_->get_logger(), "place release missing object_center_world/object_bottom_z");
      return false;
    }
    moveit_msgs::msg::AttachedCollisionObject detach;
    detach.link_name = "arm_b_gripper_tcp";
    detach.object.id = "small_part";
    detach.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    moveit_msgs::msg::CollisionObject world;
    world.id = "small_part";
    world.header.frame_id = "world";
    world.operation = moveit_msgs::msg::CollisionObject::ADD;
    world.primitives.resize(1);
    world.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    world.primitives[0].dimensions = {0.035, 0.0075};
    geometry_msgs::msg::Pose p;
    p.position.x = st.object_center_world[0];
    p.position.y = st.object_center_world[1];
    p.position.z = st.object_center_world[2];
    p.orientation.w = 1.0;  // cylinder axis is world +Z after placement
    world.primitive_poses.push_back(p);
    auto client = node_->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");
    if (!client->wait_for_service(std::chrono::seconds(2)))
    {
      RCLCPP_ERROR(node_->get_logger(), "apply_planning_scene unavailable; refusing retreat");
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    req->scene.is_diff = true;
    req->scene.robot_state.is_diff = true;
    req->scene.robot_state.attached_collision_objects = {detach};
    req->scene.world.collision_objects = {world};
    auto fut = client->async_send_request(req);
    if (rclcpp::spin_until_future_complete(node_, fut, std::chrono::seconds(3)) != rclcpp::FutureReturnCode::SUCCESS ||
        !fut.get() || !fut.get()->success)
    {
      RCLCPP_ERROR(node_->get_logger(), "apply_planning_scene rejected placement world object; refusing retreat");
      return false;
    }
    owner_logic_ = "world";
    RCLCPP_INFO(node_->get_logger(), "PLACE RELEASE PASS: small_part world center=[%.12f, %.12f, %.12f] bottom_z=%.12f",
                p.position.x, p.position.y, p.position.z, st.object_bottom_z);
    return true;
  }

  bool runStage(Stage& st, bool motion)
  {
    RCLCPP_INFO(node_->get_logger(), "----- STAGE %s kind=%s moving=%s -----", st.id.c_str(),
                st.kind.c_str(), st.moving.c_str());
    if (st.kind == "runtime_replan")
      return runDualHome(st, motion);
    if (st.kind == "dual_joint_connection")
      return runDualJointStage(st, motion);
    if (st.kind == "gripper_activate" || st.kind == "gripper_close" || st.kind == "gripper_open" ||
        st.kind == "gripper_open_event")
    {
      return runGripperStage(st, motion);
    }
    if (st.kind == "confirm_b_grasp")
    {
      RCLCPP_INFO(node_->get_logger(),
                  "PHYSICAL GRASP GATE: model Attachment is NOT success. "
                  "MoveGripper done is NOT physical grasp. "
                  "object_logic_owner=%s",
                  owner_logic_.c_str());
      if (!motion)
      {
        if (auto_handover_release_)
        {
          RCLCPP_INFO(node_->get_logger(), "AUTO_HANDOVER_RELEASE: ENABLED");
          RCLCPP_INFO(node_->get_logger(), "AUTO HANDOVER MODE");
          RCLCPP_INFO(node_->get_logger(),
                      "dry-run: gripper_close_b SENT=NO; skip HANDOVER_WAIT; "
                      "ARM_A_RELEASE_ALLOWED is NOT claimed (no real B CLOSE SUCCESS)");
        }
        else
        {
          RCLCPP_INFO(node_->get_logger(),
                      "dry-run: would pause for operator confirm via "
                      "ros2 service call /%s/confirm_b_grasp "
                      "std_srvs/srv/SetBool \"{data: true}\"; not auto-confirming",
                      node_->get_name());
        }
        if (inject_failure_ == "B_GRASP_UNCONFIRMED" || inject_failure_ == "USER_CANCEL")
        {
          RCLCPP_ERROR(node_->get_logger(), "inject_failure=%s: A will NOT open",
                       inject_failure_.c_str());
          return false;
        }
        return true;
      }
      if (inject_failure_ == "B_GRASP_UNCONFIRMED" || inject_failure_ == "USER_CANCEL")
      {
        RCLCPP_ERROR(node_->get_logger(), "inject_failure=%s: A will NOT open",
                     inject_failure_.c_str());
        return false;
      }
      if (auto_handover_release_)
      {
        if (!waitAutoHandoverRelease())
          return false;
        return true;
      }
      if (!waitBGraspConfirm())
        return false;
      physical_b_grasp_ = true;
      return true;
    }
    if (st.kind == "attachment_transfer")
    {
      RCLCPP_INFO(node_->get_logger(),
                  "ATTACHMENT TRANSFER: planning-scene Attachment is NOT physical success. "
                  "Physical transfer is B grasp confirm + A open only.");
      if (!motion)
      {
        if (inject_failure_ == "B_GRASP_UNCONFIRMED")
          return false;
        owner_logic_ = "B";
        RCLCPP_INFO(node_->get_logger(),
                    "dry-run logical owner -> B; PHYSICAL HOLD still UNVERIFIED");
        return true;
      }
      if (!physical_b_grasp_)
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "refusing attachment_transfer without runtime B grasp confirm");
        return false;
      }
      if (!handover_a_opened_)
      {
        RCLCPP_ERROR(node_->get_logger(), "refusing attachment_transfer before A has opened");
        return false;
      }
      if (!attachPartToB())
        return false;
      owner_logic_ = "B";
      return true;
    }
    if (st.requires_approach_confirm)
    {
      RCLCPP_INFO(node_->get_logger(),
                  "handover approach kind=OMPL_CONNECTION not_cartesian_linear=YES "
                  "confirm_handover_approach=%s",
                  confirm_approach_ ? "true" : "false");
      if (motion && !confirm_approach_)
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "refusing OMPL approach without confirm_handover_approach:=true");
        return false;
      }
    }

    assignConservativeTimes(st.seg, vmax_);
    auto scaled = scaleSegment(st.seg, cfg_.trajectory_speed_scale);
    if (!scaled.ok)
    {
      RCLCPP_ERROR(node_->get_logger(), "scale failed: %s", scaled.error.c_str());
      return false;
    }
    RCLCPP_INFO(node_->get_logger(),
                "EFFECTIVE_SPEED_SCALE=%.3f original_s=%.3f scaled_s=%.3f",
                cfg_.trajectory_speed_scale, segmentDuration(st.seg),
                segmentDuration(scaled.segment));
    RCLCPP_INFO(node_->get_logger(), "points=%zu original=%.3f s scaled=%.3f s start=%s end=%s",
                scaled.segment.points.size(), segmentDuration(st.seg),
                segmentDuration(scaled.segment), fmt(st.seg.start_joints).c_str(),
                fmt(st.seg.end_joints).c_str());
    RCLCPP_INFO(node_->get_logger(), "predicted A %s  predicted B %s  owner=%s",
                fmt(pred_a_).c_str(), fmt(pred_b_).c_str(), owner_logic_.c_str());
    const auto& names = st.moving == "arm_b" ? kJb : kJa;
    const auto& pred = st.moving == "arm_b" ? pred_b_ : pred_a_;
    const bool zero_start = nearZero(st.seg.start_joints);
    if (zero_start)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "stage %s frozen start is model-zero; not a valid Dual-8 start", st.id.c_str());
      return false;
    }
    if (st.id == "b_handover_approach" && handover_a_.size() == 6 && pred_a_.size() == 6)
    {
      const auto a_at = checkVec(pred_a_, handover_a_, kJa, cfg_.segment_start_tolerance_rad,
                                 "HANDOVER_A_COMPANION_MISMATCH");
      if (!a_at.ok)
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "Arm B approach requires Arm A at handover: %s predicted_a=%s expected=%s",
                     a_at.error.c_str(), fmt(pred_a_).c_str(), fmt(handover_a_).c_str());
        return false;
      }
    }
    if (st.id == "b_to_pre_handover" && home_b_.size() == 6)
    {
      RCLCPP_INFO(node_->get_logger(), "B Home→Pre using Dual-8 connection, not model-zero");
    }
    if (!st.seg.start_joints.empty() && pred.size() == 6)
    {
      const auto chk = checkVec(pred, scaled.segment.start_joints, names,
                                cfg_.segment_start_tolerance_rad, "FROZEN_START_STATE_MISMATCH");
      if (!chk.ok)
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "TRAJECTORY_CHAIN break at %s: %s predicted=%s stage_start=%s",
                     st.id.c_str(), chk.error.c_str(), fmt(pred).c_str(),
                     fmt(scaled.segment.start_joints).c_str());
        return false;
      }
    }
    if (inject_failure_ == "ACTION_REJECT")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=ACTION_REJECT at %s", st.id.c_str());
      return false;
    }
    if (inject_failure_ == "TRAJ_TIMEOUT")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=TRAJ_TIMEOUT at %s", st.id.c_str());
      return false;
    }
    if (!motion)
    {
      RCLCPP_INFO(node_->get_logger(),
                  "dry-run: comparing PREDICTED joints (not launch-time live) to stage.start; "
                  "not sending FollowJointTrajectory");
      applyPredEnd(st, scaled.segment.end_joints);
      return true;
    }
    auto live = snap();
    if (!live)
    {
      RCLCPP_ERROR(node_->get_logger(), "no live joint_states");
      return false;
    }
    if (live->age_sec > max_joint_age_sec_)
    {
      RCLCPP_ERROR(node_->get_logger(), "stale joint_states age=%.3f", live->age_sec);
      return false;
    }
    const bool skip_pregrasp_start_check = st.id == "pregrasp_to_grasp";
    const bool skip_preplace_start_check = st.id == "pre_place_to_place";
    const bool skip_lift_start_check = st.id == "lift_to_face1";
    const bool skip_fast_start_check =
        st.id == "face1_to_face2" || st.id == "face2_to_face3" ||
        st.id == "face4_to_face5" || st.id == "face5_to_face6" ||
        st.id == "face6_to_pre_place" || st.id == "place_to_retreat";
    const auto start_chk = checkDual(*live, names, scaled.segment.start_joints,
                                     cfg_.segment_start_tolerance_rad, "FROZEN_START_STATE_MISMATCH");
    if (!skip_pregrasp_start_check && !skip_preplace_start_check &&
        !skip_lift_start_check && !skip_fast_start_check && !start_chk.ok)
    {
      RCLCPP_ERROR(node_->get_logger(), "live start mismatch %s predicted=%s live=%s stage_start=%s",
                   start_chk.error.c_str(), fmt(pred).c_str(), fmt(named(*live, names)).c_str(),
                   fmt(scaled.segment.start_joints).c_str());
      return false;
    }
    if ((st.moving == "arm_b" ? servoj_block_b_ : servoj_block_a_))
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "SERVOJ_RESTORE_UNCONFIRMED: refusing FollowJointTrajectory on %s after a "
                   "gripper timeout/busy; wait for ServoJ resume, do not retry a move as if "
                   "the previous gripper call finished",
                   st.moving.c_str());
      return false;
    }
    if (inject_failure_ == "ACTION_REJECT")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=ACTION_REJECT");
      return false;
    }
    if (inject_failure_ == "TRAJ_TIMEOUT")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=TRAJ_TIMEOUT");
      return false;
    }
    if (st.id == "a_handover_to_home" && !aExitReady(*live))
      return false;
    if (!sendTraj(st.moving, scaled.segment))
      return false;
    const bool skip_pregrasp_end_check = st.id == "home_to_pregrasp";
    const bool skip_preplace_end_check = st.id == "face6_to_pre_place";
    const bool skip_lift_end_check = st.id == "grasp_to_lift";
    if (!skip_pregrasp_end_check && !skip_preplace_end_check && !skip_lift_end_check &&
        !waitEnd(st.moving, names, scaled.segment.end_joints))
      return false;
    if (skip_pregrasp_end_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_PREGRASP_END_CHECK: proceeding directly to grasp approach");
    if (skip_pregrasp_start_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_PREGRASP_START_CHECK: predecessor endpoint settle is intentionally skipped");
    if (skip_preplace_end_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_PREPLACE_END_CHECK: proceeding directly to vertical descent");
    if (skip_preplace_start_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_PREPLACE_START_CHECK: predecessor endpoint settle is intentionally skipped");
    if (skip_lift_end_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_LIFT_END_CHECK: proceeding directly to FACE1 transition");
    if (skip_lift_start_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_LIFT_START_CHECK: predecessor endpoint settle is intentionally skipped");
    if (skip_fast_start_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_FAST_STAGE_START_CHECK: %s", st.id.c_str());
    applyPredEnd(st, scaled.segment.end_joints);
    return true;
  }

  // This path is intentionally separate from sendTraj(): legacy stages are
  // serial, while a dual stage is sent to both independent controllers with
  // one future start stamp.  It is software-synchronised, not controller-bus
  // hardware synchronisation.
  bool runDualJointStage(Stage& st, bool motion)
  {
    if (st.seg_a.start_joints.size() != 6 || st.seg_b.start_joints.size() != 6)
      return false;
    // The sync-candidate generator has already put both arms on one checked
    // time base.  Do not re-time each arm independently here.
    auto a = scaleSegment(st.seg_a, cfg_.trajectory_speed_scale);
    auto b = scaleSegment(st.seg_b, cfg_.trajectory_speed_scale);
    if (!a.ok || !b.ok || std::abs(segmentDuration(a.segment) - segmentDuration(b.segment)) > 1e-6)
    {
      RCLCPP_ERROR(node_->get_logger(), "DUAL_TIMEBASE_INVALID %s", st.id.c_str());
      return false;
    }
    if (!checkVec(pred_a_, a.segment.start_joints, kJa, cfg_.segment_start_tolerance_rad,
                  "DUAL_A_START_MISMATCH").ok ||
        !checkVec(pred_b_, b.segment.start_joints, kJb, cfg_.segment_start_tolerance_rad,
                  "DUAL_B_START_MISMATCH").ok)
    {
      RCLCPP_ERROR(node_->get_logger(), "DUAL_PREDICTED_START_MISMATCH %s", st.id.c_str());
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "DUAL_SYNC %s common_duration=%.3f scale=%.3f A_points=%zu B_points=%zu",
                st.id.c_str(), segmentDuration(a.segment), cfg_.trajectory_speed_scale,
                a.segment.points.size(), b.segment.points.size());
    if (!motion)
    {
      pred_a_ = a.segment.end_joints;
      pred_b_ = b.segment.end_joints;
      RCLCPP_INFO(node_->get_logger(), "dry-run: DUAL goals suppressed; both predicted endpoints advanced");
      return true;
    }
    const bool skip_dual_live_start_check =
        st.id == "dual_face3_and_b_home_to_handover_via_pre" ||
        st.id == "dual_a_handover_to_home_and_b_handover_to_face4";
    auto live = snap();
    if (!live || live->age_sec > max_joint_age_sec_ ||
        (!skip_dual_live_start_check &&
         (!checkVec(named(*live, kJa), a.segment.start_joints, kJa, cfg_.segment_start_tolerance_rad,
                    "DUAL_A_LIVE_START_MISMATCH").ok ||
          !checkVec(named(*live, kJb), b.segment.start_joints, kJb, cfg_.segment_start_tolerance_rad,
                    "DUAL_B_LIVE_START_MISMATCH").ok)))
    {
      RCLCPP_ERROR(node_->get_logger(), "DUAL_LIVE_START_MISMATCH %s", st.id.c_str());
      return false;
    }
    if (skip_dual_live_start_check)
      RCLCPP_INFO(node_->get_logger(), "SKIP_DUAL_LIVE_START_CHECK: %s", st.id.c_str());
    if (!sendDualTrajSync(a.segment, b.segment))
      return false;
    if (!waitEnd("arm_a", kJa, a.segment.end_joints) ||
        !waitEnd("arm_b", kJb, b.segment.end_joints))
      return false;
    pred_a_ = a.segment.end_joints;
    pred_b_ = b.segment.end_joints;
    return true;
  }

  bool sendDualTrajSync(const TrajectorySegmentRecord& raw_a, const TrajectorySegmentRecord& raw_b)
  {
    TrajectorySegmentRecord a, b;
    std::string err;
    if (!remapSegmentByJointName(raw_a, kJa, a, err) || !remapSegmentByJointName(raw_b, kJb, b, err))
    {
      RCLCPP_ERROR(node_->get_logger(), "DUAL_REMAP_FAIL %s", err.c_str());
      return false;
    }
    if (!client_a_->wait_for_action_server(std::chrono::seconds(2)) ||
        !client_b_->wait_for_action_server(std::chrono::seconds(2)))
    {
      RCLCPP_ERROR(node_->get_logger(), "DUAL_ACTION_UNAVAILABLE");
      return false;
    }
    const auto start_stamp = node_->get_clock()->now() + rclcpp::Duration::from_seconds(0.25);
    FollowJointTrajectory::Goal ga, gb;
    ga.trajectory = toTraj(a); gb.trajectory = toTraj(b);
    ga.trajectory.header.stamp = start_stamp; gb.trajectory.header.stamp = start_stamp;
    RCLCPP_INFO(node_->get_logger(), "DUAL_GOALS_SEND shared_start_plus_sec=0.250 A=%s B=%s",
                action_a_.c_str(), action_b_.c_str());
    auto fa = client_a_->async_send_goal(ga);
    auto fb = client_b_->async_send_goal(gb);
    if (rclcpp::spin_until_future_complete(node_, fa, std::chrono::seconds(15)) != rclcpp::FutureReturnCode::SUCCESS ||
        rclcpp::spin_until_future_complete(node_, fb, std::chrono::seconds(15)) != rclcpp::FutureReturnCode::SUCCESS)
      return false;
    auto ha = fa.get(); auto hb = fb.get();
    if (!ha || !hb)
    {
      if (ha) client_a_->async_cancel_goal(ha);
      if (hb) client_b_->async_cancel_goal(hb);
      RCLCPP_ERROR(node_->get_logger(), "DUAL_GOAL_REJECTED; accepted peer cancellation requested; no gripper release");
      return false;
    }
    ++motion_sent_; ++motion_sent_;
    const double timeout = actionTimeoutSec(segmentDuration(a), cfg_.timeout_factor, cfg_.timeout_margin_sec);
    auto ra = client_a_->async_get_result(ha);
    auto rb = client_b_->async_get_result(hb);
    const auto wa = rclcpp::spin_until_future_complete(node_, ra, std::chrono::duration<double>(timeout));
    const auto wb = rclcpp::spin_until_future_complete(node_, rb, std::chrono::duration<double>(timeout));
    const bool oka = wa == rclcpp::FutureReturnCode::SUCCESS && ra.get().code == rclcpp_action::ResultCode::SUCCEEDED;
    const bool okb = wb == rclcpp::FutureReturnCode::SUCCESS && rb.get().code == rclcpp_action::ResultCode::SUCCEEDED;
    if (!oka || !okb)
    {
      if (!oka) client_b_->async_cancel_goal(hb);
      if (!okb) client_a_->async_cancel_goal(ha);
      RCLCPP_ERROR(node_->get_logger(), "DUAL_RESULT_FAIL A=%s B=%s; peer cancellation requested; no later stage scheduled",
                   oka ? "SUCCESS" : "FAIL", okb ? "SUCCESS" : "FAIL");
      return false;
    }
    return true;
  }

  bool sceneHasAttachedPart(bool& attached, std::string& err)
  {
    attached = false;
    auto client = node_->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
    if (!client->wait_for_service(std::chrono::seconds(2)))
    {
      err = "get_planning_scene unavailable";
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    req->components.components = moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE_ATTACHED_OBJECTS;
    auto fut = client->async_send_request(req);
    if (rclcpp::spin_until_future_complete(node_, fut, std::chrono::seconds(3)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      err = "get_planning_scene timed out";
      return false;
    }
    const auto resp = fut.get();
    attached = resp && !resp->scene.robot_state.attached_collision_objects.empty();
    return true;
  }

  bool directHomeFree(const std::string& group, const std::vector<std::string>& names,
                      const std::vector<double>& live_a, const std::vector<double>& live_b,
                      const std::vector<double>& goal, std::string& block)
  {
    auto client = node_->create_client<moveit_msgs::srv::GetStateValidity>("/check_state_validity");
    if (!client->wait_for_service(std::chrono::seconds(2)))
    {
      block = "check_state_validity unavailable";
      return false;
    }
    const std::vector<double>& from = (group == "arm_b") ? live_b : live_a;
    double jump = 0.0;
    for (size_t i = 0; i < from.size() && i < goal.size(); ++i)
      jump = std::max(jump, std::abs(goal[i] - from[i]));
    const int n = std::max(1, static_cast<int>(std::ceil(jump / 0.02)));
    for (int s = 0; s <= n; ++s)
    {
      const double t = static_cast<double>(s) / static_cast<double>(n);
      std::vector<double> qa = live_a;
      std::vector<double> qb = live_b;
      auto& moving = (group == "arm_b") ? qb : qa;
      for (size_t i = 0; i < moving.size() && i < goal.size(); ++i)
        moving[i] = from[i] + (goal[i] - from[i]) * t;
      auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
      req->group_name = group;
      req->robot_state.is_diff = false;
      req->robot_state.joint_state.name = kJa;
      req->robot_state.joint_state.name.insert(req->robot_state.joint_state.name.end(), kJb.begin(),
                                               kJb.end());
      req->robot_state.joint_state.position = qa;
      req->robot_state.joint_state.position.insert(req->robot_state.joint_state.position.end(),
                                                   qb.begin(), qb.end());
      auto fut = client->async_send_request(req);
      if (rclcpp::spin_until_future_complete(node_, fut, std::chrono::seconds(2)) !=
          rclcpp::FutureReturnCode::SUCCESS)
      {
        block = "check_state_validity timed out";
        return false;
      }
      const auto resp = fut.get();
      if (!resp || !resp->valid)
      {
        block = "direct collision";
        if (resp && !resp->contacts.empty())
          block = resp->contacts.front().contact_body_1 + " <-> " + resp->contacts.front().contact_body_2;
        return false;
      }
    }
    (void)names;
    return true;
  }

  bool fillDirectHome(const std::vector<std::string>& names, const std::vector<double>& from,
                      const std::vector<double>& goal, TrajectorySegmentRecord& out)
  {
    out.joint_names = names;
    out.points.clear();
    double jump = 0.0;
    for (size_t i = 0; i < from.size() && i < goal.size(); ++i)
      jump = std::max(jump, std::abs(goal[i] - from[i]));
    const int n = std::max(1, static_cast<int>(std::ceil(jump / 0.02)));
    double tsec = 0.0;
    std::vector<double> prev = from;
    for (int s = 0; s <= n; ++s)
    {
      const double u = static_cast<double>(s) / static_cast<double>(n);
      fr_task_planner::TrajectoryPointRecord pt;
      pt.positions.resize(from.size());
      for (size_t i = 0; i < from.size() && i < goal.size(); ++i)
        pt.positions[i] = from[i] + (goal[i] - from[i]) * u;
      if (s > 0)
      {
        double step = 0.0;
        for (size_t i = 0; i < pt.positions.size(); ++i)
          step = std::max(step, std::abs(pt.positions[i] - prev[i]));
        tsec += std::max(step / 1.0, 0.02);
      }
      pt.sec = static_cast<int32_t>(tsec);
      pt.nanosec = static_cast<uint32_t>((tsec - pt.sec) * 1e9);
      out.points.push_back(pt);
      prev = pt.positions;
    }
    out.start_joints = from;
    out.end_joints = goal;
    out.deployable = true;
    return true;
  }

  bool planCurrentToHome(const std::string& group, const std::vector<std::string>& names,
                         const std::vector<double>& live_a, const std::vector<double>& live_b,
                         const std::vector<double>& goal, TrajectorySegmentRecord& out, std::string& err,
                         bool& used_rrtconnect)
  {
    used_rrtconnect = false;
    std::string block;
    if (directHomeFree(group, names, live_a, live_b, goal, block))
    {
      RCLCPP_INFO(node_->get_logger(), "%s Current→Home direct joint path is collision-free",
                  group.c_str());
      const auto& from = (group == "arm_b") ? live_b : live_a;
      return fillDirectHome(names, from, goal, out);
    }
    RCLCPP_INFO(node_->get_logger(),
                "%s Current→Home direct blocked (%s); using existing move_group RRTConnect",
                group.c_str(), block.c_str());
    used_rrtconnect = true;
    return planArmHome(group, names, live_a, live_b, goal, out, err);
  }

  bool runDualHome(const Stage& st, bool motion)
  {
    (void)st;
    RCLCPP_INFO(node_->get_logger(),
                "DUAL-8 dual_current_to_home: live named joints → Home A + Home B. "
                "ZERO gripper commands. frozen Current_to_Home is not used.");
    RCLCPP_INFO(node_->get_logger(),
                "SIMULTANEOUS HOME: NOT AVAILABLE (two independent FollowJointTrajectory "
                "controllers; simultaneous_home=%s ignored for send)",
                simultaneous_home_ ? "true" : "false");
    RCLCPP_INFO(node_->get_logger(), "SEQUENTIAL HOME: A then B (other arm held at actual state)");
    if (home_a_.size() != 6 || home_b_.size() != 6)
    {
      RCLCPP_ERROR(node_->get_logger(), "dual8 Home vectors missing; not inventing zeros");
      return false;
    }
    auto s = snap();
    if (!s)
    {
      if (!motion)
      {
        RCLCPP_WARN(node_->get_logger(),
                    "dry-run: no live /joint_states; cannot prove a real Current→Home start");
        return true;
      }
      RCLCPP_ERROR(node_->get_logger(), "no live /joint_states; refusing Current→Home");
      return false;
    }
    if (!hasNames(*s, kJa) || !hasNames(*s, kJb))
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "live /joint_states missing arm_a_j* or arm_b_j*; not using array order");
      return false;
    }
    const auto qa = named(*s, kJa);
    const auto qb = named(*s, kJb);
    RCLCPP_INFO(node_->get_logger(), "live Arm A (by name) %s", fmt(qa).c_str());
    RCLCPP_INFO(node_->get_logger(), "live Arm B (by name) %s", fmt(qb).c_str());
    RCLCPP_INFO(node_->get_logger(), "Home A rad %s", fmt(home_a_).c_str());
    if (home_b_deg_.size() == 6)
      RCLCPP_INFO(node_->get_logger(), "Home B deg %s", fmt(home_b_deg_).c_str());
    RCLCPP_INFO(node_->get_logger(), "Home B rad %s", fmt(home_b_).c_str());
    if (nearZero(qa) && nearZero(qb))
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "MOCK_ZERO_NOT_REAL_START: live A/B look like zeros; refusing Home plan");
      return false;
    }
    if (motion && task_mode_ == "keypose_v1")
    {
      bool attached = false;
      std::string scene_err;
      if (!sceneHasAttachedPart(attached, scene_err))
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "Current→Home cannot see the planning scene (%s). Not assuming the grippers "
                     "are empty and not moving.",
                     scene_err.c_str());
        return false;
      }
      if (attached)
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "Planning scene has an attached part. Refusing Current→Home and refusing to "
                     "open either gripper. Confirm the part on site; this program will not assume "
                     "the arms are empty.");
        return false;
      }
    }
    const auto at_a = checkDual(*s, kJa, home_a_, cfg_.home_tolerance_rad, "NOT_AT_HOME_A");
    const auto at_b = checkDual(*s, kJb, home_b_, cfg_.home_tolerance_rad, "NOT_AT_HOME_B");
    if (!motion)
    {
      RCLCPP_INFO(node_->get_logger(),
                  "dry-run: gripper commands=0 FollowJointTrajectory=0 move_group=not called");
      if (!at_a.ok || !at_b.ok)
        RCLCPP_INFO(node_->get_logger(),
                    "live is not at both Homes (A err=%.6f B err=%.6f). "
                    "DRY_RUN_ASSUMED_HOME for later chain checks; this is NOT MODEL_PLAN_PASS "
                    "and NOT REAL_HOME_EXECUTION_PASS",
                    at_a.max_error, at_b.max_error);
      if (home_a_.size() == 6)
        pred_a_ = home_a_;
      if (home_b_.size() == 6)
        pred_b_ = home_b_;
      pred_valid_ = true;
      RCLCPP_INFO(node_->get_logger(), "DRY_RUN_ASSUMED_HOME A=%s B=%s", fmt(pred_a_).c_str(),
                  fmt(pred_b_).c_str());
      return true;
    }
    if (servoj_block_a_ || servoj_block_b_)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "SERVOJ_RESTORE_UNCONFIRMED: refusing Current→Home after gripper/ServoJ fault; "
                   "not auto-resetting controllers");
      return false;
    }
    if (at_a.ok && at_b.ok)
    {
      RCLCPP_INFO(node_->get_logger(), "both arms already at Dual-8 Home; skip motion");
      pred_a_ = home_a_;
      pred_b_ = home_b_;
      pred_valid_ = true;
      home_plan_pass_ = true;
      return true;
    }
    if (!plan_home_)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "not at Dual-8 Home and plan_current_to_home=false; refusing");
      return false;
    }
    auto live_a = qa;
    auto live_b = qb;
    constexpr double kRrtHomeJointVmaxRadS = 0.2;
    if (!at_a.ok)
    {
      TrajectorySegmentRecord seg;
      std::string err;
      bool used_rrtconnect = false;
      RCLCPP_INFO(node_->get_logger(), "planning Arm A Current→Home with Arm B held at live");
      if (!planCurrentToHome("arm_a", kJa, live_a, live_b, home_a_, seg, err, used_rrtconnect))
      {
        RCLCPP_ERROR(node_->get_logger(), "Arm A Current→Home plan failed: %s", err.c_str());
        return false;
      }
      // RRTConnect：直接以最终目标 0.2 rad/s 分配时间，不再乘 0.2。
      // 直接路径：保留原本的轨迹时间和 trajectory_speed_scale。
      assignConservativeTimes(seg, used_rrtconnect ? kRrtHomeJointVmaxRadS : vmax_);
      const double home_scale = used_rrtconnect ? 1.0 : cfg_.trajectory_speed_scale;
      auto scaled = scaleSegment(seg, home_scale);
      if (!scaled.ok)
      {
        RCLCPP_ERROR(node_->get_logger(), "Arm A Home scale failed: %s", scaled.error.c_str());
        return false;
      }
      RCLCPP_INFO(node_->get_logger(),
                  "Current→Home arm_a method=%s home_scale=%.3f assign_vmax=%.3f "
                  "EFFECTIVE_SPEED_SCALE=%.3f",
                  used_rrtconnect ? "rrtconnect" : "direct", home_scale,
                  used_rrtconnect ? kRrtHomeJointVmaxRadS : vmax_, home_scale);
      if (!sendTraj("arm_a", scaled.segment))
        return false;
      if (!waitEnd("arm_a", kJa, home_a_))
        return false;
      live_a = home_a_;
    }
    s = snap();
    if (!s)
      return false;
    live_b = named(*s, kJb);
    const auto at_b2 = checkDual(*s, kJb, home_b_, cfg_.home_tolerance_rad, "NOT_AT_HOME_B");
    if (!at_b2.ok)
    {
      TrajectorySegmentRecord seg;
      std::string err;
      bool used_rrtconnect = false;
      RCLCPP_INFO(node_->get_logger(), "planning Arm B Current→Home with Arm A held at Home");
      if (!planCurrentToHome("arm_b", kJb, live_a, live_b, home_b_, seg, err, used_rrtconnect))
      {
        RCLCPP_ERROR(node_->get_logger(), "Arm B Current→Home plan failed: %s", err.c_str());
        return false;
      }
      // RRTConnect：直接以最终目标 0.2 rad/s 分配时间，不再乘 0.2。
      // 直接路径：保留原本的轨迹时间和 trajectory_speed_scale。
      assignConservativeTimes(seg, used_rrtconnect ? kRrtHomeJointVmaxRadS : vmax_);
      const double home_scale = used_rrtconnect ? 1.0 : cfg_.trajectory_speed_scale;
      auto scaled = scaleSegment(seg, home_scale);
      if (!scaled.ok)
      {
        RCLCPP_ERROR(node_->get_logger(), "Arm B Home scale failed: %s", scaled.error.c_str());
        return false;
      }
      RCLCPP_INFO(node_->get_logger(),
                  "Current→Home arm_b method=%s home_scale=%.3f assign_vmax=%.3f "
                  "EFFECTIVE_SPEED_SCALE=%.3f",
                  used_rrtconnect ? "rrtconnect" : "direct", home_scale,
                  used_rrtconnect ? kRrtHomeJointVmaxRadS : vmax_, home_scale);
      if (!sendTraj("arm_b", scaled.segment))
        return false;
      if (!waitEnd("arm_b", kJb, home_b_))
        return false;
    }
    RCLCPP_INFO(node_->get_logger(), "SEQUENTIAL HOME: PASS");
    pred_a_ = home_a_;
    pred_b_ = home_b_;
    pred_valid_ = true;
    home_plan_pass_ = true;
    return true;
  }

  bool planArmHome(const std::string& group, const std::vector<std::string>& names,
                   const std::vector<double>& live_a, const std::vector<double>& live_b,
                   const std::vector<double>& goal, TrajectorySegmentRecord& out, std::string& err)
  {
    if (!move_client_)
    {
      err = "move_action client missing";
      return false;
    }
    if (!move_client_->wait_for_action_server(std::chrono::seconds(5)))
    {
      err = "move_action unavailable";
      return false;
    }
    MoveGroup::Goal g;
    g.request.group_name = group;
    g.request.num_planning_attempts = 5;
    g.request.allowed_planning_time = 10.0;
    g.request.max_velocity_scaling_factor = home_plan_velocity_scale_;
    g.request.max_acceleration_scaling_factor = home_plan_velocity_scale_;
    g.request.start_state.is_diff = false;
    g.request.start_state.joint_state.name = kJa;
    g.request.start_state.joint_state.name.insert(g.request.start_state.joint_state.name.end(),
                                                  kJb.begin(), kJb.end());
    g.request.start_state.joint_state.position = live_a;
    g.request.start_state.joint_state.position.insert(
        g.request.start_state.joint_state.position.end(), live_b.begin(), live_b.end());
    moveit_msgs::msg::Constraints c;
    for (size_t i = 0; i < names.size() && i < goal.size(); ++i)
    {
      moveit_msgs::msg::JointConstraint jc;
      jc.joint_name = names[i];
      jc.position = goal[i];
      jc.tolerance_above = 0.001;
      jc.tolerance_below = 0.001;
      jc.weight = 1.0;
      c.joint_constraints.push_back(jc);
    }
    g.request.goal_constraints.push_back(c);
    g.planning_options.plan_only = true;
    g.planning_options.look_around = false;
    g.planning_options.replan = false;
    auto send = move_client_->async_send_goal(g);
    if (rclcpp::spin_until_future_complete(node_, send, std::chrono::seconds(15)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      err = "move_action goal send failed";
      return false;
    }
    auto handle = send.get();
    if (!handle)
    {
      err = "move_action rejected";
      return false;
    }
    auto wrapped = move_client_->async_get_result(handle);
    if (rclcpp::spin_until_future_complete(node_, wrapped, std::chrono::seconds(30)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      err = "move_action result timeout";
      return false;
    }
    auto wrapped_res = wrapped.get();
    if (wrapped_res.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped_res.result)
    {
      err = "move_action plan failed";
      return false;
    }
    const auto& jt = wrapped_res.result->planned_trajectory.joint_trajectory;
    if (jt.points.empty())
    {
      err = "empty planned trajectory";
      return false;
    }
    out.joint_names = names;
    out.points.clear();
    for (const auto& pt : jt.points)
    {
      fr_task_planner::TrajectoryPointRecord rec;
      rec.positions.assign(names.size(), 0.0);
      for (size_t i = 0; i < names.size(); ++i)
      {
        auto it = std::find(jt.joint_names.begin(), jt.joint_names.end(), names[i]);
        if (it == jt.joint_names.end())
          continue;
        const size_t idx = static_cast<size_t>(std::distance(jt.joint_names.begin(), it));
        if (idx < pt.positions.size())
          rec.positions[i] = pt.positions[idx];
      }
      out.points.push_back(rec);
    }
    out.start_joints = out.points.front().positions;
    out.end_joints = out.points.back().positions;
    out.deployable = true;
    out.logical_segment = group + "_current_to_home";
    return true;
  }

  bool waitEnd(const std::string& moving, const std::vector<std::string>& names,
               const std::vector<double>& expected)
  {
    const auto t0 = std::chrono::steady_clock::now();
    int ok_n = 0;
    while (rclcpp::ok())
    {
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (elapsed > cfg_.joint_settle_timeout_sec)
      {
        RCLCPP_ERROR(node_->get_logger(), "SEGMENT_END_SETTLE_TIMEOUT %s", moving.c_str());
        return false;
      }
      rclcpp::spin_some(node_);
      auto nows = snap();
      if (!nows)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      const auto chk =
          checkDual(*nows, names, expected, cfg_.segment_end_tolerance_rad, "SEGMENT_END_STATE_MISMATCH");
      if (chk.ok)
      {
        if (++ok_n >= std::max(1, cfg_.joint_settle_required_samples))
        {
          RCLCPP_INFO(node_->get_logger(), "%s live Home confirmed err=%.6f", moving.c_str(),
                      chk.max_error);
          return true;
        }
      }
      else
        ok_n = 0;
      std::this_thread::sleep_for(std::chrono::milliseconds(
          static_cast<int>(cfg_.joint_settle_poll_period_sec * 1000.0)));
    }
    return false;
  }

  bool sendTraj(const std::string& moving, const TrajectorySegmentRecord& seg)
  {
    if ((moving == "arm_b" ? servoj_block_b_ : servoj_block_a_))
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "SERVOJ_RESTORE_UNCONFIRMED: refusing trajectory on %s", moving.c_str());
      return false;
    }
    auto client = (moving == "arm_b") ? client_b_ : client_a_;
    const auto& action = (moving == "arm_b") ? action_b_ : action_a_;
    const auto& joints = (moving == "arm_b") ? kJb : kJa;
    TrajectorySegmentRecord remapped;
    std::string error;
    if (!remapSegmentByJointName(seg, joints, remapped, error))
    {
      RCLCPP_ERROR(node_->get_logger(), "%s", error.c_str());
      return false;
    }
    FollowJointTrajectory::Goal goal;
    goal.trajectory = toTraj(remapped);
    goal.trajectory.header.stamp = node_->get_clock()->now();
    const double timeout =
        actionTimeoutSec(segmentDuration(remapped), cfg_.timeout_factor, cfg_.timeout_margin_sec);
    RCLCPP_INFO(node_->get_logger(), "sending FollowJointTrajectory %s points=%zu timeout=%.1f",
                action.c_str(), remapped.points.size(), timeout);
    if (!client->wait_for_action_server(std::chrono::seconds(2)))
    {
      RCLCPP_ERROR(node_->get_logger(), "action unavailable %s", action.c_str());
      return false;
    }
    auto send = client->async_send_goal(goal);
    if (rclcpp::spin_until_future_complete(node_, send, std::chrono::seconds(15)) !=
        rclcpp::FutureReturnCode::SUCCESS)
      return false;
    auto handle = send.get();
    if (!handle)
      return false;
    ++motion_sent_;
    auto wrapped = client->async_get_result(handle);
    if (rclcpp::spin_until_future_complete(node_, wrapped, std::chrono::duration<double>(timeout)) !=
        rclcpp::FutureReturnCode::SUCCESS)
      return false;
    return wrapped.get().code == rclcpp_action::ResultCode::SUCCEEDED;
  }

  bool runGripperStage(const Stage& st, bool motion)
  {
    const bool arm_b = st.moving == "arm_b";
    const std::string& service = arm_b ? grip_b_ : grip_a_;
    cfg_.gripper_service_name = service;
    RCLCPP_INFO(node_->get_logger(),
                "GRIPPER STAGE id=%s arm=%s service=%s kind=%s "
                "service_available=%s (not ping, not activate, not motion-done)",
                st.id.c_str(), arm_b ? "B" : "A", service.c_str(), st.kind.c_str(),
                (arm_b ? live_gb_ : live_ga_) ? "YES" : "NO");
    if (motion && task_mode_ == "keypose_v1" &&
        empty_gripper_confirmation_ != "I_CONFIRM_BOTH_GRIPPERS_ARE_EMPTY")
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "STOP before gripper %s. Current→Home does not open grippers. KEYPOSE_V1 will "
                   "not assume the arms are empty. Relaunch with "
                   "empty_gripper_confirmation:=I_CONFIRM_BOTH_GRIPPERS_ARE_EMPTY after you "
                   "confirm both grippers hold nothing.",
                   st.id.c_str());
      return false;
    }
    if (inject_failure_ == "GRIPPER_FAIL")
    {
      RCLCPP_ERROR(node_->get_logger(), "inject_failure=GRIPPER_FAIL at %s", st.id.c_str());
      return false;
    }
    if (st.kind == "gripper_activate")
    {
      RCLCPP_INFO(node_->get_logger(),
                  "explicit ActGripper(id,1) only; reset/ActGripper(id,0) will NOT be sent");
      if (!motion)
      {
        RCLCPP_INFO(node_->get_logger(),
                    "DRY-RUN GRIPPER PING then ACTIVATE arm=%s SENT=NO", arm_b ? "B" : "A");
        return true;
      }
      if (!callGripper(arm_b, makeGripperPingRequest(cfg_), "GRIPPER_PING_FAILED", "ping"))
        return false;
      if (!callGripper(arm_b, makeActivateRequest(cfg_), "GRIPPER_ACTIVATE_FAILED", "activate"))
        return false;
      if (arm_b)
        activated_b_ = true;
      else
        activated_a_ = true;
      return true;
    }

    const bool close = st.kind == "gripper_close";
    const bool activated = arm_b ? (activated_b_ || already_act_b_) : (activated_a_ || already_act_a_);
    RCLCPP_INFO(node_->get_logger(),
                "gripper activated this process=%s operator_asserted=%s",
                (arm_b ? activated_b_ : activated_a_) ? "YES" : "NO",
                (arm_b ? already_act_b_ : already_act_a_) ? "YES" : "NO");
    if (!motion)
    {
      RCLCPP_INFO(node_->get_logger(), "DRY-RUN GRIPPER %s arm=%s service=%s SENT=NO",
                  close ? "CLOSE" : "OPEN", arm_b ? "B" : "A", service.c_str());
      if (close && !arm_b && grasp_post_close_wait_sec_ >= 0.0)
        RCLCPP_INFO(node_->get_logger(),
                    "DRY-RUN grasp_post_close_wait_sec=%.3f starts after close success, not in parallel",
                    grasp_post_close_wait_sec_);
      if (close && arm_b && handover_post_close_wait_sec_ >= 0.0)
        RCLCPP_INFO(node_->get_logger(),
                    "DRY-RUN handover_post_close_wait_sec=%.3f starts after B close success",
                    handover_post_close_wait_sec_);
      if (close && !arm_b)
        owner_logic_ = "A";
      if (!close && !arm_b)
        owner_logic_ = physical_b_grasp_ ? "B" : "A";
      if (st.kind == "gripper_open_event" && arm_b)
      {
        owner_logic_ = "world";
        RCLCPP_INFO(node_->get_logger(), "DRY-RUN PLACE RELEASE: world object + wait %.3f sec; retreat follows only on real open success", place_post_open_wait_sec_);
      }
      return true;
    }
    if (st.kind == "gripper_open" && !arm_b && !physical_b_grasp_ && st.id != "gripper_open_a_startup")
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "refusing gripper_open_a without runtime B grasp confirm; A keeps the part");
      return false;
    }
    if (close && !activated)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "refusing MoveGripper before activate on Arm %s. Run segment:=gripper_activate_%s "
                   "or pass gripper_%s_already_activated:=true only if ActGripper(id,1) already "
                   "succeeded. service-available is not activate.",
                   arm_b ? "B" : "A", arm_b ? "b" : "a", arm_b ? "b" : "a");
      return false;
    }
    auto fields = close ? makeGripperCloseRequest(cfg_) : makeGripperOpenRequest(cfg_);
    if (close && task_mode_ == "keypose_v1")
      fr_task_planner::applyKeyposeCloseEffort(fields.velocity, fields.force);
    if (!callGripper(arm_b, fields, close ? "REAL_GRIPPER_CLOSE_FAILED" : "REAL_GRIPPER_OPEN_FAILED",
                     "move"))
      return false;
    if (close && arm_b)
    {
      if (servoj_block_b_)
      {
        RCLCPP_ERROR(node_->get_logger(),
                     "gripper_close_b returned but ServoJ unrestored; STOP. "
                     "B_GRIPPER_CLOSE_SUCCESS not claimed. A will NOT open.");
        return false;
      }
      b_close_success_ = true;
      RCLCPP_INFO(node_->get_logger(),
                  "B_GRIPPER_CLOSE_SUCCESS. This is command completion, not physical hold.");
      if (task_mode_ == "keypose_v1" && handover_post_close_wait_sec_ >= 0.0)
      {
        RCLCPP_INFO(node_->get_logger(),
                    "B close finished; waiting handover_post_close_wait_sec=%.3f before confirmation. "
                    "physical_b_grasp stays false.",
                    handover_post_close_wait_sec_);
        rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(handover_post_close_wait_sec_)));
        handover_wait_done_ = true;
      }
    }
    if (close && !arm_b)
    {
      owner_logic_ = "A";
      if (grasp_post_close_wait_sec_ >= 0.0)
      {
        RCLCPP_INFO(node_->get_logger(),
                    "grasp close finished; waiting grasp_post_close_wait_sec=%.3f before lift",
                    grasp_post_close_wait_sec_);
        rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(grasp_post_close_wait_sec_)));
      }
    }
    if (!close && !arm_b && st.id == "gripper_open_a")
      handover_a_opened_ = true;
    if (!close && !arm_b && physical_b_grasp_)
      owner_logic_ = "B";
    if (st.kind == "gripper_open_event" && arm_b)
    {
      if (!releasePartAtPlace(st))
        return false;
      RCLCPP_INFO(node_->get_logger(), "PLACE open completed; waiting %.3f sec before retreat", place_post_open_wait_sec_);
      rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(place_post_open_wait_sec_)));
    }
    return true;
  }

  bool callGripper(bool arm_b, const GripperBridgeRequestFields& fields, const char* prefix,
                   const char* expected_command)
  {
    auto client = arm_b ? grip_client_b_ : grip_client_a_;
    const std::string& service = arm_b ? grip_b_ : grip_a_;
    const std::string arm = arm_b ? "B" : "A";
    RCLCPP_INFO(node_->get_logger(), "%s", formatGripperRequestLog(fields).c_str());
    RCLCPP_INFO(node_->get_logger(),
                "GRIPPER CALL arm=%s service=%s command=%s gripper_id=%d position=%d "
                "velocity=%d force=%d max_time_ms=%d",
                arm.c_str(), service.c_str(), fields.command.c_str(), fields.gripper_id,
                fields.position, fields.velocity, fields.force, fields.max_time_ms);

    const auto t_wait0 = std::chrono::steady_clock::now();
    const bool available = client->wait_for_service(std::chrono::seconds(2));
    const double wait_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait0).count();
    RCLCPP_INFO(node_->get_logger(), "GRIPPER service-available=%s wait_elapsed=%.3f",
                available ? "YES" : "NO", wait_elapsed);
    if (!available)
    {
      auto classified =
          classifyGripperCall(false, false, false, 0, "", wait_elapsed, prefix);
      markServo(arm_b, "UNCONFIRMED_SERVICE_UNAVAILABLE");
      RCLCPP_ERROR(node_->get_logger(), "%s",
                   formatGripperResponseLog(0, classified.error, wait_elapsed, false).c_str());
      return false;
    }

    auto req = std::make_shared<fairino_msgs::srv::GripperBridge::Request>();
    req->command = fields.command;
    req->gripper_id = fields.gripper_id;
    req->position = fields.position;
    req->velocity = fields.velocity;
    req->force = fields.force;
    req->max_time_ms = fields.max_time_ms;
    req->block = fields.block;
    req->gripper_type = fields.gripper_type;
    req->rot_num = fields.rot_num;
    req->rot_vel = fields.rot_vel;
    req->rot_torque = fields.rot_torque;

    const rclcpp::Time req_stamp = node_->now();
    const auto t_send = std::chrono::steady_clock::now();
    RCLCPP_INFO(node_->get_logger(), "GRIPPER request timestamp=%s",
                std::to_string(req_stamp.nanoseconds()).c_str());
    auto fut = client->async_send_request(req);
    ++gripper_sent_;
    const auto wait_status = rclcpp::spin_until_future_complete(
        node_, fut, std::chrono::duration<double>(cfg_.gripper_timeout_sec));
    const rclcpp::Time resp_stamp = node_->now();
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_send).count();

    if (wait_status != rclcpp::FutureReturnCode::SUCCESS)
    {
      auto classified = classifyGripperCall(true, true, false, 0, "", elapsed, prefix);
      const auto kind = classifyGripperCompletion(classified, fields);
      markServo(arm_b, "UNCONFIRMED_TIMEOUT");
      RCLCPP_ERROR(node_->get_logger(),
                   "GRIPPER RESPONSE arm=%s service=%s command=%s request_ts=%s response_ts=%s "
                   "elapsed_sec=%.3f error_code=n/a message=\"\" motion_completion=%s "
                   "servoj_restoration=UNCONFIRMED_TIMEOUT physical_grasp=NOT_THIS_CALL",
                   arm.c_str(), service.c_str(), fields.command.c_str(),
                   std::to_string(req_stamp.nanoseconds()).c_str(),
                   std::to_string(resp_stamp.nanoseconds()).c_str(), elapsed,
                   gripperCompletionName(kind).c_str());
      RCLCPP_ERROR(node_->get_logger(),
                   "hardware may still be in SERVO_RESTART; do not send the next trajectory or "
                   "retry MoveGripper as if this call finished");
      return false;
    }

    auto resp = fut.get();
    const bool null_resp = !resp;
    const int error_code = resp ? resp->error_code : 0;
    const std::string message = resp ? resp->message : "";
    auto classified =
        classifyGripperCall(true, false, null_resp, error_code, message, elapsed, prefix);
    const auto kind = classifyGripperCompletion(classified, fields);

    std::string servoj = "UNKNOWN";
    if (fields.command == "ping")
      servoj = "NOT_APPLICABLE_PING";
    else if (message.find("gripper command busy") != std::string::npos)
      servoj = "UNCONFIRMED_BUSY";
    else if (message.find("timed out waiting for motion done") != std::string::npos ||
             message.find("GetGripperMotionDone timeout") != std::string::npos)
      servoj = "UNCONFIRMED_TIMEOUT";
    else if (message.find("ServoMove") != std::string::npos)
      servoj = "UNCONFIRMED_SERVO_START";
    else if (classified.ok && error_code == 0 && fields.command != "ping")
      servoj = "CONFIRMED_BY_GRIPPER_BRIDGE";

    if (servoj.rfind("UNCONFIRMED", 0) == 0 || servoj == "CONFIRMED_BY_GRIPPER_BRIDGE")
      markServo(arm_b, servoj);

    RCLCPP_INFO(node_->get_logger(),
                "GRIPPER RESPONSE arm=%s service=%s command=%s gripper_id=%d position=%d "
                "velocity=%d force=%d request_ts=%s response_ts=%s elapsed_sec=%.3f "
                "error_code=%d message=\"%s\" motion_completion=%s servoj_restoration=%s "
                "physical_grasp=NOT_THIS_CALL",
                arm.c_str(), service.c_str(), fields.command.c_str(), fields.gripper_id,
                fields.position, fields.velocity, fields.force,
                std::to_string(req_stamp.nanoseconds()).c_str(),
                std::to_string(resp_stamp.nanoseconds()).c_str(), elapsed, error_code,
                message.c_str(), gripperCompletionName(kind).c_str(), servoj.c_str());

    if (!classified.ok)
    {
      RCLCPP_ERROR(node_->get_logger(), "%s", classified.error.c_str());
      return false;
    }
    if (std::string(expected_command) == "move")
    {
      if (!gripperBridgeMotionDone(kind))
      {
        RCLCPP_ERROR(node_->get_logger(), "%s",
                     formatGripperCompletionFailure(classified, kind, prefix).c_str());
        return false;
      }
    }
    else if (kind != GripperCompletionKind::CommandAccepted)
    {
      RCLCPP_ERROR(node_->get_logger(), "%s",
                   formatGripperCompletionFailure(classified, kind, prefix).c_str());
      return false;
    }
    return true;
  }

  void markServo(bool arm_b, const std::string& state)
  {
    const bool block = state.rfind("UNCONFIRMED", 0) == 0;
    if (arm_b)
      servoj_block_b_ = block;
    else
      servoj_block_a_ = block;
    RCLCPP_INFO(node_->get_logger(), "ServoJ restoration state Arm %s: %s", arm_b ? "B" : "A",
                state.c_str());
  }

  void initPredFromLive()
  {
    auto s = snap();
    if (!s || !hasNames(*s, kJa) || !hasNames(*s, kJb))
    {
      pred_valid_ = false;
      RCLCPP_WARN(node_->get_logger(),
                  "predicted joints not initialized from live; chain check uses Home after "
                  "DRY_RUN_ASSUMED_HOME only if Current→Home runs");
      return;
    }
    pred_a_ = named(*s, kJa);
    pred_b_ = named(*s, kJb);
    pred_valid_ = true;
    owner_logic_ = "world";
    RCLCPP_INFO(node_->get_logger(),
                "predicted joints initialized from LIVE named /joint_states (dry-run will "
                "advance these; execute still uses live feedback)");
    RCLCPP_INFO(node_->get_logger(), "init predicted A %s", fmt(pred_a_).c_str());
    RCLCPP_INFO(node_->get_logger(), "init predicted B %s", fmt(pred_b_).c_str());
  }

  void applyPredEnd(const Stage& st, const std::vector<double>& end)
  {
    if (end.size() != 6)
      return;
    if (st.moving == "arm_b")
      pred_b_ = end;
    else if (st.moving == "arm_a")
      pred_a_ = end;
  }

  bool reconstructPredUntil(const std::string& id, std::string& err)
  {
    RCLCPP_INFO(node_->get_logger(),
                "RESUME reconstruct: walking skipped stages to expected joints (NOT executing, "
                "NOT sending gripper). resume_object_owner=%s",
                resume_object_owner_.c_str());
    bool found = false;
    for (auto& st : stages_)
    {
      if (st.id == id)
      {
        found = true;
        break;
      }
      if (st.kind == "runtime_replan")
      {
        if (home_a_.size() == 6)
          pred_a_ = home_a_;
        if (home_b_.size() == 6)
          pred_b_ = home_b_;
        continue;
      }
      if (st.kind == "gripper_close" && st.moving == "arm_a")
        owner_logic_ = "A";
      if (st.kind == "confirm_b_grasp")
        continue;
      if (st.kind == "gripper_open" && st.moving == "arm_a")
        owner_logic_ = "B";
      if (st.kind == "attachment_transfer")
        owner_logic_ = "B";
      if (!st.seg.end_joints.empty())
        applyPredEnd(st, st.seg.end_joints);
    }
    if (!found)
    {
      err = "unknown resume_from " + id;
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "resume expected A %s", fmt(pred_a_).c_str());
    RCLCPP_INFO(node_->get_logger(), "resume expected B %s", fmt(pred_b_).c_str());
    RCLCPP_INFO(node_->get_logger(), "resume expected owner=%s", owner_logic_.c_str());
    if (id == "gripper_close_b" || id == "physical_grasp_confirm_b" || id == "gripper_open_a" ||
        id == "attachment_transfer")
    {
      if (resume_object_owner_ != "A")
      {
        err = "resume at " + id + " requires resume_object_owner:=A (part still on A); "
              "cannot infer hold from joints; will not auto-open/close";
        RCLCPP_ERROR(node_->get_logger(), "%s", err.c_str());
        return false;
      }
    }
    if (id == "b_plus_x" || id == "b_minus_x" || id == "b_plus_z" || id == "a_return_home")
    {
      if (resume_object_owner_ != "B")
      {
        err = "resume at " + id + " requires resume_object_owner:=B and physical_B_grasp already "
              "confirmed; not inferring hold from joints";
        RCLCPP_ERROR(node_->get_logger(), "%s", err.c_str());
        return false;
      }
      physical_b_grasp_ = true;
      owner_logic_ = "B";
    }
    return true;
  }

  bool resumeLiveMatchesPred()
  {
    auto s = snap();
    if (!s)
    {
      RCLCPP_ERROR(node_->get_logger(), "resume: no live joints");
      return false;
    }
    bool ok = true;
    if (pred_a_.size() == 6)
    {
      const auto ca = checkDual(*s, kJa, pred_a_, cfg_.segment_start_tolerance_rad,
                                "RESUME_LIVE_A_MISMATCH");
      if (!ca.ok)
      {
        RCLCPP_ERROR(node_->get_logger(), "%s live=%s expected=%s", ca.error.c_str(),
                     fmt(named(*s, kJa)).c_str(), fmt(pred_a_).c_str());
        ok = false;
      }
    }
    if (pred_b_.size() == 6)
    {
      const auto cb = checkDual(*s, kJb, pred_b_, cfg_.segment_start_tolerance_rad,
                                "RESUME_LIVE_B_MISMATCH");
      if (!cb.ok)
      {
        RCLCPP_ERROR(node_->get_logger(), "%s live=%s expected=%s", cb.error.c_str(),
                     fmt(named(*s, kJb)).c_str(), fmt(pred_b_).c_str());
        ok = false;
      }
    }
    if (!ok)
      RCLCPP_ERROR(node_->get_logger(),
                   "refusing resume_from; live joints do not match reconstructed stage start");
    return ok;
  }

  bool waitAutoHandoverRelease()
  {
    RCLCPP_INFO(node_->get_logger(), "AUTO_HANDOVER_RELEASE: ENABLED");
    if (!b_close_success_ || servoj_block_b_)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "STOP: auto handover requires B CLOSE SUCCESS and ServoJ restored. "
                   "gripper_close_b success=%s servoj_block_b=%s. ARM A will NOT open.",
                   b_close_success_ ? "YES" : "NO", servoj_block_b_ ? "YES" : "NO");
      return false;
    }
    if (handover_wait_done_)
    {
      RCLCPP_INFO(node_->get_logger(),
                  "handover wait already completed after B close; not treating close itself as hold");
      physical_b_grasp_ = true;
      RCLCPP_INFO(node_->get_logger(), "ARM_A_RELEASE_ALLOWED");
      return true;
    }
    RCLCPP_INFO(node_->get_logger(), "AUTO HANDOVER MODE");
    RCLCPP_INFO(node_->get_logger(), "Waiting %.3f sec before Arm A release",
                handover_release_delay_sec_);
    RCLCPP_INFO(node_->get_logger(), "HANDOVER_WAIT: %.3f sec", handover_release_delay_sec_);
    const auto t0 = std::chrono::steady_clock::now();
    while (rclcpp::ok())
    {
      rclcpp::spin_some(node_);
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (elapsed >= handover_release_delay_sec_)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!rclcpp::ok())
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "HANDOVER_WAIT cancelled; ARM_A_RELEASE_ALLOWED not claimed");
      return false;
    }
    RCLCPP_INFO(node_->get_logger(), "HANDOVER_WAIT_COMPLETE");
    physical_b_grasp_ = true;
    RCLCPP_INFO(node_->get_logger(), "ARM_A_RELEASE_ALLOWED");
    return true;
  }

  bool waitBGraspConfirm()
  {
    if (confirm_grasp_b_)
    {
      RCLCPP_WARN(node_->get_logger(),
                  "handover_b_grasp_confirmed:=true was set at launch. This is only valid when "
                  "resuming AFTER the operator already confirmed B holds the part. "
                  "Do not set this at full-task start.");
      physical_b_grasp_ = true;
      return true;
    }
    confirm_vote_.store(0);
    RCLCPP_INFO(node_->get_logger(),
                "WAITING for operator B-grasp confirm. A will NOT open until confirmed.");
    RCLCPP_INFO(node_->get_logger(),
                "Confirm:  ros2 service call /%s/confirm_b_grasp "
                "std_srvs/srv/SetBool \"{data: true}\"",
                node_->get_name());
    RCLCPP_INFO(node_->get_logger(),
                "Reject:   ros2 service call /%s/confirm_b_grasp "
                "std_srvs/srv/SetBool \"{data: false}\"",
                node_->get_name());
    const auto t0 = std::chrono::steady_clock::now();
    while (rclcpp::ok())
    {
      rclcpp::spin_some(node_);
      const int v = confirm_vote_.load();
      if (v > 0)
      {
        RCLCPP_INFO(node_->get_logger(), "operator confirmed B physically holds the part");
        return true;
      }
      if (v < 0)
      {
        RCLCPP_ERROR(node_->get_logger(), "operator rejected B grasp; A will NOT open");
        return false;
      }
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (elapsed > b_grasp_confirm_timeout_sec_)
      {
        RCLCPP_ERROR(node_->get_logger(), "B grasp confirm timeout; A will NOT open");
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    RCLCPP_ERROR(node_->get_logger(), "B grasp confirm cancelled; A will NOT open");
    return false;
  }

  void logOutcome(bool file_ok, bool chain_ok, bool model_ok, bool real_ok, const std::string& why)
  {
    RCLCPP_INFO(node_->get_logger(), "FILE_LOAD_%s", file_ok ? "PASS" : "FAIL");
    RCLCPP_INFO(node_->get_logger(), "TRAJECTORY_CHAIN_%s", chain_ok ? "PASS" : "FAIL");
    RCLCPP_INFO(node_->get_logger(), "MODEL_PLAN_%s", model_ok ? "PASS" : "FAIL");
    RCLCPP_INFO(node_->get_logger(), "REAL_EXECUTION_%s", real_ok ? "PASS" : "FAIL");
    if (real_ok)
      RCLCPP_INFO(node_->get_logger(), "FULL_TASK_COMPLETE");
    else
      RCLCPP_INFO(node_->get_logger(), "FULL_TASK_COMPLETE: NO (%s)",
                  why.empty() ? "not a real full-task success" : why.c_str());
    if (!real_ok)
      RCLCPP_INFO(node_->get_logger(), "DUAL-7 REAL EXECUTION: %s",
                  why.empty() ? "NOT COMPLETE" : why.c_str());
  }

  rclcpp::Node::SharedPtr node_;
  ExecutorConfig cfg_;
  bool auto_continue_ = false;
  bool continue_after_home_ = false;
  bool simultaneous_home_ = false;
  bool plan_home_ = false;
  double home_plan_velocity_scale_ = 0.2;
  std::string empty_gripper_confirmation_;
  std::string keypose_dir_;
  std::vector<double> expected_b_handover_;
  std::vector<fr_task_planner::TrajectoryPointRecord> a_exit_points_;
  bool handover_wait_done_ = false;
  bool handover_a_opened_ = false;
  bool confirm_approach_ = false;
  bool confirm_grasp_b_ = false;
  bool auto_handover_release_ = false;
  bool b_close_success_ = false;
  double handover_release_delay_sec_ = 2.0;
  double grasp_post_close_wait_sec_ = -1.0;
  double handover_post_close_wait_sec_ = -1.0;
  double place_post_open_wait_sec_ = 1.5;
  bool already_act_a_ = false;
  bool already_act_b_ = false;
  bool activated_a_ = false;
  bool activated_b_ = false;
  bool servoj_block_a_ = false;
  bool servoj_block_b_ = false;
  bool use_sim_time_ = false;
  bool mock_zero_ = false;
  bool have_js_ = false;
  bool live_js_ = false;
  bool live_a_ = false;
  bool live_b_ = false;
  bool live_ga_ = false;
  bool live_gb_ = false;
  bool pred_valid_ = false;
  bool file_load_ok_ = false;
  bool home_plan_pass_ = false;
  bool physical_b_grasp_ = false;
  int motion_sent_ = 0;
  int gripper_sent_ = 0;
  int hw_margin_ms_ = 4000;
  double vmax_ = 0.2;
  double hw_wait_sec_ = 9.0;
  double max_joint_age_sec_ = 2.0;
  double b_grasp_confirm_timeout_sec_ = 600.0;
  std::string segment_;
  std::string resume_from_;
  std::string task_mode_ = "home_only";
  std::string inject_failure_;
  std::string resume_object_owner_ = "unknown";
  std::string last_stage_id_;
  std::string owner_logic_ = "world";
  std::string step15_path_;
  std::string segs_path_;
  std::string home_yaml_;
  std::string b_home_to_pre_file_;
  std::string action_a_;
  std::string action_b_;
  std::string grip_a_;
  std::string grip_b_;
  std::string group_a_;
  std::vector<double> home_a_;
  std::vector<double> home_b_;
  std::vector<double> home_b_deg_;
  std::vector<double> pre_b_;
  std::vector<double> handover_a_;
  std::vector<double> handover_b_;
  std::vector<double> pred_a_;
  std::vector<double> pred_b_;
  std::vector<Stage> stages_;
  std::mutex mu_;
  std::atomic<int> confirm_vote_{0};
  JointSnapshot latest_;
  rclcpp::Time last_stamp_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr client_a_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr client_b_;
  rclcpp::Client<fairino_msgs::srv::GripperBridge>::SharedPtr grip_client_a_;
  rclcpp::Client<fairino_msgs::srv::GripperBridge>::SharedPtr grip_client_b_;
  rclcpp_action::Client<MoveGroup>::SharedPtr move_client_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr confirm_srv_;
};

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("dual7_real_task_executor", opt);
  Dual7RealExecutor exec(node);
  const int rc = exec.run();
  rclcpp::shutdown();
  return rc;
}
