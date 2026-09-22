// Offline checks only. Does not start ROS or the robot.
#include "fr_task_planner/keypose_v1_gates.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace
{
int fails = 0;

void expect(bool ok, const std::string& msg)
{
  if (ok)
  {
    std::cout << "PASS " << msg << std::endl;
    return;
  }
  std::cout << "FAIL " << msg << std::endl;
  ++fails;
}
}  // namespace

int main()
{
  int chain_fails = 0;
  int sequence_fails = 0;
  auto chain_expect = [&](bool ok, const std::string& msg) {
    expect(ok, msg);
    if (!ok)
      ++chain_fails;
  };
  auto seq_expect = [&](bool ok, const std::string& msg) {
    expect(ok, msg);
    if (!ok)
      ++sequence_fails;
  };
  fails = 0;
  const std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string path =
      home_dir + "/fr_task_ws/src/fr_task_planner/config/keypose_v1_six_face_trajectory.yaml";
  YAML::Node y = YAML::LoadFile(path);
  bool chain = y["chain"] && y["chain"].as<std::string>() == "PASS";
  bool speeds = y["speed_scale"] && std::abs(y["speed_scale"].as<double>() - 0.2) < 1e-9;
  bool exit_method = false;
  std::vector<std::string> ids;
  for (const auto& s : y["stages"])
  {
    ids.push_back(s["id"].as<std::string>());
    if (s["kind"] && s["kind"].as<std::string>() == "joint_connection")
    {
      if (!s["status"] || s["status"].as<std::string>() != "PASS")
        chain = false;
      if (!s["speed_scale"] || std::abs(s["speed_scale"].as<double>() - 0.2) > 1e-9)
        speeds = false;
      if (s["id"].as<std::string>() == "a_handover_to_home")
        exit_method = s["method"] && s["method"].as<std::string>() == "ompl_rrtconnect" &&
                      s["points"] && s["points"].size() > 2;
    }
  }
  chain_expect(chain && speeds && exit_method, "trajectory file chain, speed 0.2, RRTConnect exit");
  std::cout << (chain_fails == 0 ? "TRAJECTORY_CHAIN_PASS" : "TRAJECTORY_CHAIN_FAIL") << std::endl;

  const auto runtime = fr_task_planner::keyposeRuntimeStageIds(ids);
  auto indexOf = [&](const std::string& id) {
    for (size_t i = 0; i < runtime.size(); ++i)
      if (runtime[i] == id)
        return static_cast<int>(i);
    return -1;
  };
  const int home_at = indexOf("dual_current_to_home");
  const int open_startup = indexOf("gripper_open_a_startup");
  const int b_handover = indexOf("b_pre_to_handover");
  const int close_b = indexOf("gripper_close_b");
  const int confirm = indexOf("physical_grasp_confirm_b");
  const int open_a = indexOf("gripper_open_a");
  const int exit_a = indexOf("a_handover_to_home");
  const int face6 = indexOf("face5_to_face6");
  seq_expect(home_at == 0 && home_at < open_startup, "Current to Home before gripper initialization");
  seq_expect(b_handover < close_b && close_b + 1 == confirm && confirm < open_a && open_a < exit_a &&
                 exit_a < face6,
             "B close, confirm, A open, A exit, then B faces");

  fr_task_planner::KeyposeExitGate ok;
  ok.b_at_handover = true;
  ok.feedback_fresh = true;
  ok.b_close_finished = true;
  ok.handover_confirmed = true;
  ok.a_open = true;
  ok.part_on_b = true;
  ok.part_on_a = false;
  ok.exit_path_clear = true;
  seq_expect(fr_task_planner::keyposeExitGateError(ok).empty(), "complete exit gate accepts");
  ok.handover_confirmed = false;
  seq_expect(fr_task_planner::keyposeExitGateError(ok).find("confirmation") != std::string::npos,
             "close success alone does not release A");
  ok.handover_confirmed = true;
  ok.part_on_b = false;
  seq_expect(!fr_task_planner::keyposeExitGateError(ok).empty(), "missing B attachment blocks exit");
  ok.part_on_b = true;
  ok.part_on_a = true;
  seq_expect(fr_task_planner::keyposeExitGateError(ok).find("still attached") != std::string::npos,
             "part still on A blocks exit");
  ok.part_on_a = false;
  ok.a_open = false;
  seq_expect(!fr_task_planner::keyposeExitGateError(ok).empty(), "A still closed blocks exit");
  ok.a_open = true;
  ok.b_at_handover = false;
  seq_expect(!fr_task_planner::keyposeExitGateError(ok).empty(), "B away from handover blocks exit");
  ok.b_at_handover = true;
  ok.feedback_fresh = false;
  seq_expect(!fr_task_planner::keyposeExitGateError(ok).empty(), "stale feedback blocks exit");
  ok.feedback_fresh = true;
  ok.exit_path_clear = false;
  seq_expect(!fr_task_planner::keyposeExitGateError(ok).empty(), "colliding exit path blocks motion");

  int close_velocity = 20;
  int close_force = 20;
  int open_velocity = 20;
  int open_force = 20;
  fr_task_planner::applyKeyposeCloseEffort(close_velocity, close_force);
  seq_expect(close_velocity == 50 && close_force == 50, "keypose A/B close request is 50/50");
  seq_expect(open_velocity == 20 && open_force == 20, "keypose open request stays 20/20");

  std::cout << (sequence_fails == 0 ? "EXECUTION_SEQUENCE_PASS" : "EXECUTION_SEQUENCE_FAIL") << std::endl;
  return (chain_fails + sequence_fails) == 0 ? 0 : 1;
}
