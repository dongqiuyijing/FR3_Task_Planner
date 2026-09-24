#pragma once

#include <string>
#include <vector>

namespace fr_task_planner
{
struct KeyposeExitGate
{
  bool b_at_handover = false;
  bool feedback_fresh = false;
  bool b_close_finished = false;
  bool handover_confirmed = false;
  bool a_open = false;
  bool part_on_b = false;
  bool part_on_a = false;
  bool exit_path_clear = false;
};

inline void applyKeyposeCloseEffort(int& velocity, int& force)
{
  velocity = 80;
  force = 50;
}

inline std::string keyposeExitGateError(const KeyposeExitGate& gate)
{
  if (!gate.feedback_fresh)
    return "joint feedback stale";
  if (!gate.b_at_handover)
    return "B is not at B_HANDOVER";
  if (!gate.b_close_finished)
    return "B close command has not finished";
  if (!gate.handover_confirmed)
    return "handover confirmation has not passed";
  if (!gate.a_open)
    return "A gripper is not open";
  if (gate.part_on_a)
    return "part is still attached to A";
  if (!gate.part_on_b)
    return "part is not attached to B";
  if (!gate.exit_path_clear)
    return "exit trajectory is not clear in the dual-arm scene";
  return {};
}

inline std::vector<std::string> keyposeRuntimeStageIds(const std::vector<std::string>& file_ids)
{
  std::vector<std::string> out;
  out.reserve(file_ids.size() + 2);
  out.push_back("dual_current_to_home");
  for (const auto& id : file_ids)
  {
    out.push_back(id);
    if (id == "gripper_close_b")
      out.push_back("physical_grasp_confirm_b");
  }
  return out;
}
}  // namespace fr_task_planner
