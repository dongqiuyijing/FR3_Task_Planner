#include "fr_task_planner/winner_trajectory_io.hpp"

#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace fr_task_planner
{
namespace
{
constexpr int kYamlDigits = 17;

std::string yamlScalar(double value)
{
  if (std::isnan(value))
  {
    return ".nan";
  }
  if (std::isinf(value))
  {
    return value > 0.0 ? ".inf" : "-.inf";
  }
  std::ostringstream oss;
  oss << std::setprecision(kYamlDigits) << value;
  return oss.str();
}

std::string yamlQuote(const std::string& value)
{
  bool need = value.empty();
  for (char c : value)
  {
    if (c == ':' || c == '#' || c == '"' || c == '\'' || c == '{' || c == '}' || c == '[' ||
        c == ']' || c == ',' || c == '&' || c == '*' || c == '!' || c == '|' || c == '>' ||
        c == '%' || c == '@' || c == '`')
    {
      need = true;
      break;
    }
  }
  if (!need)
  {
    return value;
  }
  std::string out = "\"";
  for (char c : value)
  {
    if (c == '\\' || c == '"')
    {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

void writeDoubleList(std::ostream& os, const std::vector<double>& values)
{
  os << "[";
  for (size_t i = 0; i < values.size(); ++i)
  {
    if (i)
    {
      os << ", ";
    }
    os << yamlScalar(values[i]);
  }
  os << "]";
}

void writeStringList(std::ostream& os, const std::vector<std::string>& values)
{
  os << "[";
  for (size_t i = 0; i < values.size(); ++i)
  {
    if (i)
    {
      os << ", ";
    }
    os << yamlQuote(values[i]);
  }
  os << "]";
}

std::vector<double> sequenceToDoubles(const YAML::Node& node)
{
  std::vector<double> out;
  if (!node || !node.IsSequence())
  {
    return out;
  }
  out.reserve(node.size());
  for (const auto& item : node)
  {
    out.push_back(item.as<double>());
  }
  return out;
}

std::vector<std::string> sequenceToStrings(const YAML::Node& node)
{
  std::vector<std::string> out;
  if (!node || !node.IsSequence())
  {
    return out;
  }
  out.reserve(node.size());
  for (const auto& item : node)
  {
    out.push_back(item.as<std::string>());
  }
  return out;
}

std::map<std::string, double> doublesToJoints(const std::vector<double>& values)
{
  std::map<std::string, double> joints;
  const size_t n = std::min(values.size(), kArmJoints.size());
  for (size_t i = 0; i < n; ++i)
  {
    joints[kArmJoints[i]] = values[i];
  }
  return joints;
}

bool parseJointsNode(const YAML::Node& node, std::map<std::string, double>& joints, std::string& error)
{
  const auto values = sequenceToDoubles(node);
  if (values.size() < kArmJoints.size())
  {
    error = "joint array shorter than 6";
    return false;
  }
  joints = doublesToJoints(values);
  return true;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
{
  if (a.size() != b.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

const TrajectorySegmentRecord* firstLogical(const PersistedTrajectory& traj, const std::string& logical)
{
  for (const auto& seg : traj.segments)
  {
    if (seg.logical_segment == logical)
    {
      return &seg;
    }
  }
  return nullptr;
}

const TrajectorySegmentRecord* lastLogical(const PersistedTrajectory& traj, const std::string& logical)
{
  const TrajectorySegmentRecord* found = nullptr;
  for (const auto& seg : traj.segments)
  {
    if (seg.logical_segment == logical)
    {
      found = &seg;
    }
  }
  return found;
}

std::vector<double> segmentStartArm(const TrajectorySegmentRecord& seg)
{
  if (seg.start_joints.size() >= kArmJoints.size())
  {
    return extractArmPositions(kArmJoints, seg.start_joints);
  }
  if (!seg.points.empty())
  {
    return extractArmPositions(seg.joint_names, seg.points.front().positions);
  }
  return {};
}

std::vector<double> segmentEndArm(const TrajectorySegmentRecord& seg)
{
  if (seg.end_joints.size() >= kArmJoints.size())
  {
    return extractArmPositions(kArmJoints, seg.end_joints);
  }
  if (!seg.points.empty())
  {
    return extractArmPositions(seg.joint_names, seg.points.back().positions);
  }
  return {};
}
}  // namespace

std::string logicalSegmentFromStage(const std::string& stage_name)
{
  if (stage_name == "MoveTo Home")
  {
    return kLogicalCurrentToHome;
  }
  if (stage_name == "MoveTo PreGrasp")
  {
    return kLogicalHomeToPreGrasp;
  }
  if (stage_name == "MoveTo Grasp")
  {
    return kLogicalPreGraspToGrasp;
  }
  if (stage_name == "MoveTo Lift")
  {
    return kLogicalGraspToLift;
  }
  if (stage_name == "MoveTo ViewA")
  {
    return kLogicalLiftToA;
  }
  if (stage_name == "MoveTo ViewB")
  {
    return kLogicalAToB;
  }
  if (stage_name == "MoveTo ViewC")
  {
    return kLogicalBToC;
  }
  return {};
}

bool isFixedDeployableLogical(const std::string& logical)
{
  return logical == kLogicalHomeToPreGrasp || logical == kLogicalPreGraspToGrasp ||
         logical == kLogicalGraspToLift || logical == kLogicalLiftToA || logical == kLogicalAToB ||
         logical == kLogicalBToC;
}

std::vector<double> jointsToVec(const std::map<std::string, double>& joints)
{
  std::vector<double> values;
  values.reserve(kArmJoints.size());
  for (const auto& name : kArmJoints)
  {
    const auto it = joints.find(name);
    values.push_back(it == joints.end() ? std::numeric_limits<double>::quiet_NaN() : it->second);
  }
  return values;
}

std::map<std::string, double> vecToJoints(const std::vector<double>& values)
{
  return doublesToJoints(values);
}

std::vector<double> extractArmPositions(const std::vector<std::string>& names,
                                        const std::vector<double>& values)
{
  std::map<std::string, double> joints;
  const size_t n = std::min(names.size(), values.size());
  for (size_t i = 0; i < n; ++i)
  {
    joints[names[i]] = values[i];
  }
  return jointsToVec(joints);
}

double maxAbsError(const std::vector<double>& a, const std::vector<double>& b)
{
  return maxAbsDiff(a, b);
}

double maxAbsErrorJoints(const std::map<std::string, double>& a,
                         const std::map<std::string, double>& b)
{
  return maxAbsDiff(jointsToVec(a), jointsToVec(b));
}

double segmentDuration(const TrajectorySegmentRecord& seg)
{
  if (seg.points.empty())
  {
    return 0.0;
  }
  const auto& last = seg.points.back();
  return static_cast<double>(last.sec) + 1e-9 * static_cast<double>(last.nanosec);
}

double computePathLength(const std::vector<TrajectorySegmentRecord>& segments, bool deployable_only)
{
  double acc = 0.0;
  for (const auto& seg : segments)
  {
    if (deployable_only && !seg.deployable)
    {
      continue;
    }
    std::vector<double> prev;
    bool have = false;
    for (const auto& pt : seg.points)
    {
      const auto cur = extractArmPositions(seg.joint_names, pt.positions);
      bool finite = true;
      for (double v : cur)
      {
        finite = finite && std::isfinite(v);
      }
      if (!finite)
      {
        continue;
      }
      if (have)
      {
        double d = 0.0;
        for (size_t i = 0; i < cur.size() && i < prev.size(); ++i)
        {
          const double e = cur[i] - prev[i];
          d += e * e;
        }
        acc += std::sqrt(d);
      }
      prev = cur;
      have = true;
    }
  }
  return acc;
}

double computeDuration(const std::vector<TrajectorySegmentRecord>& segments, bool deployable_only)
{
  double total = 0.0;
  for (const auto& seg : segments)
  {
    if (deployable_only && !seg.deployable)
    {
      continue;
    }
    if (!seg.points.empty())
    {
      total += segmentDuration(seg);
    }
  }
  return total;
}

void fillDerivedTotals(PersistedTrajectory& traj)
{
  for (auto& seg : traj.segments)
  {
    seg.duration = segmentDuration(seg);
  }
  traj.persisted_full_plan_total_time = computeDuration(traj.segments, false);
  traj.persisted_fixed_task_total_duration = computeDuration(traj.segments, true);
  traj.persisted_total_joint_path_length = computePathLength(traj.segments, false);
  traj.persisted_fixed_task_joint_path_length = computePathLength(traj.segments, true);
}

void attachStandardEvents(PersistedTrajectory& traj)
{
  traj.events = {
    { kLogicalPreGraspToGrasp, "REAL_GRIPPER_CLOSE_REQUIRED", false },
    { kLogicalPreGraspToGrasp, "ATTACH_SMALL_PART_TO_TCP", false },
    { kLogicalGraspToLift, "RESTORE_PART_TABLE_COLLISION", false },
  };
}

bool loadWinnerYaml(const std::string& path, WinnerEndpoints& out, std::string& error)
{
  try
  {
    YAML::Node root = YAML::LoadFile(path);
    if (!root)
    {
      error = "empty winner YAML";
      return false;
    }
    out.source_path = path;
    out.task_version = root["task_version"] ? root["task_version"].as<std::string>() : "";
    if (!parseJointsNode(root["home_joints_rad"], out.home_rad, error))
    {
      error = "home_joints_rad: " + error;
      return false;
    }
    if (!parseJointsNode(root["a_joints_rad"], out.a_rad, error))
    {
      error = "a_joints_rad: " + error;
      return false;
    }
    if (!parseJointsNode(root["b_joints_rad"], out.b_rad, error))
    {
      error = "b_joints_rad: " + error;
      return false;
    }
    if (!parseJointsNode(root["c_joints_rad"], out.c_rad, error))
    {
      error = "c_joints_rad: " + error;
      return false;
    }
    out.a_roll_deg = root["a_roll_deg"] ? root["a_roll_deg"].as<double>() : 0.0;
    out.b_roll_deg = root["b_roll_deg"] ? root["b_roll_deg"].as<double>() : 0.0;
    out.c_roll_deg = root["c_roll_deg"] ? root["c_roll_deg"].as<double>() : 0.0;
    out.search_score_total_time =
        root["predicted_total_trajectory_time"] ? root["predicted_total_trajectory_time"].as<double>() :
                                                  0.0;
    out.search_score_joint_path_length =
        root["total_joint_path_length"] ? root["total_joint_path_length"].as<double>() : 0.0;
    return true;
  }
  catch (const std::exception& ex)
  {
    error = ex.what();
    return false;
  }
}

bool winnerMatchesFrozenStep12c(const WinnerEndpoints& winner, std::string& error, double tol)
{
  const auto home = jointsToVec(winner.home_rad);
  const auto a = jointsToVec(winner.a_rad);
  const auto b = jointsToVec(winner.b_rad);
  const auto c = jointsToVec(winner.c_rad);
  const double home_err = maxAbsDiff(home, kStep12cHomeRad);
  const double a_err = maxAbsDiff(a, kStep12cARad);
  const double b_err = maxAbsDiff(b, kStep12cBRad);
  const double c_err = maxAbsDiff(c, kStep12cCRad);
  if (home_err > tol || a_err > tol || b_err > tol || c_err > tol ||
      std::abs(winner.a_roll_deg + 90.0) > 1e-6 || std::abs(winner.b_roll_deg + 130.0) > 1e-6 ||
      std::abs(winner.c_roll_deg + 85.0) > 1e-6)
  {
    std::ostringstream oss;
    oss << "winner endpoints drifted home=" << home_err << " A=" << a_err << " B=" << b_err
        << " C=" << c_err << " rolls=" << winner.a_roll_deg << "," << winner.b_roll_deg << ","
        << winner.c_roll_deg;
    error = oss.str();
    return false;
  }
  return true;
}

bool writeTrajectoryYaml(const std::string& path, const PersistedTrajectory& traj, std::string& error)
{
  std::ofstream os(path);
  if (!os)
  {
    error = "failed to open " + path;
    return false;
  }
  os << "trajectory_format_version: " << traj.trajectory_format_version << "\n";
  os << "task_version: " << yamlQuote(traj.task_version) << "\n";
  os << "source_winner_file: " << yamlQuote(traj.source_winner_file) << "\n";
  os << "source_winner_path: " << yamlQuote(traj.source_winner_path) << "\n";
  os << "label: " << yamlQuote(traj.label) << "\n";
  os << "execution_performed: " << (traj.execution_performed ? "true" : "false") << "\n";
  os << "real_robot_validated: " << (traj.real_robot_validated ? "true" : "false") << "\n";
  os << "requires_runtime_current_to_home: "
     << (traj.requires_runtime_current_to_home ? "true" : "false") << "\n";
  os << "requires_real_gripper_close: " << (traj.requires_real_gripper_close ? "true" : "false")
     << "\n";
  os << "requires_real_retime_or_scaling: "
     << (traj.requires_real_retime_or_scaling ? "true" : "false") << "\n";
  os << "runtime_current_to_home: " << yamlQuote(traj.runtime_current_to_home) << "\n";
  os << "fixed_trajectory_start: " << yamlQuote(traj.fixed_trajectory_start) << "\n";
  os << "real_execution_speed_scaling: " << yamlQuote(traj.real_execution_speed_scaling) << "\n";
  os << "recommended_initial_real_validation_scaling: "
     << yamlScalar(traj.recommended_initial_real_validation_scaling) << "\n";
  os << "search_score_total_time: " << yamlScalar(traj.search_score_total_time) << "\n";
  os << "search_score_joint_path_length: " << yamlScalar(traj.search_score_joint_path_length) << "\n";
  os << "persisted_full_plan_total_time: " << yamlScalar(traj.persisted_full_plan_total_time) << "\n";
  os << "persisted_fixed_task_total_duration: "
     << yamlScalar(traj.persisted_fixed_task_total_duration) << "\n";
  os << "persisted_total_joint_path_length: " << yamlScalar(traj.persisted_total_joint_path_length)
     << "\n";
  os << "persisted_fixed_task_joint_path_length: "
     << yamlScalar(traj.persisted_fixed_task_joint_path_length) << "\n";
  os << "joint_names: ";
  writeStringList(os, traj.joint_names);
  os << "\n";
  os << "deployment:\n";
  os << "  fixed_start_state: Home\n";
  os << "  runtime_current_to_home: REPLAN_REQUIRED\n";
  os << "  real_robot_validated: false\n";
  os << "  execution_performed: false\n";
  os << "  real_gripper_close_required: true\n";
  os << "  planning_scene_attach_required: true\n";
  os << "  recommended_initial_velocity_scaling: 0.05\n";
  os << "  recommended_initial_acceleration_scaling: 0.05\n";
  os << "events:\n";
  for (const auto& ev : traj.events)
  {
    os << "  - after_segment: " << yamlQuote(ev.after_segment) << "\n";
    os << "    event: " << yamlQuote(ev.event) << "\n";
    os << "    executed_now: " << (ev.executed_now ? "true" : "false") << "\n";
  }
  os << "winner:\n";
  os << "  a_roll_deg: " << yamlScalar(traj.winner.a_roll_deg) << "\n";
  os << "  b_roll_deg: " << yamlScalar(traj.winner.b_roll_deg) << "\n";
  os << "  c_roll_deg: " << yamlScalar(traj.winner.c_roll_deg) << "\n";
  os << "  home_joints_rad: ";
  writeDoubleList(os, jointsToVec(traj.winner.home_rad));
  os << "\n";
  os << "  a_joints_rad: ";
  writeDoubleList(os, jointsToVec(traj.winner.a_rad));
  os << "\n";
  os << "  b_joints_rad: ";
  writeDoubleList(os, jointsToVec(traj.winner.b_rad));
  os << "\n";
  os << "  c_joints_rad: ";
  writeDoubleList(os, jointsToVec(traj.winner.c_rad));
  os << "\n";
  os << "segments:\n";
  for (const auto& seg : traj.segments)
  {
    os << "  - name: " << yamlQuote(seg.name) << "\n";
    os << "    planning_stage_name: " << yamlQuote(seg.planning_stage_name) << "\n";
    os << "    logical_segment: " << yamlQuote(seg.logical_segment) << "\n";
    os << "    subsegment_index: " << seg.subsegment_index << "\n";
    os << "    deployable: " << (seg.deployable ? "true" : "false") << "\n";
    os << "    runtime_replan_required: " << (seg.runtime_replan_required ? "true" : "false")
       << "\n";
    os << "    joint_names: ";
    writeStringList(os, seg.joint_names);
    os << "\n";
    os << "    start_joints: ";
    writeDoubleList(os, seg.start_joints);
    os << "\n";
    os << "    end_joints: ";
    writeDoubleList(os, seg.end_joints);
    os << "\n";
    os << "    duration: " << yamlScalar(seg.duration) << "\n";
    os << "    velocities_present: " << (seg.velocities_present ? "true" : "false") << "\n";
    os << "    accelerations_present: " << (seg.accelerations_present ? "true" : "false") << "\n";
    os << "    multi_dof_present: " << (seg.multi_dof_present ? "true" : "false") << "\n";
    os << "    points:\n";
    if (seg.points.empty())
    {
      os << "      []\n";
    }
    for (const auto& pt : seg.points)
    {
      os << "      - positions: ";
      writeDoubleList(os, pt.positions);
      os << "\n";
      os << "        velocities: ";
      writeDoubleList(os, pt.velocities);
      os << "\n";
      os << "        accelerations: ";
      writeDoubleList(os, pt.accelerations);
      os << "\n";
      os << "        time_from_start:\n";
      os << "          sec: " << pt.sec << "\n";
      os << "          nanosec: " << pt.nanosec << "\n";
    }
    if (seg.multi_dof_present)
    {
      os << "    multi_dof:\n";
      os << "      joint_names: ";
      writeStringList(os, seg.multi_dof.joint_names);
      os << "\n";
      os << "      points:\n";
      for (const auto& pt : seg.multi_dof.points)
      {
        os << "        - time_from_start:\n";
        os << "            sec: " << pt.sec << "\n";
        os << "            nanosec: " << pt.nanosec << "\n";
        os << "          transforms:\n";
        for (const auto& tf : pt.transforms)
        {
          os << "            - translation: ";
          writeDoubleList(os, tf.translation);
          os << "\n";
          os << "              rotation: ";
          writeDoubleList(os, tf.rotation_xyzw);
          os << "\n";
        }
      }
    }
  }
  os.flush();
  if (!os)
  {
    error = "failed while writing " + path;
    return false;
  }
  return true;
}

bool readTrajectoryYaml(const std::string& path, PersistedTrajectory& traj, std::string& error)
{
  try
  {
    YAML::Node root = YAML::LoadFile(path);
    if (!root)
    {
      error = "empty trajectory YAML";
      return false;
    }
    traj = PersistedTrajectory{};
    traj.trajectory_format_version =
        root["trajectory_format_version"] ? root["trajectory_format_version"].as<int>() : 0;
    traj.task_version = root["task_version"] ? root["task_version"].as<std::string>() : "";
    traj.source_winner_file =
        root["source_winner_file"] ? root["source_winner_file"].as<std::string>() : "";
    traj.source_winner_path =
        root["source_winner_path"] ? root["source_winner_path"].as<std::string>() : "";
    traj.label = root["label"] ? root["label"].as<std::string>() : "";
    traj.execution_performed =
        root["execution_performed"] && root["execution_performed"].as<bool>();
    traj.real_robot_validated =
        root["real_robot_validated"] && root["real_robot_validated"].as<bool>();
    traj.requires_runtime_current_to_home =
        !root["requires_runtime_current_to_home"] ||
        root["requires_runtime_current_to_home"].as<bool>();
    traj.requires_real_gripper_close =
        !root["requires_real_gripper_close"] || root["requires_real_gripper_close"].as<bool>();
    traj.requires_real_retime_or_scaling =
        !root["requires_real_retime_or_scaling"] ||
        root["requires_real_retime_or_scaling"].as<bool>();
    traj.runtime_current_to_home = root["runtime_current_to_home"] ?
                                       root["runtime_current_to_home"].as<std::string>() :
                                       "";
    traj.fixed_trajectory_start =
        root["fixed_trajectory_start"] ? root["fixed_trajectory_start"].as<std::string>() : "";
    traj.real_execution_speed_scaling =
        root["real_execution_speed_scaling"] ?
            root["real_execution_speed_scaling"].as<std::string>() :
            "";
    traj.recommended_initial_real_validation_scaling =
        root["recommended_initial_real_validation_scaling"] ?
            root["recommended_initial_real_validation_scaling"].as<double>() :
            0.05;
    traj.search_score_total_time =
        root["search_score_total_time"] ? root["search_score_total_time"].as<double>() : 0.0;
    traj.search_score_joint_path_length =
        root["search_score_joint_path_length"] ?
            root["search_score_joint_path_length"].as<double>() :
            0.0;
    traj.persisted_full_plan_total_time =
        root["persisted_full_plan_total_time"] ?
            root["persisted_full_plan_total_time"].as<double>() :
            0.0;
    traj.persisted_fixed_task_total_duration =
        root["persisted_fixed_task_total_duration"] ?
            root["persisted_fixed_task_total_duration"].as<double>() :
            0.0;
    traj.persisted_total_joint_path_length =
        root["persisted_total_joint_path_length"] ?
            root["persisted_total_joint_path_length"].as<double>() :
            0.0;
    traj.persisted_fixed_task_joint_path_length =
        root["persisted_fixed_task_joint_path_length"] ?
            root["persisted_fixed_task_joint_path_length"].as<double>() :
            0.0;
    traj.joint_names = sequenceToStrings(root["joint_names"]);
    if (root["events"] && root["events"].IsSequence())
    {
      for (const auto& ev : root["events"])
      {
        EventRecord rec;
        rec.after_segment = ev["after_segment"] ? ev["after_segment"].as<std::string>() : "";
        rec.event = ev["event"] ? ev["event"].as<std::string>() : "";
        rec.executed_now = ev["executed_now"] && ev["executed_now"].as<bool>();
        traj.events.push_back(rec);
      }
    }
    if (root["winner"])
    {
      const auto w = root["winner"];
      traj.winner.a_roll_deg = w["a_roll_deg"] ? w["a_roll_deg"].as<double>() : 0.0;
      traj.winner.b_roll_deg = w["b_roll_deg"] ? w["b_roll_deg"].as<double>() : 0.0;
      traj.winner.c_roll_deg = w["c_roll_deg"] ? w["c_roll_deg"].as<double>() : 0.0;
      parseJointsNode(w["home_joints_rad"], traj.winner.home_rad, error);
      parseJointsNode(w["a_joints_rad"], traj.winner.a_rad, error);
      parseJointsNode(w["b_joints_rad"], traj.winner.b_rad, error);
      parseJointsNode(w["c_joints_rad"], traj.winner.c_rad, error);
      error.clear();
    }
    if (!root["segments"] || !root["segments"].IsSequence())
    {
      error = "segments missing";
      return false;
    }
    for (const auto& s : root["segments"])
    {
      TrajectorySegmentRecord seg;
      seg.name = s["name"] ? s["name"].as<std::string>() : "";
      seg.planning_stage_name =
          s["planning_stage_name"] ? s["planning_stage_name"].as<std::string>() : "";
      seg.logical_segment = s["logical_segment"] ? s["logical_segment"].as<std::string>() : "";
      seg.subsegment_index = s["subsegment_index"] ? s["subsegment_index"].as<int>() : 0;
      seg.deployable = s["deployable"] && s["deployable"].as<bool>();
      seg.runtime_replan_required =
          s["runtime_replan_required"] && s["runtime_replan_required"].as<bool>();
      seg.joint_names = sequenceToStrings(s["joint_names"]);
      seg.start_joints = sequenceToDoubles(s["start_joints"]);
      seg.end_joints = sequenceToDoubles(s["end_joints"]);
      seg.duration = s["duration"] ? s["duration"].as<double>() : 0.0;
      seg.velocities_present = s["velocities_present"] && s["velocities_present"].as<bool>();
      seg.accelerations_present = s["accelerations_present"] && s["accelerations_present"].as<bool>();
      seg.multi_dof_present = s["multi_dof_present"] && s["multi_dof_present"].as<bool>();
      if (s["points"] && s["points"].IsSequence())
      {
        for (const auto& p : s["points"])
        {
          TrajectoryPointRecord pt;
          pt.positions = sequenceToDoubles(p["positions"]);
          pt.velocities = sequenceToDoubles(p["velocities"]);
          pt.accelerations = sequenceToDoubles(p["accelerations"]);
          if (p["time_from_start"])
          {
            pt.sec = p["time_from_start"]["sec"] ? p["time_from_start"]["sec"].as<int32_t>() : 0;
            pt.nanosec =
                p["time_from_start"]["nanosec"] ? p["time_from_start"]["nanosec"].as<uint32_t>() : 0;
          }
          seg.points.push_back(pt);
        }
      }
      if (seg.multi_dof_present && s["multi_dof"])
      {
        const auto md = s["multi_dof"];
        seg.multi_dof.joint_names = sequenceToStrings(md["joint_names"]);
        if (md["points"] && md["points"].IsSequence())
        {
          for (const auto& p : md["points"])
          {
            MultiDofPointRecord pt;
            if (p["time_from_start"])
            {
              pt.sec = p["time_from_start"]["sec"] ? p["time_from_start"]["sec"].as<int32_t>() : 0;
              pt.nanosec = p["time_from_start"]["nanosec"] ?
                               p["time_from_start"]["nanosec"].as<uint32_t>() :
                               0;
            }
            if (p["transforms"] && p["transforms"].IsSequence())
            {
              for (const auto& tf : p["transforms"])
              {
                MultiDofTransformRecord rec;
                rec.translation = sequenceToDoubles(tf["translation"]);
                rec.rotation_xyzw = sequenceToDoubles(tf["rotation"]);
                pt.transforms.push_back(rec);
              }
            }
            seg.multi_dof.points.push_back(pt);
          }
        }
      }
      traj.segments.push_back(seg);
    }
    return true;
  }
  catch (const std::exception& ex)
  {
    error = ex.what();
    return false;
  }
}

PersistValidation validateRoundTrip(const PersistedTrajectory& original,
                                    const PersistedTrajectory& loaded)
{
  PersistValidation v;
  if (original.joint_names != loaded.joint_names)
  {
    v.ok = false;
    v.joint_names_identical = false;
    v.reason = "joint_names mismatch";
  }
  if (original.segments.size() != loaded.segments.size())
  {
    v.ok = false;
    v.point_count_identical = false;
    v.reason = "segment count mismatch";
    return v;
  }
  for (size_t i = 0; i < original.segments.size(); ++i)
  {
    const auto& a = original.segments[i];
    const auto& b = loaded.segments[i];
    if (a.joint_names != b.joint_names)
    {
      v.ok = false;
      v.joint_names_identical = false;
      v.reason = "segment joint_names mismatch: " + a.name;
    }
    if (a.points.size() != b.points.size())
    {
      v.ok = false;
      v.point_count_identical = false;
      v.reason = "point count mismatch: " + a.name;
      continue;
    }
    if (maxAbsDiff(a.start_joints, b.start_joints) > 1e-12 ||
        maxAbsDiff(a.end_joints, b.end_joints) > 1e-12)
    {
      v.ok = false;
      v.reason = "start/end joints mismatch: " + a.name;
    }
    for (size_t k = 0; k < a.points.size(); ++k)
    {
      v.position_max_error =
          std::max(v.position_max_error, maxAbsDiff(a.points[k].positions, b.points[k].positions));
      v.velocity_max_error =
          std::max(v.velocity_max_error, maxAbsDiff(a.points[k].velocities, b.points[k].velocities));
      v.acceleration_max_error = std::max(
          v.acceleration_max_error, maxAbsDiff(a.points[k].accelerations, b.points[k].accelerations));
      if (a.points[k].sec != b.points[k].sec || a.points[k].nanosec != b.points[k].nanosec)
      {
        v.ok = false;
        v.time_identical = false;
        v.reason = "time_from_start mismatch: " + a.name;
      }
    }
  }
  if (v.position_max_error > 1e-12 || v.velocity_max_error > 1e-12 ||
      v.acceleration_max_error > 1e-12)
  {
    v.ok = false;
    if (v.reason == "ok")
    {
      v.reason = "numeric round-trip exceeds 1e-12";
    }
  }
  return v;
}

PersistValidation validateContinuity(const PersistedTrajectory& traj, double tol)
{
  PersistValidation v;
  const std::vector<std::string> chain = {
    kLogicalHomeToPreGrasp, kLogicalPreGraspToGrasp, kLogicalGraspToLift,
    kLogicalLiftToA,        kLogicalAToB,            kLogicalBToC
  };
  for (size_t i = 1; i < chain.size(); ++i)
  {
    const auto* prev = lastLogical(traj, chain[i - 1]);
    const auto* next = firstLogical(traj, chain[i]);
    const std::string key = std::string(chain[i - 1]) + " / " + chain[i];
    if (!prev || !next)
    {
      v.ok = false;
      v.reason = "missing logical segment for continuity: " + key;
      v.discontinuities[key] = std::numeric_limits<double>::infinity();
      continue;
    }
    const double err = maxAbsDiff(segmentEndArm(*prev), segmentStartArm(*next));
    v.discontinuities[key] = err;
    v.max_discontinuity = std::max(v.max_discontinuity, err);
    if (err > tol)
    {
      v.ok = false;
      v.reason = "discontinuity " + key;
    }
  }
  for (size_t i = 1; i < traj.segments.size(); ++i)
  {
    const auto& prev = traj.segments[i - 1];
    const auto& next = traj.segments[i];
    if (prev.logical_segment != next.logical_segment || !prev.deployable || !next.deployable)
    {
      continue;
    }
    const double err = maxAbsDiff(segmentEndArm(prev), segmentStartArm(next));
    v.max_discontinuity = std::max(v.max_discontinuity, err);
    if (err > tol)
    {
      v.ok = false;
      v.reason = "subsegment discontinuity: " + prev.name;
    }
  }
  return v;
}

PersistValidation validateEndpoints(const PersistedTrajectory& traj, const WinnerEndpoints& winner,
                                    double tol)
{
  PersistValidation v;
  const auto* home_seg = firstLogical(traj, kLogicalHomeToPreGrasp);
  const auto* lift_a = lastLogical(traj, kLogicalLiftToA);
  const auto* a_b = firstLogical(traj, kLogicalAToB);
  const auto* a_b_end = lastLogical(traj, kLogicalAToB);
  const auto* b_c = firstLogical(traj, kLogicalBToC);
  const auto* b_c_end = lastLogical(traj, kLogicalBToC);
  if (!home_seg || !lift_a || !a_b || !a_b_end || !b_c || !b_c_end)
  {
    v.ok = false;
    v.reason = "required logical segments missing";
    return v;
  }
  v.home_start_error = maxAbsDiff(segmentStartArm(*home_seg), jointsToVec(winner.home_rad));
  v.a_max_error = std::max(maxAbsDiff(segmentEndArm(*lift_a), jointsToVec(winner.a_rad)),
                           maxAbsDiff(segmentStartArm(*a_b), jointsToVec(winner.a_rad)));
  v.b_max_error = std::max(maxAbsDiff(segmentEndArm(*a_b_end), jointsToVec(winner.b_rad)),
                           maxAbsDiff(segmentStartArm(*b_c), jointsToVec(winner.b_rad)));
  v.c_max_error = maxAbsDiff(segmentEndArm(*b_c_end), jointsToVec(winner.c_rad));
  if (v.home_start_error > tol || v.a_max_error > tol || v.b_max_error > tol ||
      v.c_max_error > tol)
  {
    v.ok = false;
    v.reason = "endpoint joint mismatch";
  }
  return v;
}

bool timesMonotonic(const TrajectorySegmentRecord& seg)
{
  int64_t prev = -1;
  for (const auto& pt : seg.points)
  {
    const int64_t t = static_cast<int64_t>(pt.sec) * 1000000000LL + static_cast<int64_t>(pt.nanosec);
    if (t < 0 || t < prev)
    {
      return false;
    }
    prev = t;
  }
  return !seg.points.empty() && segmentDuration(seg) > 0.0;
}
}  // namespace fr_task_planner
