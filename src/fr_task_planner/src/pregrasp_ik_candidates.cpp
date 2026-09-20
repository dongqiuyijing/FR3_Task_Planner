#include "fr_task_planner/pregrasp_ik_candidates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/robot_state/robot_state.h>
#include <rclcpp/rclcpp.hpp>

namespace fr_task_planner
{
namespace
{
double boundedDisplacement(double from, double to)
{
  return std::abs(to - from);
}

void classifyPreGraspContacts(const CollisionSnapshot& snap, PreGraspIkCandidate& cand,
                              IkValidationMode mode)
{
  cand.self_collision_ok = true;
  cand.table_collision_ok = true;
  cand.column_collision_ok = true;
  bool part_robot = false;
  bool part_non_touch = false;
  for (const auto& rec : snap.contacts)
  {
    if (rec.category == CollisionCategory::ROBOT_SELF)
    {
      cand.self_collision_ok = false;
    }
    else if (rec.category == CollisionCategory::ROBOT_TABLE)
    {
      cand.table_collision_ok = false;
    }
    else if (rec.category == CollisionCategory::ROBOT_COLUMN)
    {
      cand.column_collision_ok = false;
    }
    else if (rec.category == CollisionCategory::PART_NON_TOUCH_ROBOT)
    {
      part_non_touch = true;
      part_robot = true;
    }
    else if (rec.category == CollisionCategory::PART_TOUCH_ROBOT)
    {
      part_robot = true;
    }
  }
  const bool part_illegal =
      (mode == IkValidationMode::AttachedInspection) ? part_non_touch : part_robot;
  cand.collision_free = !snap.collision && cand.self_collision_ok && cand.table_collision_ok &&
                        cand.column_collision_ok && !part_illegal;
}

bool tryIk(moveit::core::RobotState& state, const moveit::core::JointModelGroup* jmg,
           const Eigen::Isometry3d& model_tcp, const std::string& ee_link, double timeout)
{
  return state.setFromIK(jmg, model_tcp, ee_link, timeout);
}
}  // namespace

double shortestValidJointDisplacement(double from, double to, double lower, double upper)
{
  double best = boundedDisplacement(from, to);
  for (int k : { -1, 1 })
  {
    const double alt = to + static_cast<double>(k) * 2.0 * M_PI;
    if (alt >= lower - 1e-9 && alt <= upper + 1e-9)
    {
      best = std::min(best, boundedDisplacement(from, alt));
    }
  }
  return best;
}

bool isSameIkBranch(const std::map<std::string, double>& a, const std::map<std::string, double>& b,
                    double min_distance)
{
  return jointL2(a, b) < min_distance;
}

int findDuplicateIndex(const std::vector<PreGraspIkCandidate>& cands,
                       const std::map<std::string, double>& joints, double min_distance)
{
  for (size_t i = 0; i < cands.size(); ++i)
  {
    if (isSameIkBranch(cands[i].joints, joints, min_distance))
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

double jointLimitMargin(const moveit::core::RobotState& state,
                        const moveit::core::JointModelGroup* jmg)
{
  if (!jmg)
  {
    return 0.0;
  }
  double margin = std::numeric_limits<double>::infinity();
  for (const auto* joint : jmg->getActiveJointModels())
  {
    const auto& bounds = joint->getVariableBounds();
    if (bounds.empty() || !bounds.front().position_bounded_)
    {
      continue;
    }
    const double q = state.getVariablePosition(joint->getName());
    margin = std::min(margin, q - bounds.front().min_position_);
    margin = std::min(margin, bounds.front().max_position_ - q);
  }
  return std::isfinite(margin) ? margin : 0.0;
}

void validatePreGraspCandidate(PreGraspIkCandidate& cand,
                               const planning_scene::PlanningScene& pregrasp_scene,
                               const geometry_msgs::msg::PoseStamped& pregrasp_pose,
                               const std::map<std::string, double>& home,
                               const PreGraspIkConfig& cfg)
{
  cand.valid = false;
  const auto robot_model = pregrasp_scene.getRobotModel();
  auto* jmg = robot_model->getJointModelGroup(cfg.group);
  moveit::core::RobotState fk(robot_model);
  fk.setToDefaultValues();
  applyJoints(fk, cand.joints);
  cand.bounds_ok = static_cast<bool>(jmg) && fk.satisfiesBounds(jmg);
  cand.joint_limit_margin = jointLimitMargin(fk, jmg);
  poseError(poseToIso(pregrasp_pose.pose), tcpInBase(fk, cfg.ee_link), cand.fk_position_error,
            cand.fk_orientation_error_deg);
  cand.fk_ok =
      cand.fk_position_error <= cfg.pos_tol && cand.fk_orientation_error_deg <= cfg.ori_tol_deg;
  cand.distance_from_home = jointL2(cand.joints, home);
  cand.joint_l1_from_home = jointL1(cand.joints, home);
  cand.j1_abs_displacement = std::abs(cand.joints.at("j1") - home.at("j1"));
  double j1_lower = -3.0543;
  double j1_upper = 3.0543;
  if (jmg)
  {
    const auto* j1 = robot_model->getJointModel("j1");
    if (j1 && !j1->getVariableBounds().empty() && j1->getVariableBounds().front().position_bounded_)
    {
      j1_lower = j1->getVariableBounds().front().min_position_;
      j1_upper = j1->getVariableBounds().front().max_position_;
    }
  }
  cand.j1_shortest_valid_displacement =
      shortestValidJointDisplacement(home.at("j1"), cand.joints.at("j1"), j1_lower, j1_upper);

  auto diag = cloneDiagnosticScene(pregrasp_scene);
  applyJointsToScene(*diag, cand.joints);
  cand.attached_object_present = diag->getCurrentState().hasAttachedBody(cfg.object_id);
  CollisionDiagConfig dcfg;
  dcfg.object_id = cfg.object_id;
  dcfg.table_name = cfg.table_name;
  dcfg.column_name = cfg.column_name;
  dcfg.touch_links = cfg.touch_links;
  const auto snap = collectCollisionContacts(*diag, dcfg);
  classifyPreGraspContacts(snap, cand, cfg.mode);
  cand.column_clearance = 0.0;
  collision_detection::DistanceRequest req;
  req.enable_signed_distance = true;
  req.type = collision_detection::DistanceRequestTypes::ALL;
  req.group_name = cfg.group;
  req.enableGroup(diag->getRobotModel());
  req.acm = &diag->getAllowedCollisionMatrix();
  collision_detection::DistanceResult dres;
  try
  {
    diag->getCollisionEnv()->distanceRobot(req, dres, diag->getCurrentState());
    double best = std::numeric_limits<double>::infinity();
    bool any = false;
    for (const auto& item : dres.distances)
    {
      if (item.first.first != cfg.column_name && item.first.second != cfg.column_name)
      {
        continue;
      }
      for (const auto& rec : item.second)
      {
        if (!std::isfinite(rec.distance))
        {
          continue;
        }
        any = true;
        best = std::min(best, rec.distance);
      }
    }
    if (any)
    {
      cand.column_clearance = best;
    }
  }
  catch (const std::exception&)
  {
  }

  if (cfg.mode == IkValidationMode::DetachedPreGrasp && cand.attached_object_present)
  {
    cand.failure_reason = "object attached at pregrasp";
  }
  else if (cfg.mode == IkValidationMode::AttachedInspection && !cand.attached_object_present)
  {
    cand.failure_reason = "object not attached at inspection";
  }
  else if (!cand.bounds_ok)
  {
    cand.failure_reason = "joint bounds";
  }
  else if (!cand.fk_ok)
  {
    cand.failure_reason = "fk pose mismatch";
  }
  else if (!cand.collision_free)
  {
    cand.failure_reason = snap.collision ? "collision" : "illegal contact";
  }
  else
  {
    cand.valid = true;
    cand.failure_reason = "ok";
  }
}

PreGraspIkGenerationResult generatePreGraspIkCandidates(
    const planning_scene::PlanningScene& pregrasp_scene,
    const geometry_msgs::msg::PoseStamped& pregrasp_pose, const std::map<std::string, double>& home,
    const std::map<std::string, double>& baseline_pregrasp,
    const std::vector<NamedJointSeed>& extra_seeds, const PreGraspIkConfig& cfg,
    const rclcpp::Logger& logger)
{
  PreGraspIkGenerationResult result;
  const auto robot_model = pregrasp_scene.getRobotModel();
  auto* jmg = robot_model->getJointModelGroup(cfg.group);
  if (!jmg)
  {
    return result;
  }
  const Eigen::Isometry3d t_model_base =
      pregrasp_scene.getCurrentState().getGlobalLinkTransform("base_link");
  const Eigen::Isometry3d t_model_tcp = t_model_base * poseToIso(pregrasp_pose.pose);

  auto push_unique = [&](PreGraspIkCandidate cand) {
    result.raw.push_back(cand);
    if (cand.fk_ok)
    {
      ++result.fk_valid;
    }
    if (cand.bounds_ok)
    {
      ++result.bounds_valid;
    }
    if (cand.collision_free)
    {
      ++result.collision_valid;
    }
    const int dup = findDuplicateIndex(result.unique, cand.joints, cfg.min_ik_solution_distance);
    if (dup >= 0)
    {
      if (cand.is_baseline)
      {
        result.unique[static_cast<size_t>(dup)].is_baseline = true;
        result.unique[static_cast<size_t>(dup)].candidate_id = cand.candidate_id;
        result.unique[static_cast<size_t>(dup)].seed_name = cand.seed_name;
      }
      return;
    }
    ++result.unique_raw;
    if (cand.valid || cand.is_baseline)
    {
      if (cand.valid)
      {
        ++result.unique_valid;
      }
      result.unique.push_back(std::move(cand));
    }
  };

  auto make_from_joints = [&](const std::map<std::string, double>& q, const std::string& seed_name,
                              bool baseline) {
    PreGraspIkCandidate cand;
    cand.seed_name = seed_name;
    cand.joints = q;
    cand.is_baseline = baseline;
    validatePreGraspCandidate(cand, pregrasp_scene, pregrasp_pose, home, cfg);
    cand.candidate_id = baseline ? "baseline_frozen_pregrasp" :
                                   ("ik_" + seed_name + "_" + std::to_string(result.unique.size()));
    push_unique(std::move(cand));
  };

  make_from_joints(baseline_pregrasp, "baseline", true);
  result.baseline_included = findDuplicateIndex(result.unique, baseline_pregrasp, 1e-9) >= 0 ||
                             findDuplicateIndex(result.raw, baseline_pregrasp, 1e-9) >= 0;
  if (!result.baseline_included)
  {
    PreGraspIkCandidate cand;
    cand.candidate_id = "baseline_frozen_pregrasp";
    cand.seed_name = "baseline";
    cand.joints = baseline_pregrasp;
    cand.is_baseline = true;
    validatePreGraspCandidate(cand, pregrasp_scene, pregrasp_pose, home, cfg);
    result.unique.insert(result.unique.begin(), cand);
    result.baseline_included = true;
  }

  std::vector<NamedJointSeed> seeds;
  seeds.push_back({ "home", home });
  seeds.push_back({ "baseline", baseline_pregrasp });
  seeds.insert(seeds.end(), extra_seeds.begin(), extra_seeds.end());

  auto seed_state = [&](const NamedJointSeed& seed) {
    moveit::core::RobotState state(pregrasp_scene.getCurrentState());
    applyJoints(state, seed.joints);
    return state;
  };

  const uint32_t max_unique = std::max<uint32_t>(cfg.max_unique_candidates, 1);
  auto remaining = [&]() { return result.unique.size() < max_unique; };

  for (const auto& seed : seeds)
  {
    if (!remaining() || result.attempts >= static_cast<int>(cfg.max_ik_attempts))
    {
      break;
    }
    make_from_joints(seed.joints, seed.name + "_rawfk", false);
    moveit::core::RobotState exact = seed_state(seed);
    ++result.attempts;
    if (tryIk(exact, jmg, t_model_tcp, cfg.ee_link, 0.25))
    {
      ++result.ik_success;
      make_from_joints(jointsFromState(exact), seed.name + "_exact", false);
    }
    const int nearby_n = (seed.name == "home") ? 8 : 4;
    for (int i = 0; i < nearby_n && remaining() &&
                    result.attempts < static_cast<int>(cfg.max_ik_attempts);
         ++i)
    {
      moveit::core::RobotState nearby = seed_state(seed);
      nearby.setToRandomPositionsNearBy(jmg, seed_state(seed),
                                        i < nearby_n / 2 ? cfg.nearby_radius_local :
                                                           cfg.nearby_radius);
      ++result.attempts;
      if (!tryIk(nearby, jmg, t_model_tcp, cfg.ee_link, 0.05))
      {
        continue;
      }
      ++result.ik_success;
      make_from_joints(jointsFromState(nearby), seed.name + "_nearby", false);
    }
  }

  while (remaining() && result.attempts < static_cast<int>(cfg.max_ik_attempts))
  {
    moveit::core::RobotState random(pregrasp_scene.getCurrentState());
    random.setToRandomPositions(jmg);
    if ((result.attempts % 3) == 0)
    {
      random.setVariablePosition("j1", home.at("j1"));
      random.update();
    }
    else if ((result.attempts % 3) == 1)
    {
      const double span = 1.2;
      const double j1 =
          home.at("j1") + span * (0.5 - static_cast<double>(result.attempts % 11) / 10.0);
      random.setVariablePosition("j1", j1);
      random.update();
    }
    ++result.attempts;
    if (!tryIk(random, jmg, t_model_tcp, cfg.ee_link, 0.05))
    {
      continue;
    }
    ++result.ik_success;
    make_from_joints(jointsFromState(random), "random_j1_biased", false);
  }

  if (remaining() && result.attempts < static_cast<int>(cfg.max_ik_attempts))
  {
    moveit::core::RobotState wrist(pregrasp_scene.getCurrentState());
    applyJoints(wrist, baseline_pregrasp);
    wrist.setVariablePosition("j5", -baseline_pregrasp.at("j5"));
    wrist.update();
    ++result.attempts;
    if (tryIk(wrist, jmg, t_model_tcp, cfg.ee_link, 0.25))
    {
      ++result.ik_success;
      make_from_joints(jointsFromState(wrist), "wrist_flip_seed", false);
    }
  }

  std::sort(result.unique.begin(), result.unique.end(),
            [](const PreGraspIkCandidate& a, const PreGraspIkCandidate& b) {
              if (a.is_baseline != b.is_baseline)
              {
                return a.is_baseline;
              }
              if (std::abs(a.j1_abs_displacement - b.j1_abs_displacement) > 1e-9)
              {
                return a.j1_abs_displacement < b.j1_abs_displacement;
              }
              return a.distance_from_home < b.distance_from_home;
            });
  for (size_t i = 0; i < result.unique.size(); ++i)
  {
    if (!result.unique[i].is_baseline)
    {
      result.unique[i].candidate_id = "ik_" + std::to_string(i);
    }
  }
  result.baseline_included = false;
  for (const auto& cand : result.unique)
  {
    if (cand.is_baseline)
    {
      result.baseline_included = true;
      break;
    }
  }
  RCLCPP_INFO(logger,
              "IK attempts=%d ik_success=%d unique=%zu valid=%d baseline=%s min_j1=%.3f mode=%s",
              result.attempts, result.ik_success, result.unique.size(), result.unique_valid,
              result.baseline_included ? "YES" : "NO",
              result.unique.empty() ? -1.0 : result.unique.front().j1_abs_displacement,
              cfg.mode == IkValidationMode::AttachedInspection ? "attached_inspection" :
                                                                 "detached_pregrasp");
  return result;
}
}  // namespace fr_task_planner
