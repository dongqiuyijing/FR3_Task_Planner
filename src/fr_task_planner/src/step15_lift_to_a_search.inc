planning_scene::PlanningScenePtr makeAttachedInspectionScene(
    const planning_scene::PlanningScene& src, const moveit_msgs::msg::CollisionObject& object,
    const std::map<std::string, double>& grasp_joints, const std::string& object_id,
    const std::string& attach_link, const std::string& table_name,
    const std::vector<std::string>& touch_links)
{
  auto scene = cloneDiagnosticScene(src);
  detachObjectDiagnostic(*scene, object_id);
  for (const auto& name : scene->getWorld()->getObjectIds())
  {
    if (name == object_id)
    {
      removeWorldObjectDiagnostic(*scene, object_id);
      break;
    }
  }
  scene->processCollisionObjectMsg(object);
  applyJointsToScene(*scene, grasp_joints);
  scene->getAllowedCollisionMatrixNonConst().setEntry(object_id, touch_links, true);
  scene->getAllowedCollisionMatrixNonConst().setEntry(object_id, table_name, true);
  moveit_msgs::msg::AttachedCollisionObject attached;
  attached.link_name = attach_link;
  attached.object.id = object_id;
  attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  attached.touch_links = touch_links;
  scene->processAttachedCollisionObjectMsg(attached);
  scene->getAllowedCollisionMatrixNonConst().setEntry(object_id, table_name, false);
  return scene;
}

geometry_msgs::msg::PoseStamped poseFromJoints(const moveit::core::RobotModelConstPtr& robot_model,
                                               const std::map<std::string, double>& joints,
                                               const std::string& ee_link)
{
  moveit::core::RobotState fk(robot_model);
  fk.setToDefaultValues();
  applyJoints(fk, joints);
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "base_link";
  pose.pose = isoToPose(tcpInBase(fk, ee_link));
  return pose;
}

double j1NetDisplacement(const LogicalMetrics& m)
{
  if (m.start_joints.empty() || m.end_joints.empty())
  {
    return 0.0;
  }
  return std::abs(m.end_joints[0] - m.start_joints[0]);
}

bool j1SegmentTransferred(double neu_j1, double old_j1)
{
  return neu_j1 > 1.0 && neu_j1 > old_j1 + 0.3;
}

bool largeJ1Transferred(const FrozenTaskMetrics& neu, const FrozenTaskMetrics& old_m)
{
  return j1SegmentTransferred(neu.home_to_pregrasp.j1_abs_travel, old_m.home_to_pregrasp.j1_abs_travel) ||
         j1SegmentTransferred(neu.pregrasp_to_grasp.j1_abs_travel, old_m.pregrasp_to_grasp.j1_abs_travel) ||
         j1SegmentTransferred(neu.grasp_to_lift.j1_abs_travel, old_m.grasp_to_lift.j1_abs_travel) ||
         j1SegmentTransferred(neu.a_to_b.j1_abs_travel, old_m.a_to_b.j1_abs_travel) ||
         j1SegmentTransferred(neu.b_to_c.j1_abs_travel, old_m.b_to_c.j1_abs_travel);
}

std::string j1TransferReason(const FrozenTaskMetrics& neu, const FrozenTaskMetrics& old_m)
{
  std::ostringstream oss;
  auto add = [&](const char* name, double neu_j1, double old_j1) {
    if (!j1SegmentTransferred(neu_j1, old_j1))
    {
      return;
    }
    if (oss.tellp() > 0)
    {
      oss << "; ";
    }
    oss << name << "=" << neu_j1 << " (old=" << old_j1 << ")";
  };
  add("Home_to_PreGrasp", neu.home_to_pregrasp.j1_abs_travel, old_m.home_to_pregrasp.j1_abs_travel);
  add("PreGrasp_to_Grasp", neu.pregrasp_to_grasp.j1_abs_travel, old_m.pregrasp_to_grasp.j1_abs_travel);
  add("Grasp_to_Lift", neu.grasp_to_lift.j1_abs_travel, old_m.grasp_to_lift.j1_abs_travel);
  add("A_to_B", neu.a_to_b.j1_abs_travel, old_m.a_to_b.j1_abs_travel);
  add("B_to_C", neu.b_to_c.j1_abs_travel, old_m.b_to_c.j1_abs_travel);
  return oss.str().empty() ? "none" : oss.str();
}

double fullJ1Travel(const FrozenTaskMetrics& m)
{
  return m.home_to_pregrasp.j1_abs_travel + m.pregrasp_to_grasp.j1_abs_travel +
         m.grasp_to_lift.j1_abs_travel + m.lift_to_a.j1_abs_travel + m.a_to_b.j1_abs_travel +
         m.b_to_c.j1_abs_travel;
}

void writeLogicalMetricsYaml(std::ostream& yaml, const char* prefix, const LogicalMetrics& m)
{
  yaml << prefix << "_present: " << (m.present ? "true" : "false") << "\n";
  yaml << prefix << "_j1_start: " << (m.start_joints.empty() ? 0.0 : m.start_joints[0]) << "\n";
  yaml << prefix << "_j1_end: " << (m.end_joints.empty() ? 0.0 : m.end_joints[0]) << "\n";
  yaml << prefix << "_j1_net_displacement: " << j1NetDisplacement(m) << "\n";
  yaml << prefix << "_j1_cumulative_travel: " << m.j1_abs_travel << "\n";
  yaml << prefix << "_all_joint_path_length: " << m.all_joint_abs_travel << "\n";
  yaml << prefix << "_duration: " << m.duration << "\n";
}

double j1Of(const std::map<std::string, double>& joints)
{
  const auto it = joints.find("j1");
  return it == joints.end() ? 0.0 : it->second;
}

bool endpointJ1Ok(double from, double to, double limit)
{
  return std::abs(to - from) <= limit + 1e-9;
}

const LogicalMetrics* segmentByName(const FrozenTaskMetrics& m, const char* name)
{
  if (std::string(name) == "Home_to_PreGrasp")
  {
    return &m.home_to_pregrasp;
  }
  if (std::string(name) == "PreGrasp_to_Grasp")
  {
    return &m.pregrasp_to_grasp;
  }
  if (std::string(name) == "Grasp_to_Lift")
  {
    return &m.grasp_to_lift;
  }
  if (std::string(name) == "Lift_to_A")
  {
    return &m.lift_to_a;
  }
  if (std::string(name) == "A_to_B")
  {
    return &m.a_to_b;
  }
  if (std::string(name) == "B_to_C")
  {
    return &m.b_to_c;
  }
  return nullptr;
}

bool allSegmentsUnderJ1(const FrozenTaskMetrics& m, double limit, std::string& worst)
{
  const char* names[] = { "Home_to_PreGrasp", "PreGrasp_to_Grasp", "Grasp_to_Lift",
                          "Lift_to_A",        "A_to_B",           "B_to_C" };
  bool ok = true;
  double max_v = -1.0;
  worst = "none";
  for (const char* name : names)
  {
    const auto* seg = segmentByName(m, name);
    if (!seg || !seg->present)
    {
      ok = false;
      worst = std::string(name) + "_missing";
      continue;
    }
    if (seg->j1_abs_travel > limit + 1e-9)
    {
      ok = false;
      if (seg->j1_abs_travel > max_v)
      {
        max_v = seg->j1_abs_travel;
        worst = std::string(name) + "=" + std::to_string(seg->j1_abs_travel);
      }
    }
  }
  return ok;
}

bool abnormalOtherJointDetour(const PersistedTrajectory& traj, std::string& reason)
{
  const char* names[] = { kLogicalHomeToPreGrasp, kLogicalPreGraspToGrasp, kLogicalGraspToLift,
                          kLogicalLiftToA,        kLogicalAToB,           kLogicalBToC };
  for (const char* logical : names)
  {
    const auto* seg = lastLogicalSegment(traj, logical);
    if (!seg)
    {
      continue;
    }
    for (const auto& joint : kArmJoints)
    {
      if (joint == "j1")
      {
        continue;
      }
      const double cum = jointAbsTravel(*seg, joint);
      const auto start_arm = extractArmPositions(seg->joint_names, seg->start_joints);
      const auto end_arm = extractArmPositions(seg->joint_names, seg->end_joints);
      if (start_arm.size() < 6 || end_arm.size() < 6)
      {
        continue;
      }
      const auto start = vecToJoints(start_arm);
      const auto end = vecToJoints(end_arm);
      if (start.find(joint) == start.end() || end.find(joint) == end.end())
      {
        continue;
      }
      const double net = std::abs(end.at(joint) - start.at(joint));
      if (cum > net + M_PI)
      {
        reason = std::string(logical) + " " + joint + " detour cum=" + std::to_string(cum) +
                 " net=" + std::to_string(net);
        return true;
      }
      double prev = 0.0;
      bool have = false;
      for (const auto& pt : seg->points)
      {
        const auto q = vecToJoints(extractArmPositions(seg->joint_names, pt.positions));
        const double v = q.at(joint);
        if (have && std::abs(v - prev) > M_PI)
        {
          reason = std::string(logical) + " " + joint + " step wrap d=" + std::to_string(v - prev);
          return true;
        }
        prev = v;
        have = true;
      }
    }
  }
  return false;
}

const PreGraspIkCandidate* pickLockedCandidate(const std::vector<PreGraspIkCandidate>& cands,
                                               const std::string& want_id, double target_j1)
{
  for (const auto& cand : cands)
  {
    if (cand.candidate_id == want_id && (cand.valid || cand.is_baseline))
    {
      return &cand;
    }
  }
  const PreGraspIkCandidate* best = nullptr;
  double best_d = std::numeric_limits<double>::infinity();
  for (const auto& cand : cands)
  {
    if (!cand.valid)
    {
      continue;
    }
    const double d = std::abs(j1Of(cand.joints) - target_j1);
    if (d < best_d)
    {
      best_d = d;
      best = &cand;
    }
  }
  return best;
}

std::vector<PreGraspIkCandidate> visibleInspectionIk(
    planning_scene::PlanningScene& scene, const std::vector<PreGraspIkCandidate>& unique,
    const ViewGeom& view, const Eigen::Isometry3d& t_world_base,
    const Eigen::Isometry3d& t_tcp_object, const Eigen::Vector3d& p1, const Eigen::Vector3d& d1,
    const DesignCamera& camera, const std::string& ee_link, double object_radius,
    double object_height)
{
  std::vector<PreGraspIkCandidate> out;
  for (auto cand : unique)
  {
    if (!cand.valid && !cand.is_baseline)
    {
      continue;
    }
    applyJointsToScene(scene, cand.joints);
    const Eigen::Isometry3d t_base_tcp = tcpInBase(scene.getCurrentState(), ee_link);
    const Eigen::Isometry3d t_world_object = t_world_base * t_base_tcp * t_tcp_object;
    const auto vis = checkGeometricVisibility(scene, scene.getCurrentState(), view, t_world_object,
                                              p1, d1, camera, t_world_base, object_radius,
                                              object_height);
    if (!vis.orientation_valid && !cand.is_baseline)
    {
      continue;
    }
    if (cand.valid || cand.is_baseline)
    {
      out.push_back(cand);
    }
  }
  return out;
}

void rankByJ1Near(std::vector<PreGraspIkCandidate>& cands, double target_j1)
{
  std::sort(cands.begin(), cands.end(), [&](const PreGraspIkCandidate& a,
                                            const PreGraspIkCandidate& b) {
    return std::abs(j1Of(a.joints) - target_j1) < std::abs(j1Of(b.joints) - target_j1);
  });
}

double namedJoint(const std::map<std::string, double>& joints, const std::string& name)
{
  const auto it = joints.find(name);
  return it == joints.end() ? 0.0 : it->second;
}

double proximalL1(const std::map<std::string, double>& from, const std::map<std::string, double>& to)
{
  double sum = 0.0;
  for (const char* name : { "j1", "j2", "j3", "j4", "j5" })
  {
    sum += std::abs(namedJoint(to, name) - namedJoint(from, name));
  }
  return sum;
}

bool wrapJ6IntoLimits(double& j6)
{
  const double lo = -3.0543;
  const double hi = 3.0543;
  while (j6 > hi)
  {
    j6 -= 2.0 * M_PI;
  }
  while (j6 < lo)
  {
    j6 += 2.0 * M_PI;
  }
  return j6 >= lo - 1e-9 && j6 <= hi + 1e-9;
}

const PreGraspIkCandidate* pickNearestJoints(const std::vector<PreGraspIkCandidate>& cands,
                                             const std::map<std::string, double>& target)
{
  const PreGraspIkCandidate* best = nullptr;
  double best_d = std::numeric_limits<double>::infinity();
  for (const auto& cand : cands)
  {
    if (!cand.valid && !cand.is_baseline)
    {
      continue;
    }
    double d = 0.0;
    for (const auto& name : kArmJoints)
    {
      const double dq = namedJoint(cand.joints, name) - namedJoint(target, name);
      d += dq * dq;
    }
    if (d < best_d)
    {
      best_d = d;
      best = &cand;
    }
  }
  return best;
}

void rankByJ6DominantAToB(std::vector<PreGraspIkCandidate>& cands,
                          const std::map<std::string, double>& a_joints)
{
  std::sort(cands.begin(), cands.end(),
            [&](const PreGraspIkCandidate& lhs, const PreGraspIkCandidate& rhs) {
              const double pl = proximalL1(a_joints, lhs.joints);
              const double pr = proximalL1(a_joints, rhs.joints);
              if (std::abs(pl - pr) > 1e-9)
              {
                return pl < pr;
              }
              const double j6l = std::abs(namedJoint(lhs.joints, "j6") - namedJoint(a_joints, "j6"));
              const double j6r = std::abs(namedJoint(rhs.joints, "j6") - namedJoint(a_joints, "j6"));
              const double tl = std::abs(j6l - M_PI);
              const double tr = std::abs(j6r - M_PI);
              return tl < tr;
            });
}

bool aToBKeepOtherJointsSlight(const PersistedTrajectory& traj, double max_j234, double max_j5,
                               std::string& reason)
{
  const auto* seg = lastLogicalSegment(traj, kLogicalAToB);
  if (!seg)
  {
    reason = "A_to_B_missing";
    return false;
  }
  for (const char* name : { "j2", "j3", "j4" })
  {
    const double cum = jointAbsTravel(*seg, name);
    if (cum > max_j234 + 1e-9)
    {
      reason = std::string("A_to_B ") + name + " cum=" + std::to_string(cum) +
               " > " + std::to_string(max_j234);
      return false;
    }
  }
  const double j5 = jointAbsTravel(*seg, "j5");
  if (j5 > max_j5 + 1e-9)
  {
    reason = "A_to_B j5 cum=" + std::to_string(j5) + " > " + std::to_string(max_j5);
    return false;
  }
  return true;
}

std::vector<double> j6FaceLockSamples(double a6)
{
  std::vector<double> out;
  auto add = [&](double j6) {
    if (!wrapJ6IntoLimits(j6))
    {
      return;
    }
    for (double existing : out)
    {
      if (std::abs(existing - j6) < 1e-4)
      {
        return;
      }
    }
    out.push_back(j6);
  };
  add(a6 + M_PI);
  add(a6 - M_PI);
  for (int sign : { 1, -1 })
  {
    for (int k = -8; k <= 8; ++k)
    {
      add(a6 + static_cast<double>(sign) * M_PI + static_cast<double>(k) * (M_PI / 180.0));
    }
  }
  return out;
}

std::vector<PreGraspIkCandidate> enumerateJ6FaceLockedB(
    planning_scene::PlanningScene& attached, const std::map<std::string, double>& a_joints,
    const std::map<std::string, double>& home, const ViewGeom& view_b,
    const Eigen::Isometry3d& t_world_base, const Eigen::Isometry3d& t_tcp_object,
    const Eigen::Vector3d& p1, const Eigen::Vector3d& d1, const DesignCamera& camera,
    double object_radius, double object_height, const PreGraspIkConfig& cfg, double max_j1,
    double max_j5, const rclcpp::Logger& logger)
{
  std::vector<PreGraspIkCandidate> out;
  const double lo = -3.0543;
  const double hi = 3.0543;
  const double a5 = namedJoint(a_joints, "j5");
  const double a6 = namedJoint(a_joints, "j6");
  const double j5_nudge[] = { 0.0, 0.08, -0.08 };
  const auto j6_samples = j6FaceLockSamples(a6);
  (void)camera;
  (void)object_radius;
  (void)object_height;
  int kept = 0;
  int vis_ok = 0;
  double best_center = std::numeric_limits<double>::infinity();
  double best_ndot = -2.0;
  double best_j5 = a5;
  double best_j6 = a6;
  const Eigen::Vector3d d1u = d1.normalized();
  for (double d5 : j5_nudge)
  {
    const double j5 = a5 + d5;
    if (j5 < lo || j5 > hi)
    {
      continue;
    }
    for (double j6 : j6_samples)
    {
      auto q = a_joints;
      q["j5"] = j5;
      q["j6"] = j6;
      applyJointsToScene(attached, q);
      const Eigen::Isometry3d t_base_tcp = tcpInBase(attached.getCurrentState(), cfg.ee_link);
      const Eigen::Isometry3d t_world_object = t_world_base * t_base_tcp * t_tcp_object;
      const Eigen::Vector3d actual_center =
          t_world_object.translation() + t_world_object.linear() * view_b.center_in_object;
      const Eigen::Vector3d actual_normal =
          (t_world_object.linear() * view_b.normal_in_object).normalized();
      const double center_err = (actual_center - p1).norm();
      const double ndot = actual_normal.dot(d1u);
      if (center_err < best_center ||
          (std::abs(center_err - best_center) < 1e-6 && ndot > best_ndot))
      {
        best_center = center_err;
        best_ndot = ndot;
        best_j5 = j5;
        best_j6 = j6;
      }
      if (!(ndot > 0.9986 && center_err < 0.005))
      {
        continue;
      }
      ++vis_ok;
      PreGraspIkCandidate cand;
      cand.joints = q;
      cand.seed_name = "a_j6_face";
      const auto tcp = poseFromJoints(attached.getRobotModel(), q, cfg.ee_link);
      fr_task_planner::validatePreGraspCandidate(cand, attached, tcp, home, cfg);
      if (!cand.valid)
      {
        RCLCPP_INFO(logger,
                    "B face vis but invalid j5=%.3f j6=%.3f reason=%s center=%.4f ndot=%.4f", j5,
                    j6, cand.failure_reason.c_str(), center_err, ndot);
        continue;
      }
      if (!endpointJ1Ok(j1Of(a_joints), j1Of(q), max_j1))
      {
        continue;
      }
      const double dj5v = std::abs(j5 - a5);
      if (dj5v > max_j5)
      {
        continue;
      }
      const int dup = fr_task_planner::findDuplicateIndex(out, q, 0.05);
      if (dup >= 0)
      {
        continue;
      }
      cand.candidate_id = "b_j6_" + std::to_string(kept++);
      out.push_back(cand);
      RCLCPP_INFO(logger, "B face-lock %s j5=%.3f j6=%.3f dj6=%.3f center_err=%.5f ndot=%.6f",
                  cand.candidate_id.c_str(), j5, j6, std::abs(j6 - a6), center_err, ndot);
    }
  }
  RCLCPP_INFO(logger,
              "B j6-face sweep vis_ok=%d unique_kept=%zu best_center=%.4f best_ndot=%.5f at "
              "j5=%.3f j6=%.3f (need center<0.005 ndot>0.9986)",
              vis_ok, out.size(), best_center, best_ndot, best_j5, best_j6);
  return out;
}

std::vector<double> offsetSamples(double center, std::initializer_list<double> deltas)
{
  std::vector<double> out;
  for (double d : deltas)
  {
    const double q = center + d;
    if (q >= -3.0543 && q <= 3.0543)
    {
      out.push_back(q);
    }
  }
  return out;
}

std::vector<PreGraspIkCandidate> enumerateFaceLockedC(
    planning_scene::PlanningScene& attached, const std::map<std::string, double>& b_joints,
    const std::map<std::string, double>& home, const std::map<std::string, double>& old_c,
    const std::vector<RollPose>& rolls, const ViewGeom& view_c,
    const Eigen::Isometry3d& t_world_base, const Eigen::Isometry3d& t_tcp_object,
    const Eigen::Vector3d& p1, const Eigen::Vector3d& d1, const DesignCamera& camera,
    double object_radius, double object_height, const PreGraspIkConfig& cfg, double max_j1,
    const rclcpp::Logger& logger)
{
  std::vector<PreGraspIkCandidate> out;
  const Eigen::Vector3d d1u = d1.normalized();
  auto consider = [&](std::map<std::string, double> q, const std::string& tag) {
    if (!endpointJ1Ok(j1Of(b_joints), j1Of(q), max_j1))
    {
      return;
    }
    applyJointsToScene(attached, q);
    const Eigen::Isometry3d t_base_tcp = tcpInBase(attached.getCurrentState(), cfg.ee_link);
    const Eigen::Isometry3d t_world_object = t_world_base * t_base_tcp * t_tcp_object;
    const Eigen::Vector3d actual_center =
        t_world_object.translation() + t_world_object.linear() * view_c.center_in_object;
    const Eigen::Vector3d actual_normal =
        (t_world_object.linear() * view_c.normal_in_object).normalized();
    const double center_err = (actual_center - p1).norm();
    const double ndot = actual_normal.dot(d1u);
    RCLCPP_INFO(logger, "C hybrid %s j1=%.3f center=%.4f ndot=%.5f", tag.c_str(), j1Of(q),
                center_err, ndot);
    if (!(ndot > 0.9986 && center_err < 0.005))
    {
      return;
    }
    PreGraspIkCandidate cand;
    cand.joints = q;
    cand.seed_name = tag;
    const auto tcp = poseFromJoints(attached.getRobotModel(), q, cfg.ee_link);
    fr_task_planner::validatePreGraspCandidate(cand, attached, tcp, home, cfg);
    if (!cand.valid)
    {
      RCLCPP_INFO(logger, "C hybrid %s vis ok but invalid %s", tag.c_str(),
                  cand.failure_reason.c_str());
      return;
    }
    cand.candidate_id = tag;
    out.push_back(cand);
  };
  {
    auto hybrid = old_c;
    hybrid["j1"] = namedJoint(b_joints, "j1");
    consider(hybrid, "c_oldwrist_bj1");
    for (double dj1 : { -0.49, -0.35, -0.20, 0.20, 0.35, 0.49 })
    {
      hybrid["j1"] = namedJoint(b_joints, "j1") + dj1;
      consider(hybrid, "c_oldwrist_dj1");
    }
  }
  if (!out.empty())
  {
    rankByJ1Near(out, j1Of(b_joints));
    return out;
  }
  const auto j2s = offsetSamples(namedJoint(b_joints, "j2"), { 0.0, -0.45, 0.45, -0.90, 0.90 });
  const auto j3s = offsetSamples(namedJoint(b_joints, "j3"), { 0.0, -0.50, 0.50, -1.00, 1.00 });
  const auto j4s =
      offsetSamples(namedJoint(b_joints, "j4"), { 0.0, -0.60, 0.60, -1.20, 1.20, -2.00, 2.00 });
  std::vector<double> j5s = { namedJoint(b_joints, "j5"), 0.0, 0.70, -0.70, 1.20, -1.20, 2.20,
                              -2.20, 2.80, -2.80 };
  std::vector<double> j6s = offsetSamples(namedJoint(b_joints, "j6"),
                                         { 0.0, -0.60, 0.60, -1.20, 1.20, M_PI, -M_PI });
  int vis_ok = 0;
  double best_center = std::numeric_limits<double>::infinity();
  double best_ndot = -2.0;
  int kept = 0;
  for (double j2 : j2s)
  {
    for (double j3 : j3s)
    {
      for (double j4 : j4s)
      {
        for (double j5 : j5s)
        {
          if (j5 < -3.0543 || j5 > 3.0543)
          {
            continue;
          }
          for (double j6 : j6s)
          {
            auto q = b_joints;
            q["j1"] = namedJoint(b_joints, "j1");
            q["j2"] = j2;
            q["j3"] = j3;
            q["j4"] = j4;
            q["j5"] = j5;
            q["j6"] = j6;
            applyJointsToScene(attached, q);
            const Eigen::Isometry3d t_base_tcp = tcpInBase(attached.getCurrentState(), cfg.ee_link);
            const Eigen::Isometry3d t_world_object = t_world_base * t_base_tcp * t_tcp_object;
            const Eigen::Vector3d actual_center =
                t_world_object.translation() + t_world_object.linear() * view_c.center_in_object;
            const Eigen::Vector3d actual_normal =
                (t_world_object.linear() * view_c.normal_in_object).normalized();
            const double center_err = (actual_center - p1).norm();
            const double ndot = actual_normal.dot(d1u);
            if (center_err < best_center ||
                (std::abs(center_err - best_center) < 1e-6 && ndot > best_ndot))
            {
              best_center = center_err;
              best_ndot = ndot;
            }
            if (!(ndot > 0.9986 && center_err < 0.005))
            {
              continue;
            }
            ++vis_ok;
            PreGraspIkCandidate cand;
            cand.joints = q;
            cand.seed_name = "b_j1_face";
            const auto tcp = poseFromJoints(attached.getRobotModel(), q, cfg.ee_link);
            fr_task_planner::validatePreGraspCandidate(cand, attached, tcp, home, cfg);
            if (!cand.valid)
            {
              continue;
            }
            if (fr_task_planner::findDuplicateIndex(out, q, 0.08) >= 0)
            {
              continue;
            }
            cand.candidate_id = "c_j1_" + std::to_string(kept++);
            out.push_back(cand);
            RCLCPP_INFO(logger,
                        "C j1-lock %s j=[%.3f %.3f %.3f %.3f %.3f %.3f] center=%.5f ndot=%.6f",
                        cand.candidate_id.c_str(), namedJoint(q, "j1"), j2, j3, j4, j5, j6,
                        center_err, ndot);
            if (out.size() >= 4)
            {
              RCLCPP_INFO(logger, "C j1-lock sweep vis_ok=%d unique=%zu best_center=%.4f ndot=%.5f",
                          vis_ok, out.size(), best_center, best_ndot);
              return out;
            }
          }
        }
      }
    }
  }
  RCLCPP_INFO(logger, "C j1-lock sweep vis_ok=%d unique=%zu best_center=%.4f ndot=%.5f", vis_ok,
              out.size(), best_center, best_ndot);
  (void)rolls;
  (void)camera;
  (void)object_radius;
  (void)object_height;
  (void)max_j1;
  rankByJ1Near(out, j1Of(b_joints));
  return out;
}

int runLiftToAOptimize(
    const rclcpp::Node::SharedPtr& node, const std::string& group, const std::string& ee_link,
    const std::string& attach_link, const std::string& object_id, const std::string& table_name,
    const std::string& column_name, const std::vector<std::string>& touch_links,
    const geometry_msgs::msg::PoseStamped& object_scene,
    const geometry_msgs::msg::PoseStamped& pregrasp, const geometry_msgs::msg::PoseStamped& grasp,
    const geometry_msgs::msg::PoseStamped& lift, double object_height, double object_radius,
    double planning_time, const std::map<std::string, double>& home,
    const moveit::core::RobotModelConstPtr& robot_model, SearchViz& viz, const std::string& diag_path,
    double pos_tol, double ori_tol_deg,     const geometry_msgs::msg::PoseStamped& tcp_object,
    const geometry_msgs::msg::PoseStamped& world_base, const ViewGeom& view_a,
    const ViewGeom& view_b, const ViewGeom& view_c, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& d1)
{
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  RCLCPP_INFO(node->get_logger(), "========== STEP 15 J6-DOMINANT A->B FACE-CENTER LOCK ==========");
  RCLCPP_INFO(node->get_logger(), "PLAN / RViz ONLY. No Gazebo execute. No real robot.");
  RCLCPP_INFO(node->get_logger(),
              "Lock pre/A joints and A TCP. B/C TCP unlocked; side_neg_y and bottom_circle face "
              "centers stay at P1/D1. A->B is J6-dominant with slight other joints.");
  RCLCPP_INFO(node->get_logger(),
              "0.5 rad J1 is an offline search target, not a certified limit.");

  const std::string winner_in = getString(
      node, "winner_input_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner.yaml");
  const std::string frozen_path = getString(
      node, "frozen_trajectory_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step12c_tilted_camera_winner_trajectory.yaml");
  const std::string step14_traj_path = getString(
      node, "step14_trajectory_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step14_optimized_grasp_trajectory.yaml");
  const std::string step14_winner_path = getString(
      node, "step14_winner_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/step14_optimized_grasp_winner.yaml");
  const std::string step14_diag_path = getString(
      node, "step14_diag_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step14_grasp_optimization_diagnostics.yaml");
  std::string winner_out = getString(
      node, "winner_output_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step15_optimized_lift_to_a_winner.yaml");
  std::string traj_out = getString(
      node, "trajectory_output_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step15_optimized_lift_to_a_trajectory.yaml");
  if (targetsProtectedStep15BaselineFile(winner_out) || winner_out == winner_in)
  {
    winner_out =
        "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
        "step15_optimized_lift_to_a_winner.yaml";
  }
  if (targetsProtectedStep15BaselineFile(traj_out) || traj_out == frozen_path ||
      traj_out == step14_traj_path)
  {
    traj_out =
        "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
        "step15_optimized_lift_to_a_trajectory.yaml";
  }
  const int max_ik = getInt(node, "max_pregrasp_ik_candidates", 24);
  const int max_prefix = getInt(node, "max_prefix_plan_candidates", 12);
  const int max_a_ik = getInt(node, "max_a_ik_candidates", 16);
  const int max_b_ik = getInt(node, "max_b_ik_candidates", 32);
  const int max_c_ik = getInt(node, "max_c_ik_candidates", 16);
  const int max_full_combos = getInt(node, "max_full_task_candidates", 24);
  const int plan_attempts = getInt(node, "full_candidate_plan_attempts", 3);
  const double timeout_sec = node->has_parameter("step15_search_timeout_sec") ?
                                 node->get_parameter("step15_search_timeout_sec").as_double() :
                                 (node->has_parameter("step14_search_timeout_sec") ?
                                      node->get_parameter("step14_search_timeout_sec").as_double() :
                                      900.0);
  const double prefix_planning_time = node->has_parameter("step14_prefix_planning_time") ?
                                          node->get_parameter("step14_prefix_planning_time").as_double() :
                                          std::min(5.0, planning_time);
  const double full_planning_time = node->has_parameter("step14_full_planning_time") ?
                                        node->get_parameter("step14_full_planning_time").as_double() :
                                        planning_time;
  const double max_lift_to_a_j1 = node->has_parameter("max_lift_to_a_j1_travel_rad") ?
                                      node->get_parameter("max_lift_to_a_j1_travel_rad").as_double() :
                                      0.5;
  const double max_a_to_b_proximal = node->has_parameter("max_a_to_b_proximal_travel_rad") ?
                                         node->get_parameter("max_a_to_b_proximal_travel_rad").as_double() :
                                         0.40;
  const double max_a_to_b_j5 = node->has_parameter("max_a_to_b_j5_travel_rad") ?
                                   node->get_parameter("max_a_to_b_j5_travel_rad").as_double() :
                                   0.90;
  const std::string sha12c_w_before = fileSha256(winner_in);
  const std::string sha12c_t_before = fileSha256(frozen_path);
  const std::string sha14_w_before = fileSha256(step14_winner_path);
  const std::string sha14_t_before = fileSha256(step14_traj_path);
  const std::string sha14_d_before = fileSha256(step14_diag_path);

  WinnerEndpoints winner;
  std::string error;
  if (!loadWinnerYaml(winner_in, winner, error) || !winnerMatchesFrozenStep12c(winner, error))
  {
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: WINNER_ENDPOINT_MISMATCH\n";
    return 3;
  }
  PersistedTrajectory frozen12c;
  if (!readTrajectoryYaml(frozen_path, frozen12c, error))
  {
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: FROZEN_TRAJECTORY_UNREADABLE\n";
    return 3;
  }
  PersistedTrajectory step14_traj;
  FrozenTaskMetrics old_m = computeFrozenTaskMetrics(frozen12c);
  const bool have_step14 = readTrajectoryYaml(step14_traj_path, step14_traj, error);
  if (have_step14)
  {
    old_m = computeFrozenTaskMetrics(step14_traj);
  }
  RCLCPP_INFO(node->get_logger(),
              "BASELINE Lift->A J1 travel=%.6f rad (%.2f deg) Home->PreGrasp J1=%.6f",
              old_m.lift_to_a.j1_abs_travel, old_m.lift_to_a.j1_abs_travel * 180.0 / M_PI,
              old_m.home_to_pregrasp.j1_abs_travel);

  auto world_scene = fetchPlanningScene(node, robot_model);
  if (!world_scene)
  {
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: PLANNING_SCENE_UNAVAILABLE\n";
    return 2;
  }
  const auto object = makeCylinder(object_id, object_scene, object_height, object_radius);
  auto pregrasp_scene = makePreGraspPlanningScene(*world_scene, object, object_id, table_name);
  CollisionDiagConfig dcfg;
  dcfg.object_id = object_id;
  dcfg.table_name = table_name;
  dcfg.column_name = column_name;
  dcfg.touch_links = touch_links;

  const auto locked_a_pose = poseFromJoints(robot_model, winner.a_rad, ee_link);
  RCLCPP_INFO(node->get_logger(), "LOCKED A TCP xyz=%.6f %.6f %.6f xyzw=%.6f %.6f %.6f %.6f",
              locked_a_pose.pose.position.x, locked_a_pose.pose.position.y,
              locked_a_pose.pose.position.z, locked_a_pose.pose.orientation.x,
              locked_a_pose.pose.orientation.y, locked_a_pose.pose.orientation.z,
              locked_a_pose.pose.orientation.w);

  std::map<std::string, double> grasp_seed = winner.a_rad;
  if (const auto* grasp_seg = lastLogicalSegment(frozen12c, kLogicalPreGraspToGrasp))
  {
    grasp_seed = vecToJoints(extractArmPositions(grasp_seg->joint_names, grasp_seg->end_joints));
  }
  if (have_step14)
  {
    if (const auto* g14 = lastLogicalSegment(step14_traj, kLogicalPreGraspToGrasp))
    {
      grasp_seed = vecToJoints(extractArmPositions(g14->joint_names, g14->end_joints));
    }
  }
  auto attached_scene = makeAttachedInspectionScene(*world_scene, object, grasp_seed, object_id,
                                                    attach_link, table_name, touch_links);

  PreGraspIkConfig pre_cfg;
  pre_cfg.group = group;
  pre_cfg.ee_link = ee_link;
  pre_cfg.object_id = object_id;
  pre_cfg.table_name = table_name;
  pre_cfg.column_name = column_name;
  pre_cfg.touch_links = touch_links;
  pre_cfg.pos_tol = pos_tol;
  pre_cfg.ori_tol_deg = ori_tol_deg;
  pre_cfg.min_ik_solution_distance = node->has_parameter("min_ik_solution_distance") ?
                                         node->get_parameter("min_ik_solution_distance").as_double() :
                                         0.1;
  pre_cfg.max_unique_candidates = static_cast<uint32_t>(std::max(1, max_ik));
  pre_cfg.max_ik_attempts = static_cast<uint32_t>(std::max(8, getInt(node, "max_ik_attempts", 96)));
  const auto baseline_pre = vecToJoints(kStep12cPreGraspRad);
  std::vector<NamedJointSeed> extra = { { "view_a", winner.a_rad },
                                        { "view_b", winner.b_rad },
                                        { "view_c", winner.c_rad } };
  const auto ik = generatePreGraspIkCandidates(*pregrasp_scene, pregrasp, home, baseline_pre, extra,
                                               pre_cfg, node->get_logger());

  PreGraspIkConfig a_cfg = pre_cfg;
  a_cfg.mode = IkValidationMode::AttachedInspection;
  a_cfg.max_unique_candidates = static_cast<uint32_t>(std::max(1, max_a_ik));
  std::vector<NamedJointSeed> a_extra = { { "home", home },
                                          { "view_b", winner.b_rad },
                                          { "view_c", winner.c_rad },
                                          { "baseline_pregrasp", baseline_pre } };
  const auto a_ik = generatePreGraspIkCandidates(*attached_scene, locked_a_pose, home, winner.a_rad,
                                                 a_extra, a_cfg, node->get_logger());
  const Eigen::Isometry3d t_world_base = poseToIso(world_base.pose);
  const Eigen::Isometry3d t_tcp_object = poseToIso(tcp_object.pose);
  const DesignCamera camera = fixedInspectionDesignCamera();
  int a_visibility_ok = 0;
  std::vector<PreGraspIkCandidate> a_valid;
  for (auto cand : a_ik.unique)
  {
    if (!cand.valid && !cand.is_baseline)
    {
      continue;
    }
    applyJointsToScene(*attached_scene, cand.joints);
    const Eigen::Isometry3d t_base_tcp = tcpInBase(attached_scene->getCurrentState(), ee_link);
    const Eigen::Isometry3d t_world_object = t_world_base * t_base_tcp * t_tcp_object;
    const auto vis = checkGeometricVisibility(*attached_scene, attached_scene->getCurrentState(),
                                              view_a, t_world_object, p1, d1, camera, t_world_base,
                                              object_radius, object_height);
    if (!vis.orientation_valid && !cand.is_baseline)
    {
      cand.valid = false;
      cand.failure_reason = "A face/orientation invalid";
    }
    if (cand.valid)
    {
      ++a_visibility_ok;
      a_valid.push_back(cand);
    }
    else if (cand.is_baseline)
    {
      a_valid.push_back(cand);
    }
  }
  const auto old_b_pose = poseFromJoints(robot_model, winner.b_rad, ee_link);
  const auto locked_c_pose = poseFromJoints(robot_model, winner.c_rad, ee_link);
  RCLCPP_INFO(node->get_logger(),
              "OLD STEP12C B TCP (unlocked this search) xyz=%.6f %.6f %.6f xyzw=%.6f %.6f %.6f %.6f",
              old_b_pose.pose.position.x, old_b_pose.pose.position.y, old_b_pose.pose.position.z,
              old_b_pose.pose.orientation.x, old_b_pose.pose.orientation.y,
              old_b_pose.pose.orientation.z, old_b_pose.pose.orientation.w);
  RCLCPP_INFO(node->get_logger(), "LOCKED C TCP xyz=%.6f %.6f %.6f xyzw=%.6f %.6f %.6f %.6f",
              locked_c_pose.pose.position.x, locked_c_pose.pose.position.y,
              locked_c_pose.pose.position.z, locked_c_pose.pose.orientation.x,
              locked_c_pose.pose.orientation.y, locked_c_pose.pose.orientation.z,
              locked_c_pose.pose.orientation.w);

  const std::string want_pre = getString(node, "lock_pregrasp_candidate_id", "ik_1");
  const std::string want_a = getString(node, "lock_a_candidate_id", "ik_1");
  const std::map<std::string, double> locked_pre_joints = {
      { "j1", -2.155823172284 }, { "j2", -3.222149477732 }, { "j3", -0.421558847879 },
      { "j4", -2.639635474335 }, { "j5", 0.200535044780 },  { "j6", 0.785380325417 }
  };
  const std::map<std::string, double> locked_a_joints = {
      { "j1", -1.730657532797 }, { "j2", -2.096082272275 }, { "j3", -2.361345599522 },
      { "j4", -0.254957192512 }, { "j5", 1.570792677784 },  { "j6", -2.516051849249 }
  };
  const double target_pre_j1 = old_m.home_to_pregrasp.end_joints.empty() ?
                                   -2.15582 :
                                   old_m.home_to_pregrasp.end_joints[0];
  const double target_a_j1 = -1.730656;
  const auto* pre_locked = pickNearestJoints(ik.unique, locked_pre_joints);
  const auto* a_locked = pickNearestJoints(a_valid, locked_a_joints);
  if (!pre_locked)
  {
    pre_locked = pickLockedCandidate(ik.unique, want_pre, target_pre_j1);
  }
  if (!a_locked)
  {
    a_locked = pickLockedCandidate(a_valid, want_a, target_a_j1);
  }
  if (!pre_locked || !a_locked)
  {
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: LOCKED_CANDIDATE_MISSING\n";
    yaml << "want_pre: " << want_pre << "\nwant_a: " << want_a << "\n";
    yaml << "pregrasp_ik_unique: " << ik.unique.size() << "\n";
    yaml << "a_valid: " << a_valid.size() << "\n";
    return 4;
  }
  RCLCPP_INFO(node->get_logger(), "LOCKED pre=%s j1=%.6f a=%s j1=%.6f",
              pre_locked->candidate_id.c_str(), j1Of(pre_locked->joints),
              a_locked->candidate_id.c_str(), j1Of(a_locked->joints));

  GraspPlanOutcome prefix_plan =
      planGraspCandidate(node, group, ee_link, attach_link, object_id, table_name, touch_links,
                         object_scene, pregrasp, grasp, lift, object_height, object_radius,
                         prefix_planning_time, home, pre_locked->joints, winner, false, plan_attempts,
                         dcfg, pos_tol, ori_tol_deg);
  if (!prefix_plan.ok)
  {
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: PREFIX_PLAN_FAILED\n";
    yaml << "failure: " << prefix_plan.failure << "\n";
    yaml << "locked_pre: " << pre_locked->candidate_id << "\n";
    return 4;
  }
  attached_scene = makeAttachedInspectionScene(*world_scene, object, prefix_plan.grasp_joints,
                                               object_id, attach_link, table_name, touch_links);

  PreGraspIkConfig b_cfg = a_cfg;
  b_cfg.max_unique_candidates = static_cast<uint32_t>(std::max(1, max_b_ik));
  b_cfg.max_ik_attempts = 192;
  std::vector<PreGraspIkCandidate> b_near = enumerateJ6FaceLockedB(
      *attached_scene, a_locked->joints, home, view_b, t_world_base, t_tcp_object, p1, d1, camera,
      object_radius, object_height, b_cfg, max_lift_to_a_j1, max_a_to_b_j5, node->get_logger());
  rankByJ6DominantAToB(b_near, a_locked->joints);
  if (static_cast<int>(b_near.size()) > max_b_ik)
  {
    b_near.resize(static_cast<size_t>(max_b_ik));
  }
  const auto b_visible = b_near;
  RCLCPP_INFO(node->get_logger(),
              "B face-lock unique=%zu j6_dominant_j1_ok=%zu max_proximal=%.3f", b_near.size(),
              b_near.size(), max_a_to_b_proximal);
  for (const auto& cand : b_near)
  {
    RCLCPP_INFO(node->get_logger(), "B keep %s proximal_l1=%.3f dj6=%.3f j1=%.3f",
                cand.candidate_id.c_str(), proximalL1(a_locked->joints, cand.joints),
                std::abs(namedJoint(cand.joints, "j6") - namedJoint(a_locked->joints, "j6")),
                j1Of(cand.joints));
  }

  struct Combo
  {
    PreGraspIkCandidate pre;
    PreGraspIkCandidate a;
    PreGraspIkCandidate b;
    PreGraspIkCandidate c;
    GraspPlanOutcome plan;
    std::string fail;
  };
  std::vector<Combo> attempted;
  std::optional<Combo> win;
  int full_tested = 0;
  int c_endpoint_ok = 0;
  double min_ab = std::numeric_limits<double>::infinity();
  double min_bc = std::numeric_limits<double>::infinity();
  std::string min_ab_id = "none";
  std::string min_bc_id = "none";
  int b_ik_runs = 1;
  int c_ik_runs = 0;
  const auto c_rolls = readRollPoses(node);

  for (const auto& b_cand : b_near)
  {
    if (!rclcpp::ok() || elapsed() > timeout_sec || win)
    {
      break;
    }
    PreGraspIkConfig c_cfg = b_cfg;
    c_cfg.max_unique_candidates = static_cast<uint32_t>(std::max(1, max_c_ik));
    c_cfg.max_ik_attempts = 32;
    ++c_ik_runs;
    auto c_near =
        enumerateFaceLockedC(*attached_scene, b_cand.joints, home, winner.c_rad, c_rolls, view_c,
                             t_world_base, t_tcp_object, p1, d1, camera, object_radius,
                             object_height, c_cfg, max_lift_to_a_j1, node->get_logger());
    c_endpoint_ok += static_cast<int>(c_near.size());
    RCLCPP_INFO(node->get_logger(), "C face-lock for b=%s endpoint_j1_ok=%zu",
                b_cand.candidate_id.c_str(), c_near.size());

    for (const auto& c_cand : c_near)
    {
      if (!rclcpp::ok() || elapsed() > timeout_sec || full_tested >= max_full_combos || win)
      {
        break;
      }
      ++full_tested;
      Combo combo;
      combo.pre = *pre_locked;
      combo.a = *a_locked;
      combo.b = b_cand;
      combo.c = c_cand;
      WinnerEndpoints scored = winner;
      scored.a_rad = a_locked->joints;
      scored.b_rad = b_cand.joints;
      scored.c_rad = c_cand.joints;
      RCLCPP_INFO(node->get_logger(),
                  "STEP15 first-feasible combo pre=%s a=%s b=%s c=%s a_j1=%.3f b_j1=%.3f c_j1=%.3f",
                  pre_locked->candidate_id.c_str(), a_locked->candidate_id.c_str(),
                  b_cand.candidate_id.c_str(), c_cand.candidate_id.c_str(), j1Of(a_locked->joints),
                  j1Of(b_cand.joints), j1Of(c_cand.joints));
      const auto b_pose = poseFromJoints(robot_model, b_cand.joints, ee_link);
      const auto c_pose = poseFromJoints(robot_model, c_cand.joints, ee_link);
      combo.plan = planGraspCandidate(
          node, group, ee_link, attach_link, object_id, table_name, touch_links, object_scene,
          pregrasp, grasp, lift, object_height, object_radius, full_planning_time, home,
          pre_locked->joints, scored, true, plan_attempts, dcfg, pos_tol, ori_tol_deg,
          &a_locked->joints, &locked_a_pose, &b_cand.joints, &c_cand.joints, &b_pose, &c_pose);
      if (!combo.plan.ok)
      {
        combo.fail = combo.plan.failure;
        attempted.push_back(std::move(combo));
        continue;
      }
      if (combo.plan.metrics.a_to_b.present)
      {
        if (combo.plan.metrics.a_to_b.j1_abs_travel < min_ab)
        {
          min_ab = combo.plan.metrics.a_to_b.j1_abs_travel;
          min_ab_id = b_cand.candidate_id + "/" + c_cand.candidate_id;
        }
      }
      if (combo.plan.metrics.b_to_c.present)
      {
        if (combo.plan.metrics.b_to_c.j1_abs_travel < min_bc)
        {
          min_bc = combo.plan.metrics.b_to_c.j1_abs_travel;
          min_bc_id = b_cand.candidate_id + "/" + c_cand.candidate_id;
        }
      }
      std::string worst;
      if (!allSegmentsUnderJ1(combo.plan.metrics, max_lift_to_a_j1, worst))
      {
        combo.fail = "J1_LIMIT " + worst;
        attempted.push_back(std::move(combo));
        continue;
      }
      std::string detour;
      if (abnormalOtherJointDetour(combo.plan.traj, detour))
      {
        combo.fail = "JOINT_DETOUR " + detour;
        attempted.push_back(std::move(combo));
        continue;
      }
      std::string ab_j6;
      if (!aToBKeepOtherJointsSlight(combo.plan.traj, max_a_to_b_proximal, max_a_to_b_j5, ab_j6))
      {
        combo.fail = "A_TO_B_NOT_J6_DOMINANT " + ab_j6;
        attempted.push_back(std::move(combo));
        continue;
      }
      combo.fail = "ok";
      win = combo;
      attempted.push_back(std::move(combo));
      RCLCPP_INFO(node->get_logger(), "STEP15 FIRST FEASIBLE PATH FOUND. Stopping search.");
      break;
    }
  }
  (void)max_prefix;
  (void)b_ik_runs;

  bool identity = false;
  FrozenTaskMetrics loaded_m;
  PersistValidation roundtrip;
  PersistValidation continuity;
  PersistValidation endpoints;
  if (win)
  {
    WinnerEndpoints saved_winner = winner;
    saved_winner.a_rad = win->a.joints;
    saved_winner.b_rad = win->b.joints;
    saved_winner.c_rad = win->c.joints;
    PersistedTrajectory saved = win->plan.traj;
    saved.task_version = "STEP15";
    saved.label = "STEP15 OPTIMIZED LIFT-TO-A TRAJECTORY (OFFLINE REVIEW ONLY)";
    saved.source_winner_file = fileBaseName(winner_in);
    saved.source_winner_path = winner_in;
    saved.search_score_total_time = win->plan.metrics.full_deployable_duration;
    saved.search_score_joint_path_length = win->plan.metrics.full_deployable_path_length_l2;
    saved.winner = saved_winner;
    saved.execution_performed = false;
    saved.real_robot_validated = false;
    fillDerivedTotals(saved);
    if (!writeTrajectoryYaml(traj_out, saved, error))
    {
      std::ofstream yaml(diag_path);
      yaml << "status: FAIL\nfailure_classification: TRAJECTORY_SERIALIZE_FAILED\n";
      return 5;
    }
    PersistedTrajectory loaded;
    if (!readTrajectoryYaml(traj_out, loaded, error))
    {
      std::ofstream yaml(diag_path);
      yaml << "status: FAIL\nfailure_classification: TRAJECTORY_READBACK_FAILED\n";
      return 5;
    }
    roundtrip = validateRoundTrip(saved, loaded);
    continuity = validateContinuity(loaded);
    endpoints = validateEndpoints(loaded, saved_winner, 1e-3);
    loaded_m = computeFrozenTaskMetrics(loaded);
    identity = roundtrip.ok &&
               std::abs(loaded_m.lift_to_a.j1_abs_travel - win->plan.metrics.lift_to_a.j1_abs_travel) <
                   1e-12 &&
               std::abs(loaded_m.a_to_b.j1_abs_travel - win->plan.metrics.a_to_b.j1_abs_travel) <
                   1e-12;
    const auto win_b_pose = poseFromJoints(robot_model, win->b.joints, ee_link);
    const auto win_c_pose = poseFromJoints(robot_model, win->c.joints, ee_link);
    std::ofstream winy(winner_out);
    winy.setf(std::ios::fixed);
    winy.precision(17);
    winy << "label: STEP15 OPTIMIZED LIFT-TO-A WINNER\n";
    winy << "task_version: STEP15\n";
    winy << "path_improvement_verified: YES\n";
    winy << "offline_review_only: true\n";
    winy << "ready_for_real_execution: NO\n";
    winy << "first_feasible_stop: true\n";
    winy << "pregrasp_candidate_id: " << win->pre.candidate_id << "\n";
    winy << "a_candidate_id: " << win->a.candidate_id << "\n";
    winy << "b_candidate_id: " << win->b.candidate_id << "\n";
    winy << "c_candidate_id: " << win->c.candidate_id << "\n";
    winy << "home_joints_rad: " << formatHomeList(home) << "\n";
    winy << "pregrasp_joints_rad: " << formatHomeList(win->plan.pregrasp_joints) << "\n";
    winy << "grasp_joints_rad: " << formatHomeList(win->plan.grasp_joints) << "\n";
    winy << "lift_joints_rad: " << formatHomeList(win->plan.lift_joints) << "\n";
    winy << "a_roll_deg: " << winner.a_roll_deg << "\n";
    winy << "b_roll_deg: " << winner.b_roll_deg << "\n";
    winy << "c_roll_deg: " << winner.c_roll_deg << "\n";
    winy << "a_joints_rad: " << formatHomeList(saved_winner.a_rad) << "\n";
    winy << "b_joints_rad: " << formatHomeList(saved_winner.b_rad) << "\n";
    winy << "c_joints_rad: " << formatHomeList(saved_winner.c_rad) << "\n";
    winy << "a_tcp_position: [" << locked_a_pose.pose.position.x << ", "
         << locked_a_pose.pose.position.y << ", " << locked_a_pose.pose.position.z << "]\n";
    winy << "a_tcp_orientation_xyzw: [" << locked_a_pose.pose.orientation.x << ", "
         << locked_a_pose.pose.orientation.y << ", " << locked_a_pose.pose.orientation.z << ", "
         << locked_a_pose.pose.orientation.w << "]\n";
    winy << "b_tcp_unlocked: true\n";
    winy << "b_face_center_locked: true\n";
    winy << "b_tcp_position: [" << win_b_pose.pose.position.x << ", "
         << win_b_pose.pose.position.y << ", " << win_b_pose.pose.position.z << "]\n";
    winy << "b_tcp_orientation_xyzw: [" << win_b_pose.pose.orientation.x << ", "
         << win_b_pose.pose.orientation.y << ", " << win_b_pose.pose.orientation.z << ", "
         << win_b_pose.pose.orientation.w << "]\n";
    winy << "c_tcp_unlocked: true\n";
    winy << "c_face_center_locked: true\n";
    winy << "c_tcp_position: [" << win_c_pose.pose.position.x << ", "
         << win_c_pose.pose.position.y << ", " << win_c_pose.pose.position.z << "]\n";
    winy << "c_tcp_orientation_xyzw: [" << win_c_pose.pose.orientation.x << ", "
         << win_c_pose.pose.orientation.y << ", " << win_c_pose.pose.orientation.z << ", "
         << win_c_pose.pose.orientation.w << "]\n";
    winy << "home_to_pregrasp_j1: " << loaded_m.home_to_pregrasp.j1_abs_travel << "\n";
    winy << "pregrasp_to_grasp_j1: " << loaded_m.pregrasp_to_grasp.j1_abs_travel << "\n";
    winy << "grasp_to_lift_j1: " << loaded_m.grasp_to_lift.j1_abs_travel << "\n";
    winy << "lift_to_a_j1_cumulative_travel: " << loaded_m.lift_to_a.j1_abs_travel << "\n";
    winy << "a_to_b_j1: " << loaded_m.a_to_b.j1_abs_travel << "\n";
    {
      const auto* ab = lastLogicalSegment(loaded, kLogicalAToB);
      winy << "a_to_b_j6_dominant: YES\n";
      winy << "a_to_b_j2_cum: " << (ab ? jointAbsTravel(*ab, "j2") : -1.0) << "\n";
      winy << "a_to_b_j3_cum: " << (ab ? jointAbsTravel(*ab, "j3") : -1.0) << "\n";
      winy << "a_to_b_j4_cum: " << (ab ? jointAbsTravel(*ab, "j4") : -1.0) << "\n";
      winy << "a_to_b_j5_cum: " << (ab ? jointAbsTravel(*ab, "j5") : -1.0) << "\n";
      winy << "a_to_b_j6_cum: " << (ab ? jointAbsTravel(*ab, "j6") : -1.0) << "\n";
    }
    winy << "b_to_c_j1: " << loaded_m.b_to_c.j1_abs_travel << "\n";
    winy << "full_duration: " << loaded_m.full_deployable_duration << "\n";
    winy << "max_segment_j1_travel_rad: " << max_lift_to_a_j1 << "\n";
    winy << "scored_trajectory_is_saved_trajectory: " << (identity ? "YES" : "NO") << "\n";
    winy << "execution: false\n";
    if (viz.traj)
    {
      viz.traj->publish(displayFromPersisted(loaded, robot_model->getName()));
    }
  }

  const std::string sha12c_w_after = fileSha256(winner_in);
  const std::string sha12c_t_after = fileSha256(frozen_path);
  const std::string sha14_w_after = fileSha256(step14_winner_path);
  const std::string sha14_t_after = fileSha256(step14_traj_path);
  const std::string sha14_d_after = fileSha256(step14_diag_path);
  const bool baseline_preserved =
      sha12c_w_before == sha12c_w_after && sha12c_t_before == sha12c_t_after &&
      sha14_w_before == sha14_w_after && sha14_t_before == sha14_t_after &&
      sha14_d_before == sha14_d_after;
  const char* failure = "none";
  if (!win)
  {
    failure = b_near.empty() ? "NO_FEASIBLE_J6_DOMINANT_A_TO_B" :
                               "NO_FEASIBLE_CANDIDATE_UNDER_J1_LIMIT";
  }
  const auto report_b_pose =
      win ? poseFromJoints(robot_model, win->b.joints, ee_link) : old_b_pose;
  const auto report_c_pose =
      win ? poseFromJoints(robot_model, win->c.joints, ee_link) : locked_c_pose;
  std::ofstream yaml(diag_path);
  yaml.setf(std::ios::fixed);
  yaml.precision(17);
  yaml << "status: " << (win ? "PASS" : "FAIL") << "\n";
  yaml << "task_version: STEP15\n";
  yaml << "failure_classification: " << failure << "\n";
  yaml << "path_improvement_verified: " << (win ? "YES" : "NO") << "\n";
  yaml << "offline_review_only: true\n";
  yaml << "ready_for_real_execution: NO\n";
  yaml << "first_feasible_stop: true\n";
  yaml << "max_lift_to_a_j1_travel_rad: " << max_lift_to_a_j1 << "\n";
  yaml << "max_a_to_b_proximal_travel_rad: " << max_a_to_b_proximal << "\n";
  yaml << "max_a_to_b_j5_travel_rad: " << max_a_to_b_j5 << "\n";
  yaml << "a_to_b_prefer_j6: YES\n";
  yaml << "note_j1_limit: offline search target, not a manufacturer certified threshold\n";
  yaml << "baseline_preserved: " << (baseline_preserved ? "YES" : "NO") << "\n";
  yaml << "step12c_winner_sha256_before: " << sha12c_w_before << "\n";
  yaml << "step12c_winner_sha256_after: " << sha12c_w_after << "\n";
  yaml << "step12c_trajectory_sha256_before: " << sha12c_t_before << "\n";
  yaml << "step12c_trajectory_sha256_after: " << sha12c_t_after << "\n";
  yaml << "step14_winner_sha256_before: " << sha14_w_before << "\n";
  yaml << "step14_winner_sha256_after: " << sha14_w_after << "\n";
  yaml << "step14_trajectory_sha256_before: " << sha14_t_before << "\n";
  yaml << "step14_trajectory_sha256_after: " << sha14_t_after << "\n";
  yaml << "step14_diag_sha256_before: " << sha14_d_before << "\n";
  yaml << "step14_diag_sha256_after: " << sha14_d_after << "\n";
  yaml << "locked_pre_id: " << pre_locked->candidate_id << "\n";
  yaml << "locked_a_id: " << a_locked->candidate_id << "\n";
  yaml << "locked_pre_j1: " << j1Of(pre_locked->joints) << "\n";
  yaml << "locked_a_j1: " << j1Of(a_locked->joints) << "\n";
  yaml << "a_tcp_position: [" << locked_a_pose.pose.position.x << ", "
       << locked_a_pose.pose.position.y << ", " << locked_a_pose.pose.position.z << "]\n";
  yaml << "a_tcp_orientation_xyzw: [" << locked_a_pose.pose.orientation.x << ", "
       << locked_a_pose.pose.orientation.y << ", " << locked_a_pose.pose.orientation.z << ", "
       << locked_a_pose.pose.orientation.w << "]\n";
  yaml << "old_step12c_b_tcp_position: [" << old_b_pose.pose.position.x << ", "
       << old_b_pose.pose.position.y << ", " << old_b_pose.pose.position.z << "]\n";
  yaml << "b_tcp_position: [" << report_b_pose.pose.position.x << ", "
       << report_b_pose.pose.position.y << ", " << report_b_pose.pose.position.z << "]\n";
  yaml << "b_tcp_orientation_xyzw: [" << report_b_pose.pose.orientation.x << ", "
       << report_b_pose.pose.orientation.y << ", " << report_b_pose.pose.orientation.z << ", "
       << report_b_pose.pose.orientation.w << "]\n";
  yaml << "c_tcp_position: [" << report_c_pose.pose.position.x << ", "
       << report_c_pose.pose.position.y << ", " << report_c_pose.pose.position.z << "]\n";
  yaml << "c_tcp_orientation_xyzw: [" << report_c_pose.pose.orientation.x << ", "
       << report_c_pose.pose.orientation.y << ", " << report_c_pose.pose.orientation.z << ", "
       << report_c_pose.pose.orientation.w << "]\n";
  yaml << "pregrasp_ik_unique: " << ik.unique.size() << "\n";
  yaml << "pregrasp_ik_valid: " << ik.unique_valid << "\n";
  yaml << "a_ik_unique: " << a_ik.unique.size() << "\n";
  yaml << "a_ik_valid: " << a_ik.unique_valid << "\n";
  yaml << "a_visibility_ok: " << a_visibility_ok << "\n";
  yaml << "b_ik_unique: " << b_near.size() << "\n";
  yaml << "b_visible: " << b_visible.size() << "\n";
  yaml << "b_endpoint_j1_ok: " << b_near.size() << "\n";
  yaml << "c_ik_runs: " << c_ik_runs << "\n";
  yaml << "c_endpoint_j1_ok: " << c_endpoint_ok << "\n";
  yaml << "full_combinations_tested: " << full_tested << "\n";
  yaml << "full_combinations_feasible: " << (win ? 1 : 0) << "\n";
  yaml << "search_elapsed_sec: " << elapsed() << "\n";
  yaml << "search_timeout_sec: " << timeout_sec << "\n";
  yaml << "timed_out: " << (elapsed() > timeout_sec ? "true" : "false") << "\n";
  yaml << "fixed_bc_tcp_locked: NO\n";
  yaml << "b_tcp_pose_locked: NO\n";
  yaml << "b_face_center_locked: YES\n";
  yaml << "c_tcp_pose_locked: NO\n";
  yaml << "c_face_center_locked: YES\n";
  yaml << "bc_joint_ik_unlocked: YES\n";
  yaml << "a_tcp_pose_locked: YES\n";
  yaml << "pilz_lin_preserved: YES\n";
  yaml << "real_robot_commands: 0\n";
  yaml << "gripper_commands: 0\n";
  yaml << "execution: false\n";
  yaml << "prefix_plan_ok: " << (prefix_plan.ok ? "true" : "false") << "\n";
  yaml << "prefix_home_to_pregrasp_j1: " << prefix_plan.metrics.home_to_pregrasp.j1_abs_travel << "\n";
  yaml << "prefix_lift_to_a_j1_if_unplanned: n/a\n";
  writeLogicalMetricsYaml(yaml, "old_home_to_pregrasp", old_m.home_to_pregrasp);
  writeLogicalMetricsYaml(yaml, "old_pregrasp_to_grasp", old_m.pregrasp_to_grasp);
  writeLogicalMetricsYaml(yaml, "old_grasp_to_lift", old_m.grasp_to_lift);
  writeLogicalMetricsYaml(yaml, "old_lift_to_a", old_m.lift_to_a);
  writeLogicalMetricsYaml(yaml, "old_a_to_b", old_m.a_to_b);
  writeLogicalMetricsYaml(yaml, "old_b_to_c", old_m.b_to_c);
  yaml << "old_full_j1_travel: " << fullJ1Travel(old_m) << "\n";
  yaml << "old_full_home_to_c_time: " << old_m.full_deployable_duration << "\n";
  if (win)
  {
    writeLogicalMetricsYaml(yaml, "new_home_to_pregrasp", loaded_m.home_to_pregrasp);
    writeLogicalMetricsYaml(yaml, "new_pregrasp_to_grasp", loaded_m.pregrasp_to_grasp);
    writeLogicalMetricsYaml(yaml, "new_grasp_to_lift", loaded_m.grasp_to_lift);
    writeLogicalMetricsYaml(yaml, "new_lift_to_a", loaded_m.lift_to_a);
    writeLogicalMetricsYaml(yaml, "new_a_to_b", loaded_m.a_to_b);
    writeLogicalMetricsYaml(yaml, "new_b_to_c", loaded_m.b_to_c);
    yaml << "new_full_j1_travel: " << fullJ1Travel(loaded_m) << "\n";
    yaml << "new_full_home_to_c_time: " << loaded_m.full_deployable_duration << "\n";
    yaml << "new_minimum_clearance: " << win->plan.min_clearance << "\n";
    yaml << "collision_free: " << (win->plan.collision_free ? "true" : "false") << "\n";
    yaml << "joint_limits_ok: " << (win->plan.joint_limits_ok ? "true" : "false") << "\n";
    yaml << "pilz_lin: " << (win->plan.pilz_lin ? "true" : "false") << "\n";
    yaml << "winner_pregrasp_id: " << win->pre.candidate_id << "\n";
    yaml << "winner_a_id: " << win->a.candidate_id << "\n";
    yaml << "winner_b_id: " << win->b.candidate_id << "\n";
    yaml << "winner_c_id: " << win->c.candidate_id << "\n";
    yaml << "winner_a_joints: " << formatHomeList(win->a.joints) << "\n";
    yaml << "winner_b_joints: " << formatHomeList(win->b.joints) << "\n";
    yaml << "winner_c_joints: " << formatHomeList(win->c.joints) << "\n";
    yaml << "scored_trajectory_is_saved_trajectory: " << (identity ? "YES" : "NO") << "\n";
    yaml << "round_trip_ok: " << (roundtrip.ok ? "true" : "false") << "\n";
    yaml << "continuity_ok: " << (continuity.ok ? "true" : "false") << "\n";
    yaml << "endpoint_ok: " << (endpoints.ok ? "true" : "false") << "\n";
  }
  else
  {
    yaml << "best_attempt_a_to_b_j1: "
         << (std::isfinite(min_ab) ? min_ab : -1.0) << "\n";
    yaml << "best_attempt_a_to_b_id: " << min_ab_id << "\n";
    yaml << "best_attempt_b_to_c_j1: "
         << (std::isfinite(min_bc) ? min_bc : -1.0) << "\n";
    yaml << "best_attempt_b_to_c_id: " << min_bc_id << "\n";
    yaml << "b_near_empty: " << (b_near.empty() ? "true" : "false") << "\n";
  }
  yaml << "attempted_combinations:\n";
  for (const auto& c : attempted)
  {
    const auto& m = c.plan.metrics;
    yaml << "  - pre_id: " << c.pre.candidate_id << "\n";
    yaml << "    a_id: " << c.a.candidate_id << "\n";
    yaml << "    b_id: " << c.b.candidate_id << "\n";
    yaml << "    c_id: " << c.c.candidate_id << "\n";
    yaml << "    a_j1: " << j1Of(c.a.joints) << "\n";
    yaml << "    b_j1: " << j1Of(c.b.joints) << "\n";
    yaml << "    c_j1: " << j1Of(c.c.joints) << "\n";
    yaml << "    home_to_pregrasp_j1: " << m.home_to_pregrasp.j1_abs_travel << "\n";
    yaml << "    pregrasp_to_grasp_j1: " << m.pregrasp_to_grasp.j1_abs_travel << "\n";
    yaml << "    grasp_to_lift_j1: " << m.grasp_to_lift.j1_abs_travel << "\n";
    yaml << "    lift_to_a_j1: " << m.lift_to_a.j1_abs_travel << "\n";
    yaml << "    a_to_b_j1: " << m.a_to_b.j1_abs_travel << "\n";
    yaml << "    b_to_c_j1: " << m.b_to_c.j1_abs_travel << "\n";
    yaml << "    full_j1: " << fullJ1Travel(m) << "\n";
    yaml << "    duration: " << m.full_deployable_duration << "\n";
    yaml << "    fail: " << c.fail << "\n";
  }
  yaml << "a_ik_candidates:\n";
  for (const auto& cand : a_ik.unique)
  {
    yaml << "  - id: " << cand.candidate_id << "\n";
    yaml << "    valid: " << (cand.valid ? "true" : "false") << "\n";
    yaml << "    baseline: " << (cand.is_baseline ? "true" : "false") << "\n";
    yaml << "    joints: " << formatHomeList(cand.joints) << "\n";
    yaml << "    j1: " << cand.joints.at("j1") << "\n";
    yaml << "    fk_position_error: " << cand.fk_position_error << "\n";
    yaml << "    fk_orientation_error_deg: " << cand.fk_orientation_error_deg << "\n";
    yaml << "    collision_free: " << (cand.collision_free ? "true" : "false") << "\n";
    yaml << "    attached_object_present: " << (cand.attached_object_present ? "true" : "false")
         << "\n";
    yaml << "    failure_reason: " << cand.failure_reason << "\n";
  }
  yaml << "winner_output_path: " << winner_out << "\n";
  yaml << "trajectory_output_path: " << traj_out << "\n";
  if (!win)
  {
    return 4;
  }
  if (!identity || !roundtrip.ok)
  {
    return 5;
  }
  return 0;
}
