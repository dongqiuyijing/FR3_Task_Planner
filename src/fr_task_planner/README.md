# fr_task_planner — Step 2

THIS STEP IS PLAN-ONLY.
NO TRAJECTORY EXECUTION IS PERFORMED.

## Step 2 目标

最小 MTC 集成测试：用现有 FR3 MoveIt 模型做

```text
CurrentState → MoveTo(joint-space)
```

只 `task.plan()`，生成完整 MTC Solution，不执行机械臂。

## 环境

- Ubuntu 22.04
- ROS 2 Humble
- 机器人：FAIRINO FR3，6-DOF
- 主工程（只读）：`~/fairino_ws`
- 本工作空间：`~/fr_task_ws`

## 依赖

已存在，本阶段不安装、不升级：

- `/opt/ros/humble` 中的 `moveit_task_constructor_core` 0.1.3
- `/opt/ros/humble` 中的 `pilz_industrial_motion_planner`（本步不用）
- `~/fairino_ws` 中的 `fairino3_v6_moveit2_config`

## source 顺序

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
```

## 使用的 planning group

从现有 SRDF `fairino3_v6_robot.srdf` 读取，不是猜测：

- planning group: `fairino3_v6_group`
- base frame: `base_link`（模型 frame / SRDF chain base）
- EE/TCP: `gripper_tcp`
- joint names: `j1 j2 j3 j4 j5 j6`

## 使用的 joint names / Home / 测试 goal

Home 从现有只读文件复制：

`~/fairino_ws/src/fr_control/config/stage4_config.yaml`
`robot.initial_joint_positions`

**源单位：degree。节点内转换成 radian 再交给 MTC / MoveIt。**

| joint | Home (deg) | Goal (deg) | Goal (rad) |
| --- | --- | --- | --- |
| j1 | -134.053 | -134.053 | -2.339666 |
| j2 | -123.046 | -123.046 | -2.147546 |
| j3 | -112.585 | -112.585 | -1.964995 |
| j4 | -24.971 | -24.971 | -0.435828 |
| j5 | -34.448 | -34.448 | -0.601231 |
| j6 | 47.587 | 52.587 | 0.917818 |

只改 `j6 + 5 deg`。选择理由：

- URDF `j6` 限位 `+/-3.0543 rad`（约 `+/-175 deg`），52.587 deg 在限位内
- `j6` 是腕部旋转，不改变肩/大臂占用空间
- 侧装 FR3 + 立柱/桌子场景下，不选 `j1/j2/j3` 做 smoke-test

Start 是运行时 `CurrentState`（期望已有 bringup 把机器人放在 Home）。
Goal 是上面的固定 joint-space 目标。Planner 是现有 FR3 MoveIt 默认 OMPL（RRTConnect）。现有 `fairino3_v6_moveit2_config` 没有 `ompl_planning.yaml` 命名 planner config，因此不强制 `RRTConnectkConfigDefault`。

## 编译命令

```bash
cd ~/fr_task_ws
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
colcon build --packages-select fr_task_planner --symlink-install
```

不要全量编译 MTC。

## 启动命令

终端 A — 已有 FR3 MoveIt / RViz 仿真，不要用本 package 重写：

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_control stage4_full.launch.py
```

若只要官方 MoveIt demo（无 Gazebo 工作站）：

```bash
ros2 launch fairino3_v6_moveit2_config demo.launch.py
```

此时把终端 B 的 `use_sim_time:=false`。

终端 B — 本测试：

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_smoke_test.launch.py
```

只跑节点：

```bash
ros2 run fr_task_planner fr3_mtc_smoke_test
```

`ros2 run` 前仍需已有 `move_group`，且建议用 launch 传入 OMPL 参数。

## 预期输出

日志中应看到：

- `THIS STEP IS PLAN-ONLY`
- 成功加载 FR3 RobotModel / `fairino3_v6_group`
- `MTC Task: CurrentState -> MoveTo`
- `planning result: SUCCESS`
- `solution 数量: >= 1`
- 目标 joint state（radian + degree）
- 节点保持运行，供 RViz introspection 查看，不 execute

RViz Motion Planning Tasks 面板应能看到：

```text
FR3 MTC Smoke Test
 ├── CurrentState
 └── MoveTo
```

并可查看规划 trajectory。

## PASS / FAIL

STEP 2 PASS（本机已验证：`colcon build`、加载 `fairino3_v6_robot` / `fairino3_v6_group`、`task.plan()` 得到 1 个完整 solution、RViz 已订阅 `/description` `/statistics` `/solution`、未调用 execute、`~/fairino_ws` 无新增修改）。

本次自动化验证使用已有 `fairino3_v6_moveit2_config/move_group.launch.py`。当时没有 `/joint_states`，因此 CurrentState 记录为全 0；OMPL 仍规划到上面的固定 Home+j6 5° 目标。STEP 3 禁止再用这种方式验收。

## 已知问题

- 本 workspace 没有 MTC 源码；使用 apt 安装的 Humble `moveit_task_constructor_core` 0.1.3
- 节点会从正在运行的 `/move_group` 覆盖 `robot_description` / `robot_description_semantic`，以匹配现有 PlanningScene（含 Gazebo world_to_base）
- 本步不启动夹爪、零件、Pilz LIN、多候选、缓存、真机执行

# Step 3 — Home → PreGrasp（plan-only）

THIS STEP IS PLAN-ONLY.
NO TRAJECTORY EXECUTION IS PERFORMED.

## Step 3 目标

在真实 `stage4_full` 环境中：

```text
真实 /joint_states Home
        ↓
   CurrentState
        ↓
      OMPL
        ↓
Stage4 PreGrasp TCP Pose
```

只规划，不执行。CurrentState 为全 0 时直接 FAIL。

## PreGrasp 来源

不在 C++ 重写抓取几何。launch helper `launch/stage4_pregrasp.py` 只调用：

- `fr_control.stage4_config`
- `fr_control.grasp_poses.compute_grasp_poses`

然后将 `base_link` 下的 Pose 参数传给 `fr3_mtc_pregrasp_test`。

Home 容差：`0.03 rad`。YAML `initial_joint_positions` 单位是 degree，helper 转成 radian。

目标误差沿用 Stage4：

- `motion.position_tolerance` = 0.005 m
- `motion.orientation_tolerance_deg` = 3.0 deg

## 启动命令

终端 A：

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
ros2 launch fr_control stage4_full.launch.py
```

等待 Gazebo、`/joint_states`、`move_group`、PlanningScene 稳定。

终端 B：

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_pregrasp_test.launch.py
```

STEP 2 回归：

```bash
ros2 launch fr_task_planner mtc_fr3_smoke_test.launch.py
```

## STEP 3 已知问题

`stage4_full` 的 `/joint_states` 来自 `fairino3_gazebo/urdf/gz_ros2_control.xacro` 的 `initial_value`（弧度），不是 YAML Home（度）。

实测 Current ≠ Home，最大误差约 3.18 rad。按 STEP 3 规则：不规划、不自动回 Home。

不修改主工程的替代：先用现有 Stage4/MoveIt 把机器人放到 YAML Home，再重新跑本节点。本节点不会发运动命令。

# STEP 4 — Home → PreGrasp → Grasp（plan-only）

THIS STEP IS PLAN-ONLY.
NO TRAJECTORY EXECUTION IS PERFORMED.

STEP 4 validates task-level chained motion planning, not a physically complete grasp.

```text
Task:
Home → PreGrasp → Grasp

OMPL:
Home → PreGrasp

Pilz LIN:
PreGrasp → Grasp

Plan only:
YES

Object collision included:
NO

Gripper close:
NO

Attach:
NO

Lift:
NO
```

一个 MTC Task：

```text
CurrentState
      ↓
MoveTo PreGrasp   (OMPL)
      ↓
MoveTo Grasp      (Pilz LIN)
```

OMPL 终点状态必须直接成为 LIN 起点。禁止拆成两个独立 `task.plan()`。

`small_part` 当前不在 PlanningScene。STEP 5 才加入：

- small_part collision object
- grasp contact policy
- gripper close
- attach object

table / mounting_column 碰撞保持启用。

## 启动命令

终端 A：

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
ros2 launch fr_control stage4_full.launch.py
```

终端 B：

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_grasp_chain_test.launch.py
```

# STEP 5 — Collision-Aware Grasp + Attach + Lift（plan-only）

THIS STEP IS PLAN-ONLY.
NO TRAJECTORY EXECUTION IS PERFORMED.

Physical gripper close: NOT EXECUTED.
Predicted grasp scene transition: YES.

```text
CurrentState
      ↓
Prepare Object On Table
      ↓
MoveTo PreGrasp     (OMPL)
      ↓
Allow Gripper-Part Contact
      ↓
MoveTo Grasp        (Pilz LIN)
      ↓
Attach Part To TCP  (GRASP_COMMIT / predicted attach)
      ↓
MoveTo Lift         (Pilz LIN, world +Z)
      ↓
Restore Part-Table Collision
```

几何、Home、touch links、lift_distance 全部继续来自 `fr_control.stage4_config` + `compute_grasp_poses()`。
`small_part` 只进入 MTC candidate PlanningScene，不永久污染 live `/move_group` scene，也不做 Gazebo weld。

终端 B：

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_attach_lift_test.launch.py
```

# STEP 6 — Cylinder Inspection View Geometry

Geometry only. No Gazebo, no MoveTo Inspection, no view order, no J6 constraint.

P1 is the inspected-region / surface-center target in **world**, not the object
origin and not a `base_link` constant.

Authoritative values live only in
`~/fairino_ws/src/fr_control/config/stage4_config.yaml`:

```text
P1  world [0.0, 0.4, 1.1]
D1  world [0.0, -1.0, 0.0]
up  world [0.0, 0.0, 1.0]
```

STEP 6–11 numbers computed before this correction are **STALE**.
Historical STEP 9 winner `C-B-A 15.5884 s` and STEP 9A winner
`A-B-C trial 4 13.6962 s` are **INVALIDATED BY INSPECTION GEOMETRY CORRECTION**.
Do not treat them as the current fastest order.

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
python3 ~/fr_task_ws/src/fr_task_planner/launch/stage6_inspection_views.py
python3 -m unittest ~/fr_task_ws/src/fr_task_planner/test/test_inspection_view_geometry.py
```

Legacy `stage4_inspection_test.py` still uses old object-center-at-P1 semantics.
New FR3_Task_Planner STEP 6 uses corrected inspection-view-center-at-P1 semantics.
Legacy executor migration is deferred until the new planner geometry is validated.

# STEP 7 — Lift → Individual Inspection View Reachability

Plan-only. One complete grasp chain per view. No order search.

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_inspection_view_test.launch.py view_name:=side_pos_y
ros2 launch fr_task_planner mtc_fr3_inspection_view_test.launch.py view_name:=side_neg_y
ros2 launch fr_task_planner mtc_fr3_inspection_view_test.launch.py view_name:=top_circle
```

# STEP 8 — Pairwise Inspection View Transition Feasibility

```text
Goal:
pairwise canonical inspection transition feasibility

Directed edges:
6

Full prefix included:
YES

Order optimization:
NO

Roll sampling:
NO

Execution:
NO
```

Plan-only. One complete Home→…→source→target Task per directed edge.

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_view_transition_test.launch.py \
  source_view:=side_pos_y target_view:=side_neg_y
python3 ~/fr_task_ws/src/fr_task_planner/launch/stage8_transition_matrix.py
```

# STEP 9 — Full-Task Inspection Order Search

```text
STEP 9
Full canonical inspection-order search.

Permutations:
6

Candidate budget:
configurable, default 5 per order

Ranking:
1. total predicted motion duration
2. within 0.3 s of fastest → minimum total joint path length

Global optimum claim:
NO

Execution:
NO
```

Each permutation is one complete Home→View1→View2→View3 MTC Task.
Do not add STEP 7/8 edge costs. Do not execute.

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_full_order_test.launch.py \
  view_order:=side_pos_y,side_neg_y,top_circle \
  solutions_per_order:=5
python3 ~/fr_task_ws/src/fr_task_planner/launch/stage9_order_search.py
```

# STEP 9A — Canonical Multi-Run Complete-Task Sampling

```text
STEP 9A

Purpose:
prove genuine multiple complete-task sampling under canonical views.

Method:
multiple independent fresh full-task planning trials per order.

Default:
5 trials/order × 6 orders = 30 trials.

No roll sampling.
No execution.
```

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
python3 ~/fr_task_ws/src/fr_task_planner/launch/stage9a_multi_run_search.py
```

# STEP 10 — Inspection Roll / Endpoint Candidate Generation

```text
STEP 10

Purpose:
one Inspection View → many D1-roll Cartesian poses → many collision-checked IK endpoint states.

Hard:
view center = P1
view normal = D1

Soft:
preferred up (recorded, not a rejection gate)

Default:
roll_step_deg=30 → 12 unique poses/view
max_ik_solutions_per_pose=8
min_ik_solution_distance=0.1 rad

Same Lift attached scene for every roll / IK candidate.
No Lift→View path.
No 6-order search.
No motion-time ranking.
No execution.
```

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
python3 -m unittest \
  ~/fr_task_ws/src/fr_task_planner/test/test_inspection_view_geometry.py \
  ~/fr_task_ws/src/fr_task_planner/test/test_inspection_roll_candidates.py
ros2 launch fr_task_planner mtc_fr3_roll_candidate_test.launch.py
python3 ~/fr_task_ws/src/fr_task_planner/launch/stage10_roll_candidates.py
```

# STEP 11 — Integrated Roll×IK Endpoint Branching

```text
STEP 11

Purpose:
one Inspection View, one complete Home→Lift Task, one Alternatives
container of exact joint-goal OMPL branches.

No View→View.
No 6-order search.
No ranking.
No execution.
```

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_endpoint_branch_test.launch.py \
  view_name:=side_pos_y
python3 ~/fr_task_ws/src/fr_task_planner/launch/stage11_endpoint_branching.py
```

# STEP 11A — Correct World-Frame Inspection Geometry

Configuration + frame correction + regression. Not a new planner architecture.

```text
P1/D1/up authority: stage4_config.yaml only
working frame: world
MoveTo goals: converted dynamically to planning_frame (base_link)
setFromIK: T_model_tcp = T_world_base * T_base_tcp
```

Do not re-run STEP 9 / 9A ranking on this correction. Those historical
winners are invalidated. Next combinatorial search is STEP 12.

# STEP 11B — Diagnose Top-Circle Target Collisions

Diagnosis only. `top_circle` may remain unreachable. Do not relax ACM,
change P1/D1/up, or execute.

```bash
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
ros2 launch fr_task_planner mtc_fr3_top_circle_collision_diag.launch.py
python3 ~/fr_task_ws/src/fr_task_planner/launch/stage8_transition_matrix.py
```

Authoritative STEP 8 edge result is `/tmp/fr3_step8_<src>_<tgt>.yaml`
(`result` / `reachable` / `complete_solution_count`), not launch exit code.
Authoritative STEP 11B dump is `/tmp/fr3_step11b_top_circle.yaml`.

# STEP 11C — Simulation Visualization Of The Current Planner

Show what the current planner actually does. Not STEP 12. `top_circle` remains
unreachable and is never executed.

Default sequence (real MTC + OMPL + Pilz, not a handcrafted joint path):

```text
Home → PreGrasp → Grasp → Attach(predicted) → Lift → side_pos_y → side_neg_y
```

`top_circle` is diagnostic only: target markers + colliding IK ghost +
`forearm_link ↔ mounting_column` contact.

Isolated domain: `ROS_DOMAIN_ID=77`. Does not start `real_bringup`. Gazebo
execution is **not implemented**; RViz plays the planned solution.

```bash
cd ~/fr_task_ws
source /opt/ros/humble/setup.bash
source ~/fairino_ws/install/setup.bash
source ~/fr_task_ws/install/setup.bash
export ROS_DOMAIN_ID=77
ros2 launch fr_task_planner mtc_fr3_sim_visualization.launch.py
```

If Stage 4 Gazebo / MoveIt is already running on domain 77:

```bash
export ROS_DOMAIN_ID=77
ros2 launch fr_task_planner mtc_fr3_sim_visualization.launch.py start_stage4:=false visualizer_delay:=0.0
```

`execute_gazebo:=true` is rejected / not implemented. It never sends trajectories
to a real FR3.

RViz MarkerArray: `/fr3_vis/markers`
Collision ghost: `/fr3_vis/collision_robot_state`
MTC animation: `/solution`
Dump: `/tmp/fr3_step11c_visualization.yaml`
