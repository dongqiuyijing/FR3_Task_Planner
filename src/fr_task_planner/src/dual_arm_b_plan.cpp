
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>

using MoveGroup =
    moveit::planning_interface::MoveGroupInterface;

bool loadRobotDescription(
    const rclcpp::Node::SharedPtr& node)
{
    auto client =
        std::make_shared<rclcpp::SyncParametersClient>(
            node, "/move_group");

    if (!client->wait_for_service(
            std::chrono::seconds(10)))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "/move_group unavailable");

        return false;
    }

    auto values = client->get_parameters({
        "robot_description",
        "robot_description_semantic"
    });

    if (values.size() != 2)
        return false;

    for (const auto& p : values)
    {
        if (p.get_type() !=
                rclcpp::ParameterType::PARAMETER_STRING ||
            p.as_string().empty())
        {
            return false;
        }

        node->declare_parameter<std::string>(
            p.get_name(),
            p.as_string());
    }

    return true;
}



int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared(
        "dual_arm_b_plan");

    RCLCPP_INFO(
        node->get_logger(),
        "DUAL-3 PLAN ONLY. NO EXECUTION.");

    if (!loadRobotDescription(node))
    {
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    std::thread spinner(
        [&executor]() {
            executor.spin();
        });

    auto stop = [&]() {
        executor.cancel();

        if (spinner.joinable())
            spinner.join();

        rclcpp::shutdown();
    };

    // 1. Connect to Arm B in the existing MoveIt.
    MoveGroup arm_b(node, "arm_b");

    arm_b.setPlanningTime(10.0);
    arm_b.setNumPlanningAttempts(5);

    arm_b.setMaxVelocityScalingFactor(0.05);
    arm_b.setMaxAccelerationScalingFactor(0.05);

    // 2. Get the COMPLETE dual-arm robot model.
    auto model = arm_b.getRobotModel();

    if (!model ||
        model->getName() != "fairino3_dual_robot")
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Wrong robot model");

        stop();
        return 1;
    }

    const auto* group =
        model->getJointModelGroup("arm_b");

    if (!group)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "arm_b group not found");

        stop();
        return 1;
    }

    // 3. Read the complete current robot state.
    auto current = arm_b.getCurrentState(10.0);

    if (!current)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Current state unavailable");

        stop();
        return 1;
    }

    // 4. Copy BOTH arms.
    moveit::core::RobotState target(*current);

    // 5. Only modify Arm B J6.
    const std::string joint = "arm_b_j6";

    const double current_j6 =
        current->getVariablePosition(joint);

    const double offset = 5.0 * M_PI / 180.0;

    const double target_j6 =
        current_j6 + offset;

    target.setVariablePosition(
        joint, target_j6);

    target.update();

    RCLCPP_INFO(
        node->get_logger(),
        "Arm B J6: %.6f -> %.6f rad",
        current_j6,
        target_j6);

    // 6. Check real joint bounds.
    if (!target.satisfiesBounds(group))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Target violates Arm B joint limits");

        stop();
        return 2;
    }

    // 7. Start from the current COMPLETE
    //    dual-arm state.
    arm_b.setStartState(*current);

    // 8. Only set Arm B joint targets.
    std::map<std::string, double> goal;

    for (int i = 1; i <= 6; ++i)
    {
        std::string name =
            "arm_b_j" + std::to_string(i);

        goal[name] =
            target.getVariablePosition(name);
    }

    if (!arm_b.setJointValueTarget(goal))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Invalid Arm B target");

        stop();
        return 2;
    }

    // 9. Plan only.
    MoveGroup::Plan plan;

    auto result = arm_b.plan(plan);

    if (result !=
        moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Arm B planning failed");

        stop();
        return 3;
    }

    // 10. Inspect the planned trajectory.
    const auto& trajectory =
        plan.trajectory_.joint_trajectory;

    RCLCPP_INFO(
        node->get_logger(),
        "Arm B planning SUCCESS");

    RCLCPP_INFO(
        node->get_logger(),
        "Trajectory points: %zu",
        trajectory.points.size());

    for (const auto& name :
         trajectory.joint_names)
    {
        RCLCPP_INFO(
            node->get_logger(),
            "Planned joint: %s",
            name.c_str());

        if (name.rfind("arm_b_", 0) != 0)
        {
            RCLCPP_ERROR(
                node->get_logger(),
                "Unexpected joint in Arm B trajectory");

            stop();
            return 4;
        }
    }

    RCLCPP_INFO(
        node->get_logger(),
        "DUAL-3 PASS");

    RCLCPP_INFO(
        node->get_logger(),
        "No robot execution performed.");

    stop();

    return 0;
}