
#include <cmath>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <yaml-cpp/yaml.h>

using MoveGroup =
    moveit::planning_interface::MoveGroupInterface;

using JointMap = std::map<std::string, double>;

// Load the robot model from the EXISTING dual-arm move_group.
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
            rclcpp::ParameterType::PARAMETER_STRING)
            return false;

        if (p.as_string().empty())
            return false;

        node->declare_parameter<std::string>(
            p.get_name(),
            p.as_string());
    }

    return true;
}

// Convert the original j1...j6 into arm_a_j1...arm_a_j6.
JointMap armAJoints(const YAML::Node& values)
{
    if (!values || !values.IsSequence() ||
        values.size() != 6)
    {
        throw std::runtime_error(
            "Expected exactly six joint values");
    }

    JointMap result;

    for (int i = 0; i < 6; ++i)
    {
        double q = values[i].as<double>();

        if (!std::isfinite(q))
            throw std::runtime_error(
                "Non-finite joint value");

        result["arm_a_j" + std::to_string(i + 1)] = q;
    }

    return result;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(
        true);

    auto node = rclcpp::Node::make_shared(
        "dual_arm_a_prefix_plan",
        options);

    RCLCPP_WARN(
        node->get_logger(),
        "PLAN ONLY. ROBOT EXECUTION IS DISABLED.");

    if (!node->has_parameter("trajectory_file"))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Missing trajectory_file parameter");

        rclcpp::shutdown();
        return 1;
    }

    const std::string file =
        node->get_parameter("trajectory_file").as_string();

    YAML::Node yaml;

    try
    {
        yaml = YAML::LoadFile(file);
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "YAML error: %s",
            e.what());

        rclcpp::shutdown();
        return 1;
    }

    // Read the actual STEP14 Home.
    JointMap home;

    // Read the actual STEP14 PreGrasp endpoint.
    JointMap pregrasp;

    try
    {
        home = armAJoints(
            yaml["winner"]["home_joints_rad"]);

        bool found = false;

        for (const auto& segment : yaml["segments"])
        {
            if (segment["name"].as<std::string>() ==
                "Home_to_PreGrasp")
            {
                pregrasp = armAJoints(
                    segment["end_joints"]);

                found = true;
                break;
            }
        }

        if (!found)
            throw std::runtime_error(
                "Home_to_PreGrasp not found");
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Trajectory parsing failed: %s",
            e.what());

        rclcpp::shutdown();
        return 1;
    }

    if (!loadRobotDescription(node))
    {
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    std::thread spinner(
        [&executor]() { executor.spin(); });

    auto stop = [&]()
    {
        executor.cancel();

        if (spinner.joinable())
            spinner.join();

        rclcpp::shutdown();
    };

    // Connect to the existing dual-arm MoveIt.
    MoveGroup arm_a(node, "arm_a");

    arm_a.setPlanningTime(10.0);

    arm_a.setNumPlanningAttempts(5);

    arm_a.setMaxVelocityScalingFactor(0.05);

    arm_a.setMaxAccelerationScalingFactor(0.05);

    auto model = arm_a.getRobotModel();

    if (!model ||
        model->getName() != "fairino3_dual_robot")
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Wrong robot model");

        stop();
        return 1;
    }

    auto current = arm_a.getCurrentState(10.0);

    if (!current)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "No current robot state");

        stop();
        return 1;
    }

    // Important:
    // This state contains BOTH Arm A and Arm B.
    moveit::core::RobotState home_state(*current);

    for (const auto& [name, q] : home)
    {
        home_state.setVariablePosition(name, q);
    }

    home_state.update();

    const auto* arm_a_group =
        model->getJointModelGroup("arm_a");

    if (!arm_a_group ||
        !home_state.satisfiesBounds(arm_a_group))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Arm A Home violates joint limits");

        stop();
        return 1;
    }

    // First plan: Current -> Home.
    arm_a.setStartState(*current);

    if (!arm_a.setJointValueTarget(home))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Invalid Home goal");

        stop();
        return 1;
    }

    MoveGroup::Plan home_plan;

    auto home_result = arm_a.plan(home_plan);

    if (home_result !=
        moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Current -> Home planning failed");

        stop();
        return 2;
    }

    RCLCPP_INFO(
        node->get_logger(),
        "Current -> Home: SUCCESS");

    // Second plan must start from PLANNED Home,
    // not the robot's actual current state.
    arm_a.setStartState(home_state);

    if (!arm_a.setJointValueTarget(pregrasp))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Invalid PreGrasp goal");

        stop();
        return 1;
    }

    MoveGroup::Plan pregrasp_plan;

    auto pregrasp_result =
        arm_a.plan(pregrasp_plan);

    if (pregrasp_result !=
        moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Home -> PreGrasp planning failed");

        stop();
        return 2;
    }

    const auto& trajectory =
        pregrasp_plan.trajectory_
            .joint_trajectory;

    RCLCPP_INFO(
        node->get_logger(),
        "Home -> PreGrasp: SUCCESS");

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

        if (name.rfind("arm_b_", 0) == 0)
        {
            RCLCPP_ERROR(
                node->get_logger(),
                "Unexpected Arm B trajectory!");

            stop();
            return 3;
        }
    }

    RCLCPP_INFO(
        node->get_logger(),
        "DUAL-2 FIRST TEST PASS");

    RCLCPP_INFO(
        node->get_logger(),
        "No execution was performed.");

    stop();

    return 0;
}