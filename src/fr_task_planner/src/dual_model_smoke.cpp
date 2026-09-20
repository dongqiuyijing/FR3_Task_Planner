
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model/robot_model.h>
#include <moveit/task_constructor/task.h>

#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>

using namespace std::chrono_literals;

bool loadRobotDescription(
    const rclcpp::Node::SharedPtr& node)
{
    auto client =
        std::make_shared<rclcpp::SyncParametersClient>(
            node, "/move_group");

    if (!client->wait_for_service(10s))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Cannot connect to /move_group parameters");

        return false;
    }

    const std::vector<std::string> names = {
        "robot_description",
        "robot_description_semantic"
    };

    auto parameters = client->get_parameters(names);

    for (const auto& parameter : parameters)
    {
        if (parameter.get_type() ==
            rclcpp::ParameterType::PARAMETER_NOT_SET)
        {
            RCLCPP_ERROR(
                node->get_logger(),
                "Missing parameter: %s",
                parameter.get_name().c_str());

            return false;
        }

        node->declare_parameter<std::string>(
            parameter.get_name(),
            parameter.as_string());
    }

    return true;
}

bool checkGroup(
    const moveit::core::RobotModelConstPtr& model,
    const rclcpp::Logger& logger,
    const std::string& group_name,
    const std::string& tcp_name)
{
    const auto* group =
        model->getJointModelGroup(group_name);

    if (!group)
    {
        RCLCPP_ERROR(
            logger,
            "Missing group: %s",
            group_name.c_str());

        return false;
    }

    if (!model->hasLinkModel(tcp_name))
    {
        RCLCPP_ERROR(
            logger,
            "Missing TCP: %s",
            tcp_name.c_str());

        return false;
    }

    RCLCPP_INFO(
        logger,
        "GROUP: %s",
        group_name.c_str());

    for (const auto& name : group->getVariableNames())
    {
        RCLCPP_INFO(
            logger,
            "  joint: %s",
            name.c_str());
    }

    RCLCPP_INFO(
        logger,
        "  TCP: %s",
        tcp_name.c_str());

    return true;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared(
        "dual_model_smoke");

    RCLCPP_INFO(
        node->get_logger(),
        "DUAL-1 READ ONLY: NO ROBOT MOTION");

    if (!loadRobotDescription(node))
    {
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    std::thread spinner(
        [&executor]()
        {
            executor.spin();
        });

    auto stop = [&]()
    {
        executor.cancel();

        if (spinner.joinable())
        {
            spinner.join();
        }

        rclcpp::shutdown();
    };

    // Reuse the robot-model loading method
    // already used in this repository.
    moveit::task_constructor::Task task("", true);

    try
    {
        task.loadRobotModel(node);
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "RobotModel loading failed: %s",
            e.what());

        stop();
        return 1;
    }

    auto model = task.getRobotModel();

    if (!model)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "RobotModel is null");

        stop();
        return 1;
    }

    RCLCPP_INFO(
        node->get_logger(),
        "ROBOT MODEL: %s",
        model->getName().c_str());

    RCLCPP_INFO(
        node->get_logger(),
        "MODEL FRAME: %s",
        model->getModelFrame().c_str());

    if (model->getName() != "fairino3_dual_robot")
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Wrong robot model!");

        stop();
        return 1;
    }

    bool groups_ok = true;

    groups_ok &= checkGroup(
        model,
        node->get_logger(),
        "arm_a",
        "arm_a_gripper_tcp");

    groups_ok &= checkGroup(
        model,
        node->get_logger(),
        "arm_b",
        "arm_b_gripper_tcp");

    const auto* dual_group =
        model->getJointModelGroup("dual_arms");

    if (!dual_group)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Missing dual_arms group");

        groups_ok = false;
    }
    else
    {
        RCLCPP_INFO(
            node->get_logger(),
            "DUAL GROUP VARIABLES: %zu",
            dual_group->getVariableNames().size());

        if (dual_group->getVariableNames().size() != 12)
        {
            groups_ok = false;
        }
    }

    if (!groups_ok)
    {
        stop();
        return 1;
    }

    // Query the EXISTING move_group PlanningScene.
    // Do not create an independent workcell scene.
    auto scene_client =
        node->create_client<
            moveit_msgs::srv::GetPlanningScene>(
                "/get_planning_scene");

    if (!scene_client->wait_for_service(10s))
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "PlanningScene service unavailable");

        stop();
        return 1;
    }

    auto request =
        std::make_shared<
            moveit_msgs::srv::GetPlanningScene::Request>();

    request->components.components =
        moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE |
        moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES;

    auto future =
        scene_client->async_send_request(request);

    if (future.wait_for(10s) !=
        std::future_status::ready)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "PlanningScene request timeout");

        stop();
        return 1;
    }

    auto response = future.get();

    if (!response)
    {
        RCLCPP_ERROR(
            node->get_logger(),
            "Empty PlanningScene response");

        stop();
        return 1;
    }

    const auto& scene = response->scene;

    RCLCPP_INFO(
        node->get_logger(),
        "WORLD OBJECTS:");

    for (const auto& object :
         scene.world.collision_objects)
    {
        RCLCPP_INFO(
            node->get_logger(),
            "  %s",
            object.id.c_str());
    }

    std::map<std::string, double> joints;

    const auto& js =
        scene.robot_state.joint_state;

    const size_t count =
        std::min(js.name.size(), js.position.size());

    for (size_t i = 0; i < count; ++i)
    {
        joints[js.name[i]] = js.position[i];
    }

    RCLCPP_INFO(
        node->get_logger(),
        "CURRENT ARM JOINTS:");

    bool joints_ok = true;

    for (const auto& prefix :
         {"arm_a_", "arm_b_"})
    {
        for (int i = 1; i <= 6; ++i)
        {
            std::string name =
                std::string(prefix) +
                "j" + std::to_string(i);

            auto it = joints.find(name);

            if (it == joints.end())
            {
                RCLCPP_ERROR(
                    node->get_logger(),
                    "Missing joint: %s",
                    name.c_str());

                joints_ok = false;
                continue;
            }

            RCLCPP_INFO(
                node->get_logger(),
                "%s = %.6f rad",
                name.c_str(),
                it->second);
        }
    }

    if (joints_ok)
    {
        RCLCPP_INFO(
            node->get_logger(),
            "DUAL-1 PASS: Both arms loaded");
    }

    stop();

    return joints_ok ? 0 : 1;
}