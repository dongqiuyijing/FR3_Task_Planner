// Independent, deliberately narrow real-hardware handover alignment test.
// It never opens a gripper, transfers an attachment, retreats, homes, or runs inspection.
#include "fr_task_planner/frozen_executor.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <yaml-cpp/yaml.h>
using FJT = control_msgs::action::FollowJointTrajectory;
using MG = moveit_msgs::action::MoveGroup;
namespace { const std::vector<std::string> A={"arm_a_j1","arm_a_j2","arm_a_j3","arm_a_j4","arm_a_j5","arm_a_j6"}; const std::vector<std::string> B={"arm_b_j1","arm_b_j2","arm_b_j3","arm_b_j4","arm_b_j5","arm_b_j6"};
bool vec(const YAML::Node& n,std::vector<double>& v){if(!n||!n.IsSequence()||n.size()!=6)return false;v.clear();for(auto x:n)v.push_back(x.as<double>());return true;}
std::vector<double> get(const std::map<std::string,double>& m,const std::vector<std::string>& n){std::vector<double> v;for(auto&s:n){auto i=m.find(s);if(i==m.end())return {};v.push_back(i->second);}return v;}
double err(const std::vector<double>&a,const std::vector<double>&b){double e=0;for(size_t i=0;i<a.size();++i)e=std::max(e,std::abs(a[i]-b[i]));return e;}
}
class Test { public: explicit Test(rclcpp::Node::SharedPtr n):n_(std::move(n)){}
int run(){
 const auto home=std::string(std::getenv("HOME")?std::getenv("HOME"):""); auto y=YAML::LoadFile(home+"/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/b_handover_live_test.yaml");
 if(!vec(y["arm_a_handover_joints"],qa_)||!vec(y["arm_b_pre_handover_joints"],qbpre_)||!vec(y["arm_b_handover_joints"],qb_)){RCLCPP_ERROR(n_->get_logger(),"invalid target YAML");return 2;}
 auto bool_param=[this](const char* key,bool fallback){if(!n_->has_parameter(key))return n_->declare_parameter<bool>(key,fallback);return n_->get_parameter(key).as_bool();};
 auto string_param=[this](const char* key,const std::string& fallback){if(!n_->has_parameter(key))return n_->declare_parameter<std::string>(key,fallback);return n_->get_parameter(key).as_string();};
 execute_=bool_param("execute",false); auto token=string_param("real_robot_confirmation",""); armed_=execute_&&token==fr_task_planner::kRealRobotConfirmation;
 final_token_=string_param("final_approach_confirmation",""); v_=y["approach_velocity_scale"].as<double>(); tol_=y["start_tolerance_rad"].as<double>();
 sub_=n_->create_subscription<sensor_msgs::msg::JointState>("/joint_states",rclcpp::SensorDataQoS(),[this](sensor_msgs::msg::JointState::SharedPtr m){for(size_t i=0;i<m->name.size()&&i<m->position.size();++i) js_[m->name[i]]=m->position[i];});
 mg_=rclcpp_action::create_client<MG>(n_,"move_action"); ca_=rclcpp_action::create_client<FJT>(n_,y["arm_a_trajectory_action"].as<std::string>()); cb_=rclcpp_action::create_client<FJT>(n_,y["arm_b_trajectory_action"].as<std::string>());
 confirm_=n_->create_service<std_srvs::srv::SetBool>("confirm_final_b_handover",[this](std::shared_ptr<std_srvs::srv::SetBool::Request> r,std::shared_ptr<std_srvs::srv::SetBool::Response> s){approved_=r->data;s->success=r->data;s->message=r->data?"FINAL_APPROACH_APPROVED":"FINAL_APPROACH_REJECTED";});
 RCLCPP_INFO(n_->get_logger(),"B_HANDOVER LIVE TEST mode=%s; gripper commands=0; B TCP compensation world Y=-6mm Z=-10mm",armed_?"EXECUTE_AFTER_GATES":"PLAN_ONLY");
 if(!waitLive()) return 1; auto a=get(js_,A),b=get(js_,B); if(a.empty()||b.empty())return 1;
 if(!plan("arm_a",A,a,b,qa_,pa_))return 1; if(!plan("arm_b",B,qa_,b,qbpre_,pp_))return 1; if(!plan("arm_b",B,qa_,qbpre_,qb_,pf_))return 1;
 RCLCPP_INFO(n_->get_logger(),"PLAN PASS A(current->A_HANDOVER), B(current->B_PRE), B_PRE->OFFSET_B_HANDOVER. final target=[%.9f %.9f %.9f %.9f %.9f %.9f]",qb_[0],qb_[1],qb_[2],qb_[3],qb_[4],qb_[5]);
 if(!armed_){RCLCPP_INFO(n_->get_logger(),"PLAN_ONLY complete: no arm or gripper command sent");return 0;}
 if(!send("arm_a",ca_,pa_,qa_))return 1; if(!send("arm_b",cb_,pp_,qbpre_))return 1;
 RCLCPP_WARN(n_->get_logger(),"PAUSED BEFORE FINAL B APPROACH. Call /confirm_final_b_handover SetBool true after visual check.");
 while(rclcpp::ok()&&!approved_){rclcpp::spin_some(n_);std::this_thread::sleep_for(std::chrono::milliseconds(100));} if(final_token_!="I_CONFIRM_FINAL_B_HANDOVER_APPROACH" ){RCLCPP_ERROR(n_->get_logger(),"final_approach_confirmation token missing; final motion refused");return 1;}
 if(!send("arm_b",cb_,pf_,qb_))return 1; RCLCPP_INFO(n_->get_logger(),"TEST COMPLETE: holding positions; no gripper command, release, retreat, Home, or inspection sent.");return 0;
 }
 private:
 bool waitLive(){for(int i=0;i<50&&rclcpp::ok();++i){rclcpp::spin_some(n_);if(get(js_,A).size()==6&&get(js_,B).size()==6)return true;std::this_thread::sleep_for(std::chrono::milliseconds(100));}RCLCPP_ERROR(n_->get_logger(),"missing named /joint_states");return false;}
 bool plan(const std::string& group,const std::vector<std::string>& names,const std::vector<double>& a,const std::vector<double>& b,const std::vector<double>& goal,trajectory_msgs::msg::JointTrajectory& out){
 if(!mg_->wait_for_action_server(std::chrono::seconds(5))){RCLCPP_ERROR(n_->get_logger(),"move_action unavailable");return false;} MG::Goal g;g.request.group_name=group;g.request.num_planning_attempts=5;g.request.allowed_planning_time=10;g.request.max_velocity_scaling_factor=v_;g.request.max_acceleration_scaling_factor=v_;g.request.start_state.is_diff=false;g.request.start_state.joint_state.name=A;g.request.start_state.joint_state.name.insert(g.request.start_state.joint_state.name.end(),B.begin(),B.end());g.request.start_state.joint_state.position=a;g.request.start_state.joint_state.position.insert(g.request.start_state.joint_state.position.end(),b.begin(),b.end());moveit_msgs::msg::Constraints c;for(size_t i=0;i<6;++i){moveit_msgs::msg::JointConstraint j;j.joint_name=names[i];j.position=goal[i];j.tolerance_above=j.tolerance_below=.001;j.weight=1;c.joint_constraints.push_back(j);}g.request.goal_constraints.push_back(c);g.planning_options.plan_only=true;auto f=mg_->async_send_goal(g);if(rclcpp::spin_until_future_complete(n_,f,std::chrono::seconds(15))!=rclcpp::FutureReturnCode::SUCCESS||!f.get())return false;auto r=mg_->async_get_result(f.get());if(rclcpp::spin_until_future_complete(n_,r,std::chrono::seconds(30))!=rclcpp::FutureReturnCode::SUCCESS||r.get().code!=rclcpp_action::ResultCode::SUCCEEDED)return false;out=r.get().result->planned_trajectory.joint_trajectory;if(out.points.empty()){RCLCPP_ERROR(n_->get_logger(),"empty plan");return false;}RCLCPP_INFO(n_->get_logger(),"%s collision-checked plan points=%zu duration=%.3f velocity_scale=%.3f",group.c_str(),out.points.size(),out.points.back().time_from_start.sec+1e-9*out.points.back().time_from_start.nanosec,v_);return true;}
 bool send(const std::string& arm,rclcpp_action::Client<FJT>::SharedPtr c,const trajectory_msgs::msg::JointTrajectory& t,const std::vector<double>& target){
  const auto& names=arm=="arm_a"?A:B; auto now=get(js_,names);
  if(now.size()!=6){RCLCPP_ERROR(n_->get_logger(),"%s live joint state unavailable",arm.c_str());return false;}
  if(err(now,target)<=tol_){RCLCPP_INFO(n_->get_logger(),"%s already at target (err=%.6f); skip zero-duration command",arm.c_str(),err(now,target));return true;}
  trajectory_msgs::msg::JointTrajectory remap; remap.joint_names=names;
  for(const auto& p:t.points){trajectory_msgs::msg::JointTrajectoryPoint q; q.time_from_start=p.time_from_start; for(const auto& name:names){auto it=std::find(t.joint_names.begin(),t.joint_names.end(),name);if(it==t.joint_names.end()){RCLCPP_ERROR(n_->get_logger(),"%s plan lacks joint %s",arm.c_str(),name.c_str());return false;}auto idx=static_cast<size_t>(std::distance(t.joint_names.begin(),it));if(idx>=p.positions.size())return false;q.positions.push_back(p.positions[idx]);}remap.points.push_back(q);}
  if(remap.points.empty()||err(now,remap.points.front().positions)>tol_){RCLCPP_ERROR(n_->get_logger(),"%s start state changed; refusing stale plan err=%.6f",arm.c_str(),err(now,remap.points.empty()?target:remap.points.front().positions));return false;}
  if(!c->wait_for_action_server(std::chrono::seconds(5))){RCLCPP_ERROR(n_->get_logger(),"%s FollowJointTrajectory action unavailable",arm.c_str());return false;}
  FJT::Goal g;g.trajectory=remap;g.trajectory.header.stamp=n_->now();RCLCPP_INFO(n_->get_logger(),"sending %s points=%zu duration=%.3f",arm.c_str(),remap.points.size(),remap.points.back().time_from_start.sec+1e-9*remap.points.back().time_from_start.nanosec);auto f=c->async_send_goal(g);
  if(rclcpp::spin_until_future_complete(n_,f,std::chrono::seconds(15))!=rclcpp::FutureReturnCode::SUCCESS||!f.get()){RCLCPP_ERROR(n_->get_logger(),"%s trajectory goal rejected/timed out",arm.c_str());return false;}auto r=c->async_get_result(f.get());
  if(rclcpp::spin_until_future_complete(n_,r,std::chrono::seconds(60))!=rclcpp::FutureReturnCode::SUCCESS||r.get().code!=rclcpp_action::ResultCode::SUCCEEDED){RCLCPP_ERROR(n_->get_logger(),"%s trajectory result failed/timed out",arm.c_str());return false;}RCLCPP_INFO(n_->get_logger(),"%s executed target",arm.c_str());return true;}
 rclcpp::Node::SharedPtr n_;std::map<std::string,double> js_;std::vector<double>qa_,qbpre_,qb_;double v_=0.1,tol_=0.02;bool execute_=false,armed_=false,approved_=false;std::string final_token_;rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;rclcpp_action::Client<MG>::SharedPtr mg_;rclcpp_action::Client<FJT>::SharedPtr ca_,cb_;rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr confirm_;trajectory_msgs::msg::JointTrajectory pa_,pp_,pf_;};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::NodeOptions o;o.automatically_declare_parameters_from_overrides(true);auto n=rclcpp::Node::make_shared("keypose_v1_b_handover_live_test",o);Test t(n);auto rc=t.run();rclcpp::shutdown();return rc;}
