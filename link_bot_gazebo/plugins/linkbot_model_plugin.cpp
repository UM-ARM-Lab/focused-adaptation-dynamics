#include "linkbot_model_plugin.h"

#include <ignition/math/Vector3.hh>
#include <memory>

namespace gazebo {

    void LinkBotModelPlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf) {
        // Make sure the ROS node for Gazebo has already been initalized
        // Initialize ros, if it has not already bee initialized.
        if (!ros::isInitialized()) {
            auto argc = 0;
            char **argv = nullptr;
            ros::init(argc, argv, "linkbot_model_plugin", ros::init_options::NoSigintHandler);
        }

        ros_node_ = std::make_unique<ros::NodeHandle>("linkbot_model_plugin");

        auto action_bind = boost::bind(&LinkBotModelPlugin::OnAction, this, _1);
        auto action_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotAction>("/link_bot_action", 1,
                                                                                       action_bind, ros::VoidPtr(),
                                                                                       &queue_);
        auto config_bind = boost::bind(&LinkBotModelPlugin::OnConfiguration, this, _1);
        auto config_so = ros::SubscribeOptions::create<link_bot_gazebo::LinkBotConfiguration>("/link_bot_configuration",
                                                                                              1, config_bind,
                                                                                              ros::VoidPtr(), &queue_);
        cmd_sub_ = ros_node_->subscribe(action_so);
        config_sub_ = ros_node_->subscribe(config_so);
        ros_queue_thread_ = std::thread(std::bind(&LinkBotModelPlugin::QueueThread, this));

        if (!sdf->HasElement("kP")) {
            printf("using default kP=%f\n", kP_);
        } else {
            kP_ = sdf->GetElement("kP")->Get<double>();
        }

        if (!sdf->HasElement("kI")) {
            printf("using default kI=%f\n", kI_);
        } else {
            kI_ = sdf->GetElement("kI")->Get<double>();
        }

        if (!sdf->HasElement("kD")) {
            printf("using default kD=%f\n", kD_);
        } else {
            kD_ = sdf->GetElement("kD")->Get<double>();
        }

        if (!sdf->HasElement("action_scale")) {
            printf("using default action_scale=%f\n", action_scale);
        } else {
            action_scale = sdf->GetElement("action_scale")->Get<double>();
        }

        if (!sdf->HasElement("control_link_name")) {
            printf("using default control_link_name=%s\n", control_link_name_.c_str());
        } else {
            control_link_name_ = sdf->GetElement("control_link_name")->Get<std::string>();
        }

        ROS_INFO("kP=%f, kI=%f, kD=%f", kP_, kI_, kD_);

        model_ = parent;

        control_link_ = model_->GetLink(control_link_name_);
        if (not control_link_) {
            std::cout << "invalid link pointer. Link name " << control_link_name_ << " is not one of:\n";
            for (auto const& link : model_->GetLinks()) {
               std::cout << link->GetName() << "\n";
            }
        }

        updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&LinkBotModelPlugin::OnUpdate, this));
        x_vel_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);
        y_vel_pid_ = common::PID(kP_, kI_, kD_, 100, -100, 800, -800);
    }

    void LinkBotModelPlugin::OnUpdate() {
        if (use_force_) {
            control_link_->AddForce(target_force_);
        } else {
            auto const current_linear_vel = control_link_->WorldLinearVel();
            auto const error = current_linear_vel - target_linear_vel_;
            ignition::math::Vector3d force{};
            force.X(x_vel_pid_.Update(error.X(), 0.001));
            force.Y(y_vel_pid_.Update(error.Y(), 0.001));
            control_link_->AddForce(force);
        }
    }

    void LinkBotModelPlugin::OnAction(link_bot_gazebo::LinkBotActionConstPtr msg) {
        control_link_name_ = msg->control_link_name;
        control_link_ = model_->GetLink(control_link_name_);
        if (not control_link_) {
            std::cout << "invalid link pointer. Link name " << control_link_name_ << " is not one of:\n";
            for (auto const link : model_->GetLinks()) {
                std::cout << link->GetName() << "\n";
            }
            return;
        }
        use_force_ = msg->use_force;
        if (msg->use_force) {
            target_force_.X(msg->wrench.force.x * action_scale);
            target_force_.Y(msg->wrench.force.y * action_scale);
        }
        else {
            target_linear_vel_.X(msg->twist.linear.x * action_scale);
            target_linear_vel_.Y(msg->twist.linear.y * action_scale);
        }
    }

    void LinkBotModelPlugin::OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr _msg) {
        auto const &joints = model_->GetJoints();

        if (joints.size() != _msg->joint_angles_rad.size()) {
            ROS_ERROR("Model as %lu joints config message had %lu", joints.size(), _msg->joint_angles_rad.size());
            return;
        }

        ignition::math::Pose3d pose{};
        pose.Pos().X(_msg->tail_pose.x);
        pose.Pos().Y(_msg->tail_pose.y);
        pose.Pos().Z(0.1);
        pose.Rot() = ignition::math::Quaterniond::EulerToQuaternion(0, 0, _msg->tail_pose.theta);
        model_->SetWorldPose(pose);
        model_->SetWorldTwist({0, 0, 0}, {0, 0, 0});

        for (size_t i = 0; i < joints.size(); ++i) {
            auto const &joint = joints[i];
            joint->SetPosition(0, _msg->joint_angles_rad[i]);
            joint->SetVelocity(0, 0);
        }
    }

    void LinkBotModelPlugin::QueueThread() {
        double constexpr timeout = 0.01;
        while (ros_node_->ok()) {
            queue_.callAvailable(ros::WallDuration(timeout));
        }
    }

    LinkBotModelPlugin::~LinkBotModelPlugin() {
        queue_.clear();
        queue_.disable();
        ros_node_->shutdown();
        ros_queue_thread_.join();
    }

    // Register this plugin with the simulator
    GZ_REGISTER_MODEL_PLUGIN(LinkBotModelPlugin)
}
