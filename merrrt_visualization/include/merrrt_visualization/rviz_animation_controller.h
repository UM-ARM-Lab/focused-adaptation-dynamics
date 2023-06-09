#pragma once

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <rviz/panel.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int64.h>

#include <QObject>
#include <QPropertyAnimation>
#include <QWidget>
#include <thread>

#include "ui_animation_controller.h"

namespace merrrt_visualization {
class RVizAnimationController : public rviz::Panel {
  Q_OBJECT

 public:
  explicit RVizAnimationController(QWidget *parent = nullptr);

  virtual ~RVizAnimationController();

  void load(const rviz::Config &config) override;
  void save(rviz::Config config) const override;

  void TimeCallback(const std_msgs::Int64::ConstPtr &msg);

  void MaxTimeCallback(const std_msgs::Int64::ConstPtr &msg);

 public slots:

  void ForwardClicked();

  void BackwardClicked();

  void PauseClicked();

  void PlayForwardClicked();

  void PlayBackwardClicked();

  void DoneClicked();

  void LoopToggled();

  void AutoNextToggled();

  void AutoPlayToggled();

  void StepNumberChanged();

  void PeriodChanged(double period);

  void TopicEdited(const QString& text);

 signals:
  void setStepText(const QString& text);
  void setMaxText(const QString& text);

 private:
  void QueueThread();

  Ui_MainWidget ui;
  ros::NodeHandle ros_node_;
  std::optional<ros::Publisher> command_pub_;
  ros::ServiceServer get_state_srv_;
  ros::Subscriber time_sub_;
  ros::Subscriber max_time_sub_;

  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;

  QPropertyAnimation *highlight_animation_;
};

}  // namespace merrrt_visualization