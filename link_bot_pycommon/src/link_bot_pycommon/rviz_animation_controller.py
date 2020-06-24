from time import sleep

import numpy as np

import rospy
from peter_msgs.srv import GetFloat32, GetFloat32Request, GetBool, GetBoolRequest
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Int64
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyRequest



class RvizAnimationController:

    def __init__(self, time_steps, start_playing=True):
        self.epsilon = 1e-3
        self.time_steps = np.array(time_steps, dtype=np.int64)
        self.fwd_sub = rospy.Subscriber("rviz_anim/forward", EmptyMsg, self.on_fwd)
        self.bwd_sub = rospy.Subscriber("rviz_anim/backward", EmptyMsg, self.on_bwd)
        self.play_pause_sub = rospy.Subscriber("rviz_anim/play_pause", EmptyMsg, self.on_play_pause)
        self.done_sub = rospy.Subscriber("rviz_anim/done", EmptyMsg, self.on_done)
        self.period_srv = rospy.ServiceProxy("rviz_anim/period", GetFloat32)
        self.time_pub = rospy.Publisher("rviz_anim/time", Int64, queue_size=10)
        self.max_time_pub = rospy.Publisher("rviz_anim/max_time", Int64, queue_size=10)
        self.auto_play_srv = rospy.ServiceProxy("rviz_anim/auto_play", GetBool)

        rospy.wait_for_service("rviz_anim/period")

        self.idx = 0
        self.max_idx = self.time_steps.shape[0]
        self.max_t = self.time_steps[-1]
        self.period = self.period_srv(GetFloat32Request()).data
        self.auto_play = self.auto_play_srv(GetBoolRequest()).data
        self.playing = self.auto_play
        self.should_step = False
        self.fwd = True
        self.done = False

    def on_fwd(self, msg):
        self.should_step = True
        self.playing = False
        self.fwd = True

    def on_bwd(self, msg):
        self.should_step = True
        self.playing = False
        self.fwd = False

    def on_play_pause(self, msg):
        self.playing = not self.playing
        self.fwd = True

    def step(self):
        if self.playing:
            # don't use ros time because we don't want to rely on simulation time
            sleep(self.period)
        else:
            while not self.should_step and not self.playing and not self.done:
                sleep(0.01)

        if self.fwd:
            if self.idx < self.max_idx - 1:
                self.idx += 1
            else:
                if self.auto_play:
                    self.done = True
                else:
                    self.playing = False
        else:
            if self.idx > 0:
                self.idx -= 1

        t_msg = Int64()
        t_msg.data = self.time_steps[self.idx]
        self.time_pub.publish(t_msg)

        self.should_step = False

        max_t_msg = Int64()
        max_t_msg.data = self.time_steps[-1]
        self.max_time_pub.publish(max_t_msg)

        return self.done

    def on_done(self, msg):
        self.done = True

    def t(self):
        return self.time_steps[self.idx]
