#!/usr/bin/env python3

"""
Command multiplexer: selects between PID (current pipeline) and MPC control outputs
to publish on /car_cmd_switch_node/cmd. Defaults to pid to preserve current behavior.
"""

import rospy
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


class CmdMux:
    def __init__(self):
        rospy.init_node('cmd_mux', anonymous=True)
        self.source = rospy.get_param('~cmd_source', 'pid')  # pid|mpc
        self.robot_name = rospy.get_param('~robot_name', 'blueduckie')

        # Output
        self.pub_cmd = rospy.Publisher('/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        self.pub_info = rospy.Publisher('/lane_follower/cmd_mux_info', String, queue_size=1)

    # Inputs (remapped by launch):
    rospy.Subscriber('/car_cmd_switch_node/pid/cmd', Twist2DStamped, self._pid_cb)
    rospy.Subscriber('/car_cmd_switch_node/mpc/cmd', Twist2DStamped, self._mpc_cb)

        self.last_pid = None
        self.last_mpc = None

        self.timer = rospy.Timer(rospy.Duration(0.02), self._tick)  # 50 Hz
        rospy.loginfo("cmd_mux running - source=%s", self.source)

    # Runtime services to switch source
    self._srv_pid = rospy.Service('~use_pid', Trigger, self._srv_use_pid)
    self._srv_mpc = rospy.Service('~use_mpc', Trigger, self._srv_use_mpc)

    def _pid_cb(self, msg):
        self.last_pid = msg

    def _mpc_cb(self, msg):
        self.last_mpc = msg

    def _tick(self, event):
        # Allow dynamic updates
        self.source = rospy.get_param('~cmd_source', self.source)

        msg = None
        if self.source == 'mpc':
            msg = self.last_mpc or self.last_pid
        else:
            msg = self.last_pid or self.last_mpc

        if msg is not None:
            self.pub_cmd.publish(msg)

        # Throttled info
        self.pub_info.publish(String(f"cmd_mux source={self.source}"))

    def _srv_use_pid(self, req):
        try:
            rospy.set_param('~cmd_source', 'pid')
            self.source = 'pid'
            return TriggerResponse(success=True, message='cmd_source=pid')
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def _srv_use_mpc(self, req):
        try:
            rospy.set_param('~cmd_source', 'mpc')
            self.source = 'mpc'
            return TriggerResponse(success=True, message='cmd_source=mpc')
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))


if __name__ == '__main__':
    try:
        CmdMux()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
