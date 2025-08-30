#!/usr/bin/env python3

"""
Lane topic multiplexer: selects between advanced, neural, or fused sources and republishes
canonical topics consumed by controllers. Defaults to advanced to preserve current behavior.
"""

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger, TriggerResponse


class LaneMux:
    def __init__(self):
        rospy.init_node('lane_mux', anonymous=True)

        # Params
        self.lane_source = rospy.get_param('~lane_source', 'advanced')  # advanced|neural|fused
        self.robot_name = rospy.get_param('~robot_name', 'blueduckie')

        # Output publishers (canonical topics)
        self.pub_lane_pose = rospy.Publisher('/lane_follower/lane_pose', Point, queue_size=1)
        self.pub_lane_found = rospy.Publisher('/lane_follower/lane_found', Bool, queue_size=1)
        self.pub_lane_center = rospy.Publisher('/lane_follower/lane_center', Point, queue_size=1)
        self.pub_lane_angle = rospy.Publisher('/lane_follower/lane_angle', Float32, queue_size=1)

        # State for latest messages from each source
        self.latest = {
            'advanced': {'pose': None, 'found': None, 'center': None, 'angle': None},
            'neural': {'pose': None, 'found': None, 'center': None, 'angle': None},
            'fused': {'pose': None}  # fused provides PointStamped pose only
        }

        # Subscriptions for advanced (remapped by launch to .advanced)
        rospy.Subscriber('/lane_follower/lane_pose.advanced', Point, self._mk_cb('advanced', 'pose'))
        rospy.Subscriber('/lane_follower/lane_found.advanced', Bool, self._mk_cb('advanced', 'found'))
        rospy.Subscriber('/lane_follower/lane_center.advanced', Point, self._mk_cb('advanced', 'center'))
        rospy.Subscriber('/lane_follower/lane_angle.advanced', Float32, self._mk_cb('advanced', 'angle'))

        # Subscriptions for neural (remapped by launch to .neural)
        rospy.Subscriber('/lane_follower/lane_pose.neural', Point, self._mk_cb('neural', 'pose'))
        rospy.Subscriber('/lane_follower/lane_found.neural', Bool, self._mk_cb('neural', 'found'))
        rospy.Subscriber('/lane_follower/lane_center.neural', Point, self._mk_cb('neural', 'center'))
        rospy.Subscriber('/lane_follower/lane_angle.neural', Float32, self._mk_cb('neural', 'angle'))

        # Subscriptions for fused (PointStamped)
        rospy.Subscriber('/lane_follower/fused_lane_pose', PointStamped, self._fused_pose_cb)

        # Timer to republish selected source
        self._timer = rospy.Timer(rospy.Duration(0.05), self._publish_selected)  # 20 Hz

    # Runtime services to switch source without restart
    self._srv_adv = rospy.Service('~use_advanced', Trigger, self._srv_use_advanced)
    self._srv_neu = rospy.Service('~use_neural', Trigger, self._srv_use_neural)
    self._srv_fus = rospy.Service('~use_fused', Trigger, self._srv_use_fused)

        rospy.loginfo("lane_mux running - source=%s", self.lane_source)

    def _mk_cb(self, src, field):
        def _cb(msg):
            self.latest[src][field] = msg
        return _cb

    def _fused_pose_cb(self, msg: PointStamped):
        # Convert to Point for internal storage
        p = Point()
        p.x = msg.point.x
        p.y = msg.point.y
        p.z = msg.point.z
        self.latest['fused']['pose'] = p

    def _publish_selected(self, event):
        # Allow dynamic param updates
        self.lane_source = rospy.get_param('~lane_source', self.lane_source)

        if self.lane_source == 'neural':
            src = 'neural'
        elif self.lane_source == 'fused':
            src = 'fused'
        else:
            src = 'advanced'

        # Prepare messages
        if src in ('advanced', 'neural'):
            pose = self.latest[src]['pose']
            found = self.latest[src]['found']
            center = self.latest[src]['center']
            angle = self.latest[src]['angle']

            if pose is not None:
                self.pub_lane_pose.publish(pose)
                # Derive defaults if some fields missing
                if found is None:
                    self.pub_lane_found.publish(Bool(pose.z > 0.5))
                else:
                    self.pub_lane_found.publish(found)
                if center is None:
                    c = Point(); c.x = pose.x; c.y = 0.0; c.z = 0.0
                    self.pub_lane_center.publish(c)
                else:
                    self.pub_lane_center.publish(center)
                if angle is None:
                    self.pub_lane_angle.publish(Float32(pose.y))
                else:
                    self.pub_lane_angle.publish(angle)
        else:
            # fused: only have pose
            pose = self.latest['fused'].get('pose')
            if pose is not None:
                self.pub_lane_pose.publish(pose)
                self.pub_lane_found.publish(Bool(pose.z > 0.5))
                c = Point(); c.x = pose.x; c.y = 0.0; c.z = 0.0
                self.pub_lane_center.publish(c)
                self.pub_lane_angle.publish(Float32(pose.y))

    def _srv_use_advanced(self, req):
        try:
            rospy.set_param('~lane_source', 'advanced')
            self.lane_source = 'advanced'
            return TriggerResponse(success=True, message='lane_source=advanced')
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def _srv_use_neural(self, req):
        try:
            rospy.set_param('~lane_source', 'neural')
            self.lane_source = 'neural'
            return TriggerResponse(success=True, message='lane_source=neural')
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def _srv_use_fused(self, req):
        try:
            rospy.set_param('~lane_source', 'fused')
            self.lane_source = 'fused'
            return TriggerResponse(success=True, message='lane_source=fused')
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))


if __name__ == '__main__':
    try:
        LaneMux()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
