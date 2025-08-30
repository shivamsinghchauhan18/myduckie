#!/usr/bin/env python3

"""
Lane Comparator (shadow mode): compares advanced vs neural and fused lane poses.
Publishes differences and periodic summaries for evaluation; does not affect control path.
"""

import rospy
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Float32, Float32MultiArray, String
import numpy as np


class LaneComparator:
    def __init__(self):
        rospy.init_node('lane_comparator', anonymous=True)

        # Latest messages
        self.adv_pose = None
        self.neu_pose = None
        self.fused_pose = None  # Point (converted from PointStamped)

        # Confidence (optional)
        self.neu_conf = None
        self.fused_conf = None

    # Subscribers
    rospy.Subscriber('/lane_follower/advanced/lane_pose', Point, self._adv_cb)
    rospy.Subscriber('/lane_follower/neural/lane_pose', Point, self._neu_cb)
    rospy.Subscriber('/lane_follower/fused_lane_pose', PointStamped, self._fused_cb)
    rospy.Subscriber('/lane_follower/lane_confidence', Float32, self._neu_conf_cb)
    rospy.Subscriber('/lane_follower/fusion_confidence', Float32, self._fused_conf_cb)

        # Publishers
        self.diff_pub = rospy.Publisher('/lane_follower/shadow_diffs', Float32MultiArray, queue_size=1)
        self.status_pub = rospy.Publisher('/lane_follower/shadow_status', String, queue_size=1)

        # Stats
        self.diff_history = []  # list of dicts

        # Timers
        self.tick_timer = rospy.Timer(rospy.Duration(0.1), self._tick)  # 10 Hz
        self.report_timer = rospy.Timer(rospy.Duration(5.0), self._report)  # every 5s

        rospy.loginfo('lane_comparator running (shadow mode)')

    def _adv_cb(self, msg: Point):
        self.adv_pose = msg

    def _neu_cb(self, msg: Point):
        self.neu_pose = msg

    def _fused_cb(self, msg: PointStamped):
        p = Point()
        p.x, p.y, p.z = msg.point.x, msg.point.y, msg.point.z
        self.fused_pose = p

    def _neu_conf_cb(self, msg: Float32):
        try:
            self.neu_conf = msg.data
        except Exception:
            self.neu_conf = None

    def _fused_conf_cb(self, msg: Float32):
        try:
            self.fused_conf = msg.data
        except Exception:
            self.fused_conf = None

    def _tick(self, event):
        # Compute diffs when data available
        adv = self.adv_pose
        diffs = []
        if adv is None:
            return

        # neural vs advanced
        if self.neu_pose is not None:
            dlat_n = float(self.neu_pose.x - adv.x)
            dhead_n = float(self.neu_pose.y - adv.y)
        else:
            dlat_n = np.nan
            dhead_n = np.nan

        # fused vs advanced
        if self.fused_pose is not None:
            dlat_f = float(self.fused_pose.x - adv.x)
            dhead_f = float(self.fused_pose.y - adv.y)
        else:
            dlat_f = np.nan
            dhead_f = np.nan

        diffs = [dlat_n, dhead_n, dlat_f, dhead_f]

        # Publish vector
        arr = Float32MultiArray()
        arr.data = [x if not np.isnan(x) else np.nan for x in diffs]
        self.diff_pub.publish(arr)

        # Record stats (limit size)
        self.diff_history.append(diffs)
        if len(self.diff_history) > 200:
            self.diff_history.pop(0)

    def _report(self, event):
        if not self.diff_history:
            return
        data = np.array([[np.nan if v is None else v for v in row] for row in self.diff_history], dtype=float)
        # Compute nanmean
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        msg = (
            f"Shadow diffs (neural-adv lat,head | fused-adv lat,head): "
            f"mean=({means[0]:.3f},{means[1]:.3f}) | ({means[2]:.3f},{means[3]:.3f}), "
            f"std=({stds[0]:.3f},{stds[1]:.3f}) | ({stds[2]:.3f},{stds[3]:.3f})"
        )
        self.status_pub.publish(String(msg))
        rospy.loginfo_throttle(5, msg)


if __name__ == '__main__':
    try:
        LaneComparator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
