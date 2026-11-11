"""Capture one timestamp-matched set of every published stream for README figures.

Usage (with the simulation running):  python3 tools/capture_frames.py OUT.npz [--seconds 20]
"""
import sys, time, json, argparse
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--seconds', type=float, default=20)
ap.add_argument('--skip', type=int, default=8, help='complete sets to skip so the stream is settled')
a = ap.parse_args()
rclpy.init(); node = rclpy.create_node('capture_d435i_frames')
topics = {'rgb': ('/camera/camera/color/image_raw', Image),
          'depth': ('/camera/camera/depth/image_rect_raw', Image),
          'aligned': ('/camera/camera/aligned_depth_to_color/image_raw', Image),
          'truth': ('/sim/ground_truth/depth/image_rect_raw', Image),
          'points': ('/camera/camera/depth/color/points', PointCloud2)}
by_stamp = {}
def cb(key, m):
    t = (m.header.stamp.sec, m.header.stamp.nanosec)
    by_stamp.setdefault(t, {})[key] = m
subs = [node.create_subscription(typ, topic, lambda m, k=k: cb(k, m), qos_profile_sensor_data) for k, (topic, typ) in topics.items()]
def image(m):
    dt = {'rgb8': np.uint8, '16UC1': '<u2', '32FC1': '<f4'}[m.encoding]
    arr = np.frombuffer(m.data, dtype=dt)
    return arr.reshape(m.height, m.width, 3) if m.encoding == 'rgb8' else arr.reshape(m.height, m.width)
start = time.monotonic(); done = None; complete = 0
while time.monotonic() - start < a.seconds and done is None:
    rclpy.spin_once(node, timeout_sec=0.1)
    for t in sorted(by_stamp):
        if len(by_stamp[t]) == len(topics):
            complete += 1
            if complete > a.skip: done = t; break
            by_stamp.pop(t); break
if done is None:
    sys.exit('no complete timestamp-matched set within %.0f s' % a.seconds)
s = by_stamp[done]
pc = s['points']
cloud = np.frombuffer(pc.data, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')]).reshape(pc.height, pc.width)
np.savez_compressed(a.out, rgb=image(s['rgb']), depth_mm=image(s['depth']), aligned_mm=image(s['aligned']),
                    truth_m=image(s['truth']), cloud_xyz=np.stack([cloud['x'], cloud['y'], cloud['z']], -1),
                    cloud_rgb=cloud['rgb'], stamp=np.array(done))
print('CAPTURED', json.dumps({'stamp': done, 'cloud_shape': [pc.height, pc.width]}))
node.destroy_node(); rclpy.shutdown()
