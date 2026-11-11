import json,time,hashlib
from pathlib import Path
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
from sensor_msgs.msg import Image,CameraInfo,PointCloud2
from tf2_msgs.msg import TFMessage
from d435i_model import intrinsics
root=Path(__file__).resolve().parent;p=json.loads((root/'d435i_profile.json').read_text());W,H=p['width'],p['height'];kd=intrinsics(p)['depth']
rclpy.init();node=rclpy.create_node('validate_d435i_sim');seen={};times={};subs=[];stamps={};hashes={};history={}
def cb(key,m):
 seen[key]=m;times.setdefault(key,[]).append(time.monotonic())
 if hasattr(m,'header'):
  t=(m.header.stamp.sec,m.header.stamp.nanosec);stamps.setdefault(key,set()).add(t)
  hist=history.setdefault(key,{});hist[t]=m
  while len(hist)>8:hist.pop(next(iter(hist)))
 if key in ['rgb','depth']:hashes.setdefault(key,set()).add(hashlib.blake2b(m.data,digest_size=8).hexdigest())
for key,topic,typ in [('rgb','color/image_raw',Image),('depth','depth/image_rect_raw',Image),('aligned','aligned_depth_to_color/image_raw',Image),('ci','color/camera_info',CameraInfo),('di','depth/camera_info',CameraInfo),('ai','aligned_depth_to_color/camera_info',CameraInfo),('points','depth/color/points',PointCloud2)]:
 subs.append(node.create_subscription(typ,'/camera/camera/'+topic,lambda m,k=key:cb(k,m),qos_profile_sensor_data))
subs.append(node.create_subscription(Image,'/sim/ground_truth/depth/image_rect_raw',lambda m:cb('truth',m),qos_profile_sensor_data))
subs.append(node.create_subscription(TFMessage,'/tf_static',lambda m:cb('tf',m),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)))
start=time.monotonic()
while time.monotonic()-start<15:rclpy.spin_once(node,timeout_sec=.1)
out={'profile':p['name'],'topics_received':sorted(seen),'rates_hz':{k:round((len(v)-1)/(v[-1]-v[0]),2) for k,v in times.items() if len(v)>1},'checks':{}}
c=out['checks'];c['all_streams_received']=all(k in seen for k in ['rgb','depth','aligned','ci','di','ai','points','truth','tf'])
if 'depth' in seen:
 m=seen['depth'];z=np.frombuffer(m.data,dtype='<u2').reshape(m.height,m.width);cy,cx=H//2,W//2;patch=z[cy-15:cy+15,cx-15:cx+15];validpatch=patch[patch>0]
 out['depth']={'encoding':m.encoding,'shape':[m.height,m.width],'center_patch_median_mm':float(np.median(validpatch)) if len(validpatch) else None,'center_patch_rms_vs_1m_mm':float(np.sqrt(np.mean((validpatch.astype(float)-1000)**2))) if len(validpatch) else None,'valid_fraction':float((z>0).mean()),'left_column_valid_pixels':int((z[:,0]>0).sum())}
 c['metric_depth']=len(validpatch)>800 and abs(float(np.median(validpatch))-1000)<20
 c['depth_size']=m.height==H and m.width==W and m.step==W*2 and m.encoding=='16UC1'
 c['stereo_invalid_left_band']=not np.any(z[:,0]);c['observation_is_not_perfect_truth']=out['depth']['center_patch_rms_vs_1m_mm'] is not None and .5<out['depth']['center_patch_rms_vs_1m_mm']<20
 c['usable_but_incomplete_depth']=.5<out['depth']['valid_fraction']<.999
if 'truth' in seen:
 m=seen['truth'];gt=np.frombuffer(m.data,dtype='<f4').reshape(m.height,m.width)
 out['truth']={'encoding':m.encoding,'center_m':float(gt[H//2,W//2])}
 c['clean_truth_separate']=m.encoding=='32FC1' and abs(float(gt[H//2,W//2])-1)<1e-4
 # Known off-axis object centers and projected size are tested in clean geometry, not noisy observations.
 def sample(x,y,z):
  u=int(round(kd['fx']*x/z+kd['cx']));v=int(round(kd['fy']*y/z+kd['cy']));return float(gt[v,u])
 out['truth']['blue_box_m']=sample(-.44,-.22,1.19);out['truth']['green_box_m']=sample(.42,.25,1.29)
 c['off_center_geometry']=abs(out['truth']['blue_box_m']-1.19)<.002 and abs(out['truth']['green_box_m']-1.29)<.002
 ys,xs=np.where(np.abs(gt-1.)<.001);width=int(xs.max()-xs.min()+1) if len(xs) else 0
 out['truth']['center_target_width_px']=width;out['truth']['expected_width_px']=float(kd['fx']*.28)
 c['projected_geometry']=abs(width-kd['fx']*.28)<4
if 'rgb' in seen:
 m=seen['rgb'];rgb=np.frombuffer(m.data,dtype=np.uint8).reshape(m.height,m.width,3);c['rgb_image']=m.encoding=='rgb8' and rgb.std()>5 and m.width==W and m.height==H
if 'di' in seen:
 m=seen['di'];out['depth_K']=list(m.k);c['camera_info']=abs(float(m.k[0])-kd['fx'])<.01 and abs(float(m.k[4])-kd['fy'])<.01 and m.width==W and m.height==H
if 'points' in seen:
 m=seen['points'];pts=np.frombuffer(m.data,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')]);out['cloud']={'width':m.width,'height':m.height,'points':len(pts),'finite_points':int(np.isfinite(pts['z']).sum()),'frame':m.header.frame_id}
 c['full_resolution_cloud']=m.width==W and m.height==H and len(pts)==W*H
 common=set(history.get('points',{}))&set(history.get('depth',{}))
 if common:
  ts=max(common);pm=history['points'][ts];dm=history['depth'][ts]
  pc=np.frombuffer(pm.data,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')]).reshape(H,W);dz=np.frombuffer(dm.data,dtype='<u2').reshape(H,W);v=dz>0
  err=np.abs(pc['z'][v]-dz[v].astype(np.float32)*.001)
  out['cloud']['max_z_vs_observation_error_m']=float(err.max()) if err.size else None
  c['cloud_uses_observation_not_truth']=bool(err.size and err.max()<1e-6 and np.all(~np.isfinite(pc['z'][~v])))
 else:c['cloud_uses_observation_not_truth']=False
out['unique_frames']={k:len(v) for k,v in hashes.items()};c['live_rgb_motion']=out['unique_frames'].get('rgb',0)>3;c['live_depth_frames']=out['unique_frames'].get('depth',0)>3
if 'rgb' in stamps and 'depth' in stamps:
 out['matched_rgb_depth_stamps']=len(stamps['rgb']&stamps['depth']);c['paired_timestamps']=out['matched_rgb_depth_stamps']>=.9*min(len(stamps['rgb']),len(stamps['depth']))
if 'aligned' in seen:
 m=seen['aligned'];az=np.frombuffer(m.data,dtype='<u2');out['aligned_valid_fraction']=float((az>0).mean());c['registered_depth_usable']=out['aligned_valid_fraction']>.5
for key in ['rgb','depth','points','truth']:c[key+'_delivery_rate']=out['rates_hz'].get(key,0)>4
out['counts']={k:len(v) for k,v in times.items()};out['checks']={k:bool(v) for k,v in c.items()};out['passed']=bool(c) and all(c.values())
(root/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));node.destroy_node();rclpy.shutdown();raise SystemExit(0 if out['passed'] else 1)
