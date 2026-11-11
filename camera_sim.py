"""Synthetic D435i-style ROS 2 interface. Not a hardware/firmware replica."""
import argparse, os, json, time, signal, ctypes
from pathlib import Path
root=Path(__file__).resolve().parent
profile=json.loads((root/'d435i_profile.json').read_text())
p=argparse.ArgumentParser()
p.add_argument('--duration',type=float,default=0)
p.add_argument('--fps',type=float,default=profile['requested_fps'])
p.add_argument('--noise-disparity',type=float,default=None)
p.add_argument('--observation-mode',choices=['documented','clean'],default=profile['default_observation_mode'])
p.add_argument('--cloud-stride',type=int,default=profile['default_cloud_stride'])
a,_=p.parse_known_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':False,'width':1280,'height':800,'renderer':'RaytracedLighting'})
import numpy as np
from d435i_model import DepthObservationModel,intrinsics
import omni.usd, omni.timeline
from pxr import UsdGeom, UsdLux, Gf, Sdf
from isaacsim.core.experimental.utils.app import enable_extension
from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
enable_extension('isaacsim.ros2.bridge')
app.update()
import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time
root=Path(__file__).resolve().parent
W,H=profile['width'],profile['height']
if a.cloud_stride<1:raise ValueError('cloud-stride must be positive')
model=DepthObservationModel(profile)
aligner=ctypes.CDLL(str(root/'libalignment.so')).align_depth_mm
aligner.argtypes=[np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS'),ctypes.c_float]
aligner.restype=None
# Nominal rectified profiles, not measurements from the user's D435i.
profiles=intrinsics(profile)
D=np.array([0.,0,2.2]);C=D+np.array([profile['rgb_to_depth_offset_m'],0,0]);R=D+np.array([profile['stereo_baseline_m'],0,0])
stage=omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage,1.)
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
world=UsdGeom.Xform.Define(stage,'/World'); stage.SetDefaultPrim(world.GetPrim())
light=UsdLux.DomeLight.Define(stage,'/World/Light'); light.CreateIntensityAttr(1200.)
def box(name,pos,size,color):
    b=UsdGeom.Cube.Define(stage,'/World/'+name); b.CreateSizeAttr(1.)
    x=UsdGeom.XformCommonAPI(b); x.SetTranslate(Gf.Vec3d(*pos)); x.SetScale(Gf.Vec3f(*size))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return b
box('Floor',(0,0,-.03),(6,6,.05),(.18,.19,.21))
box('Table',(0,0,.72),(1.8,1.4,.06),(.58,.51,.40))
# Center patch: top surface at z=1.2 => exactly 1.0 m below the camera.
box('OneMeterTarget',(0,0,.975),(.28,.28,.45),(.8,.18,.08))
box('BlueBox',(-.44,.22,.88),(.25,.25,.26),(.05,.25,.70))
box('GreenBox',(.42,-.25,.83),(.22,.28,.16),(.05,.50,.28))
for i in range(9):
    box('Ruler%02d'%i,(-.65+i*.16,.52,.757),(.075,.035,.008),(.9,.9,.87) if i%2 else (.04,.04,.05))
moving=box('MovingMarker',(.42,.20,.84),(.10,.10,.18),(.95,.70,.08))
sensors={}
for name,pos in [('color',C),('depth',D),('right',R)]:
    intr=profiles['depth' if name=='right' else name]
    cam=RtxCamera('/World/D435i/'+name,tick_rate=a.fps,translations=pos,orientations=np.array([1.,0.,0.,0.]))
    # USD camera looks along -Z; optical frame looks +Z with Y down.
    # Set focal length and aperture in the SAME USD camera units below.
    prim=stage.GetPrimAtPath('/World/D435i/'+name)
    usd=UsdGeom.Camera(prim)
    usd.GetFocalLengthAttr().Set(10.)
    usd.GetHorizontalApertureAttr().Set(10.*W/intr['fx'])
    usd.GetVerticalApertureAttr().Set(10.*H/intr['fy'])
    usd.GetClippingRangeAttr().Set(Gf.Vec2f(.01,1000.))
    sensors[name]=CameraSensor(cam,resolution=(H,W),annotators=['rgb','distance_to_image_plane'] if name=='color' else ['distance_to_image_plane'])
# View the actual virtual camera in the Isaac window.
try:
    from omni.kit.viewport.utility import get_active_viewport
    get_active_viewport().set_active_camera('/World/D435i/color')
except Exception as e: print('VIEWPORT_NOTE',str(e),flush=True)
stage.Export(str(root/'scene.usda'))
rclpy.init()
node=rclpy.create_node('d435i_isaac_sim')
base='/camera/camera/'
pubs={}
for key,topic,typ in [
 ('rgb','color/image_raw',Image),('depth','depth/image_rect_raw',Image),
 ('aligned','aligned_depth_to_color/image_raw',Image),
 ('ci','color/camera_info',CameraInfo),('di','depth/camera_info',CameraInfo),
 ('ai','aligned_depth_to_color/camera_info',CameraInfo),('points','depth/color/points',PointCloud2)]:
    pubs[key]=node.create_publisher(typ,base+topic,qos_profile_sensor_data)
gtpub=node.create_publisher(Image,'/sim/ground_truth/depth/image_rect_raw',qos_profile_sensor_data)
gtinfo=node.create_publisher(CameraInfo,'/sim/ground_truth/depth/camera_info',qos_profile_sensor_data)
clock_pub=node.create_publisher(Clock,'/clock',10)
staticq=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
tfpub=node.create_publisher(TFMessage,'/tf_static',staticq)
def trans(parent,child,xyz,q):
    t=TransformStamped();t.header.frame_id=parent;t.child_frame_id=child
    t.transform.translation.x=float(xyz[0]);t.transform.translation.y=float(xyz[1]);t.transform.translation.z=float(xyz[2])
    t.transform.rotation.x=float(q[0]);t.transform.rotation.y=float(q[1]);t.transform.rotation.z=float(q[2]);t.transform.rotation.w=float(q[3]);return t
# Camera_link uses the ROS body convention (+X forward, +Y left, +Z up).
tfpub.publish(TFMessage(transforms=[
 trans('world','camera_link',D,[-.5,.5,.5,.5]),
 trans('camera_link','camera_depth_frame',[0,0,0],[0,0,0,1]),
 trans('camera_depth_frame','camera_depth_optical_frame',[0,0,0],[-.5,.5,-.5,.5]),
 trans('camera_link','camera_infra2_frame',[0,-profile['stereo_baseline_m'],0],[0,0,0,1]),
 trans('camera_infra2_frame','camera_infra2_optical_frame',[0,0,0],[-.5,.5,-.5,.5]),
 trans('camera_link','camera_color_frame',[0,-profile['rgb_to_depth_offset_m'],0],[0,0,0,1]),
 trans('camera_color_frame','camera_color_optical_frame',[0,0,0],[-.5,.5,-.5,.5])]))
def stamp(t):
    sec=int(t);return Time(sec=sec,nanosec=int((t-sec)*1e9))
def image(arr,encoding,frame,ts):
    arr=np.ascontiguousarray(arr);msg=Image();msg.header.stamp=ts;msg.header.frame_id=frame
    msg.height=H;msg.width=W;msg.encoding=encoding;msg.is_bigendian=0;msg.step=int(arr.strides[0]);msg.data=arr.tobytes();return msg
def info(which,frame,ts):
    k=profiles[which];fx,fy,cx,cy=[k[x] for x in ['fx','fy','cx','cy']]
    m=CameraInfo();m.header.stamp=ts;m.header.frame_id=frame;m.width=W;m.height=H
    m.distortion_model='plumb_bob';m.d=[0.]*5;m.k=[fx,0.,cx,0.,fy,cy,0.,0.,1.]
    m.r=[1.,0.,0.,0.,1.,0.,0.,0.,1.];m.p=[fx,0.,cx,0.,0.,fy,cy,0.,0.,0.,1.,0.];return m
yy,xx=np.indices((H,W),dtype=np.float32)
kd=profiles['depth'];kc=profiles['color']
rx=(xx-kd['cx'])/kd['fx'];ry=(yy-kd['cy'])/kd['fy']
kd_c=np.array([kd[k] for k in ['fx','fy','cx','cy']],dtype=np.float32)
kc_c=np.array([kc[k] for k in ['fx','fy','cx','cy']],dtype=np.float32)
keep=True
signal.signal(signal.SIGTERM,lambda *_:globals().update(keep=False))
start=time.monotonic();last_save=0;count=0;last_pub=-1.;last_wall=0.;rates=[]
timeline=omni.timeline.get_timeline_interface();timeline.play()
print('D435I_SCENE_READY '+str(root),flush=True)
try:
    while app.is_running() and keep:
        app.update()
        if not timeline.is_playing():
            rclpy.spin_once(node,timeout_sec=0.);continue
        simtime=timeline.get_current_time()
        if time.monotonic()-last_wall < 1./a.fps:continue
        if simtime-last_pub < 1./a.fps-1e-5:continue
        publish_start=time.monotonic()
        rg,ri=sensors['color'].get_data('rgb')
        zd,zi=sensors['depth'].get_data('distance_to_image_plane')
        if rg is None or zd is None:continue
        rgb=np.array(rg.numpy()[...,:3],dtype=np.uint8)
        z=np.array(zd.numpy(),dtype=np.float32).reshape(H,W)
        if rgb.shape != (H,W,3):continue
        right,_=sensors['right'].get_data('distance_to_image_plane')
        if right is None:continue
        ground_truth=z.copy()
        right_truth=np.array(right.numpy(),dtype=np.float32).reshape(H,W)
        depth,observation_diagnostics=model.observe(ground_truth,right_truth,mode=a.observation_mode,noise_override=a.noise_disparity)
        valid=depth>0
        z=depth.astype(np.float32)*profile['depth_unit_m']
        X=rx*z;Y=ry*z
        # Camera extrinsic: color origin is +15 mm along optical X, no rotation.
        uc=np.rint(kc['fx']*(X-profile['rgb_to_depth_offset_m'])/np.maximum(z,.001)+kc['cx']).astype(np.int32)
        vc=np.rint(kc['fy']*Y/np.maximum(z,.001)+kc['cy']).astype(np.int32)
        mask=valid&(uc>=0)&(uc<W)&(vc>=0)&(vc<H)
        aligned=np.zeros((H,W),dtype=np.uint16)
        aligner(depth,aligned,W,H,kd_c,kc_c,profile['rgb_to_depth_offset_m'])
        ts=stamp(simtime);df='camera_depth_optical_frame';cf='camera_color_optical_frame'
        gt=np.where(np.isfinite(ground_truth),ground_truth,0).astype('<f4')
        gtpub.publish(image(gt,'32FC1',df,ts));gtinfo.publish(info('depth',df,ts))
        pubs['rgb'].publish(image(rgb,'rgb8',cf,ts));pubs['depth'].publish(image(depth,'16UC1',df,ts))
        pubs['aligned'].publish(image(aligned,'16UC1',cf,ts))
        for key,profile_name,frame in [('ci','color',cf),('di','depth',df),('ai','color',cf)]:pubs[key].publish(info(profile_name,frame,ts))
        # Organized cloud in depth optical coordinates, RGB sampled after reprojection.
        # Production cloud uses the OBSERVATION depth, not hidden clean ground truth.
        stride=a.cloud_stride
        sl=(slice(None,None,stride),slice(None,None,stride))
        xyz=np.stack([X[sl],Y[sl],z[sl]],axis=-1);xyz[~valid[sl]]=np.nan
        colors=np.zeros((H,W,3),dtype=np.uint8);colors[mask]=rgb[vc[mask],uc[mask]]
        cc=colors[sl].astype(np.uint32);packed=(cc[...,0]<<16)|(cc[...,1]<<8)|cc[...,2]
        cloud=np.empty(xyz.shape[:2],dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
        for i,field in enumerate(['x','y','z']):cloud[field]=xyz[...,i]
        cloud['rgb']=packed
        pm=PointCloud2();pm.header.stamp=ts;pm.header.frame_id=df;pm.height=xyz.shape[0];pm.width=xyz.shape[1]
        pm.fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32 if i<3 else PointField.UINT32,count=1) for i,n in enumerate(['x','y','z','rgb'])]
        pm.is_bigendian=False;pm.is_dense=False;pm.point_step=16;pm.row_step=pm.width*16;pm.data=cloud.tobytes();pubs['points'].publish(pm)
        clock_pub.publish(Clock(clock=ts));rclpy.spin_once(node,timeout_sec=0.)
        now=time.monotonic();rates.append(now);rates=rates[-120:];count+=1;last_pub=simtime;last_wall=publish_start
        # Visibly changing geometry demonstrates a live simulation rather than replay.
        UsdGeom.XformCommonAPI(moving).SetTranslate(Gf.Vec3d(.42+.08*np.sin(simtime),.20,.84))
        if now-last_save>=5:
            from PIL import Image as PILImage
            PILImage.fromarray(rgb).save(root/'rgb.png')
            heat=np.clip(depth.astype(np.float32)/2500,0,1)
            visual=np.stack([1.-heat,1.-np.abs(heat-.5)*2,heat],axis=-1)
            visual[depth==0]=0
            PILImage.fromarray((visual*255).astype(np.uint8)).save(root/'depth_preview.png')
            np.save(root/'depth_mm.npy',depth)
            np.save(root/'ground_truth_depth_m.npy',ground_truth)
            stats={'frames':count,'wall_fps':(len(rates)-1)/(rates[-1]-rates[0]) if len(rates)>1 else 0,
              'sim_time':simtime,'shape':[H,W],'encoding':'16UC1','depth_unit_m':.001,
              'center_depth_m':float(depth[H//2,W//2])*profile['depth_unit_m'],'expected_center_depth_m':1.,
              'valid_depth_fraction':float(valid.mean()),'aligned_valid_fraction':float((aligned>0).mean()),
              'observation':observation_diagnostics,'profile_name':profile['name'],'cloud_stride':a.cloud_stride,'cloud_shape':[pm.height,pm.width],'nominal_depth_hfov_deg':profile['depth_hfov_deg'],'nominal_color_hfov_deg':profile['color_hfov_deg'],'profiles':profiles,'optics':{n:{'usd_focal_length':float(UsdGeom.Camera(stage.GetPrimAtPath('/World/D435i/'+n)).GetFocalLengthAttr().Get()),'usd_horizontal_aperture':float(UsdGeom.Camera(stage.GetPrimAtPath('/World/D435i/'+n)).GetHorizontalApertureAttr().Get())} for n in profiles},'noise_disparity_px':a.noise_disparity,
              'limitations':['Nominal calibration, not measured D435i calibration','Observation uses a documentation-informed surrogate; D4 matching and material/IR behavior are not emulated','Alignment uses RealSense-style projected pixel footprints and nearest-depth overlap resolution','No IMU or infrared streams']}
            (root/'status.json').write_text(json.dumps(stats,indent=2))
            print('D435I_STREAM '+json.dumps(stats),flush=True);last_save=now
        if a.duration and now-start>a.duration:break
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    timeline.stop();node.destroy_node();rclpy.shutdown();app.close()
