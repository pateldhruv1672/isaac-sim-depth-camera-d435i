import json,time
from pathlib import Path
import numpy as np
from d435i_model import DepthObservationModel,intrinsics
root=Path(__file__).resolve().parent;p=json.loads((root/'d435i_profile.json').read_text())
k=intrinsics(p)['depth'];report={'plane_tests':[]};ok=True
for distance in [.5,1.,2.,4.]:
 model=DepthObservationModel(p);gt=np.full((p['height'],p['width']),distance,dtype=np.float32)
 samples=[];s=time.monotonic()
 for _ in range(8):
  z,d=model.observe(gt,gt);err=z[z>0].astype(float)*p['depth_unit_m']-distance;samples.append(np.sqrt(np.mean(err**2)))
 measured=float(np.mean(samples));theory=distance**2*p['reference_disparity_rms_px']/(k['fx']*p['stereo_baseline_m'])
 expected_band=int(np.ceil(k['fx']*p['stereo_baseline_m']/distance))
 row={'z_m':distance,'rms_m':measured,'linearized_reference_rms_m':theory,'theoretical_left_band_px':expected_band,'first_column_valid':int(np.count_nonzero(z[:,0])),'model_ms_per_frame':1000*(time.monotonic()-s)/8}
 row['passed']=bool(abs(measured-theory)<max(.0006,.15*theory) and not np.any(z[:,0]));ok &= row['passed'];report['plane_tests'].append(row)
model=DepthObservationModel(p);near=np.full((p['height'],p['width']),.18,dtype=np.float32)
z,_=model.observe(near,near);report['min_z_rejects_180mm']=bool(not np.any(z));ok &= report['min_z_rejects_180mm']
gt=np.ones((p['height'],p['width']),dtype=np.float32);z,_=model.observe(gt,gt,mode='clean');report['clean_mode_1m_exact']=bool(np.all(z==1000));ok &= report['clean_mode_1m_exact']
report['passed']=bool(ok);(root/'model-unit-tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(0 if ok else 1)
