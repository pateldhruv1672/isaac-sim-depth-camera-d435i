"""Documentation-informed depth surrogate, NOT an emulation of proprietary D4 matching.
Geometry (occlusions/min-Z) is separate from explicitly assumed disparity statistics.
"""
import numpy as np

def intrinsics(profile):
    w,h=profile['width'],profile['height']
    out={}
    for name in ['depth','color']:
        f=w/(2*np.tan(np.deg2rad(profile[name+'_hfov_deg'])/2))
        out[name]={'fx':float(f),'fy':float(f),'cx':w/2.,'cy':h/2.}
    return out

class DepthObservationModel:
    def __init__(self,profile,seed=435):
        self.p=profile;self.k=intrinsics(profile)['depth'];self.rng=np.random.default_rng(seed)
        self.h,self.w=profile['height'],profile['width']
        self.yy,self.xx=np.indices((self.h,self.w),dtype=np.float32)
        fixed=self.rng.normal(size=(self.h,self.w)).astype(np.float32)
        # Spatial correlation is an engineering assumption, not a measured D435i kernel.
        fixed=(fixed+np.roll(fixed,1,0)+np.roll(fixed,-1,0)+np.roll(fixed,1,1)+np.roll(fixed,-1,1))/5
        fixed=(fixed-fixed.mean())/fixed.std()
        total=profile['reference_disparity_rms_px'];temporal=profile['assumed_temporal_disparity_sigma_px']
        self.fixed=fixed*np.sqrt(max(0,total*total-temporal*temporal))

    def observe(self,ground_truth,right_ground_truth,mode='documented',noise_override=None):
        p=self.p;gt=np.asarray(ground_truth,dtype=np.float32)
        valid=np.isfinite(gt)&(gt>=p['min_depth_m'])&(gt<=p['max_depth_m'])
        initial=valid.copy();stereo_invalid=np.zeros_like(valid)
        fb=self.k['fx']*p['stereo_baseline_m']
        zsafe=np.where(valid,gt,1.)
        if mode=='documented':
            # Left reference => right-image correspondence u_r=u_l-f*baseline/Z.
            # Outside right FOV produces the documented left invalid-depth band.
            ur=np.rint(self.xx-fb/zsafe).astype(np.int32)
            inside=(ur>=0)&(ur<self.w)
            zr=right_ground_truth[self.yy.astype(np.int32),np.clip(ur,0,self.w-1)]
            tol=p['visibility_tolerance_m']+p['visibility_relative_tolerance']*zsafe
            visible=inside&np.isfinite(zr)&(np.abs(zr-gt)<=tol)
            stereo_invalid=initial&~visible
            valid &= visible
            disparity=fb/zsafe
            if noise_override is None:
                noise=self.fixed+self.rng.normal(0,p['assumed_temporal_disparity_sigma_px'],gt.shape)
            else:
                noise=self.rng.normal(0,noise_override,gt.shape)
            step=p['disparity_step_px']
            noisy=np.rint((disparity+noise)/step)*step
            z=np.divide(fb,noisy,out=np.zeros_like(gt),where=valid&(noisy>0))
        else:
            z=np.where(valid,gt,0)
        valid &= np.isfinite(z)&(z>=p['min_depth_m'])&(z<=p['max_depth_m'])
        values=np.rint(np.where(valid,z,0)/p['depth_unit_m'])
        valid &= values>0
        out=np.where(valid,np.clip(values,0,65535),0).astype('<u2')
        compared=valid&initial
        errors=(out.astype(np.float32)*p['depth_unit_m']-gt)[compared]
        diag={'mode':mode,'valid_fraction':float((out>0).mean()),
              'stereo_invalid_fraction':float(stereo_invalid.mean()),
              'observation_minus_truth_rms_m':float(np.sqrt(np.mean(errors*errors))) if errors.size else None,
              'observation_minus_truth_bias_m':float(errors.mean()) if errors.size else None,
              'reference_disparity_rms_px':p['reference_disparity_rms_px'],
              'note':'Noise RMS reference is illustrative; temporal/spatial split is assumed, not hardware validated.'}
        return out,diag
