# Isaac Sim depth camera: Intel RealSense D435i

Developed and tested with Isaac Sim 6.0.1 (native ARM64, NVIDIA DGX Spark, Ubuntu 24.04) and ROS 2 Jazzy.
No physical camera is needed to run or configure this model.
This is a documentation-informed sensor surrogate, not a reproduction of proprietary D4 firmware.

## Run

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim   # folder containing python.sh
bash run.sh
```

In another terminal, `bash rviz.sh`. Both launchers use ROS_DOMAIN_ID=43 and
fastdds.xml (64 MiB shared-memory segments). Existing robot ROS domains are untouched.
Close the Isaac window to stop the publisher. `--duration 120` is a bounded run.
`--observation-mode clean` disables surrogate noise and stereo visibility masking;
it is a geometric baseline, not the training default. Default is `documented`.
`--cloud-stride 2` is an explicitly requested preview reduction; default cloud stride is 1.
`--fps 15` requests the sensor/publish rate but is not a measured real-time guarantee.
Actual independent receiver measurement for this version was about 10.7 Hz with RViz open.

## Current data contract

Color and depth are 848x480. RGB is rgb8. Observation depth is 16UC1, one unit=1 mm,
zero=invalid. The organized cloud has 407,040 sample positions; invalid depth becomes
NaN XYZ, never an invented surface. It is derived from the SAME observation image.

- /camera/camera/color/image_raw and color/camera_info
- /camera/camera/depth/image_rect_raw and depth/camera_info
- /camera/camera/aligned_depth_to_color/image_raw and aligned_depth_to_color/camera_info
- /camera/camera/depth/color/points
- /sim/ground_truth/depth/image_rect_raw (32FC1, meters) and camera_info
- /tf_static, /clock

Image and cloud QoS is best-effort sensor data. Timestamps use simulation time;
consumers should use use_sim_time=true. Optical axes: +X right, +Y down, +Z forward.
Ground-truth topics are for evaluation only, not policy/perception inputs.

## What is documented, derived or assumed

`d435i_profile.json` is the machine-readable source of settings and provenance.

Documented: 50 mm stereo baseline; 848x480 supported and manufacturer-recommended
for depth; HD depth HFOV=87 degrees and RGB HFOV=69 degrees; MinZ=195 mm at 848x480;
1 mm default depth units; left invalid band proportional to B*f/Z.
The datasheet distinguishes VGA 75x62 degree depth FOV and HD 87x58 degree depth FOV.
A fixed FOV across aspect ratios would be wrong.

Derived: centered nominal rectified intrinsics from HFOV with square pixels.
fx=fy=446.8028 for depth, 616.9238 for color; principal point=(424,240).
Derived vertical FOV is about 56.5 degrees for depth and 42.5 degrees for RGB.
These are consistent nominal choices within the table's lens tolerances, NOT actual
factory calibration. Rounded nominal HFOV/VFOV values do not uniquely define a camera.

Illustrative reference, not a universal specification: 1/32-pixel disparity step and
0.08-pixel RMS disparity noise from the manufacturer white papers. Noise is applied
in disparity space so depth error grows approximately with Z squared.
The temporal sigma=0.03 px and spatial correlation kernel are engineering assumptions.
They are explicitly editable and are NOT a manufacturer-validated noise distribution.
No generic +/-2% Gaussian-noise setting has been invented from the quality spec.

Assumed: RGB origin 15 mm along optical X from the left depth origin, parallel axes,
zero rectified distortion, centered principal point, 10 m application cutoff, and
visibility tolerance of 3 mm + 0.2% of depth. The RGB baseline is NOT the 50 mm stereo
baseline. Real extrinsics and intrinsics must be loaded from device calibration later.

## Observation construction

1. Isaac renders separate left/right axial depth views with a 50 mm baseline.
2. Points outside the right image or occluded there are invalidated using geometric
   consistency. This creates stereo occlusion holes and the left invalid band.
3. Apply minimum range and illustrative spatial/temporal disparity noise and quantization.
4. Convert valid depth to Z16-compatible millimeter values. Keep invalid depth zero.
5. Align using a compiled zero-distortion, parallel-axis specialization of librealsense
   align.cpp's pixel-corner projection, footprint rasterization and nearest-depth rule.
6. Project the observation depth to XYZ and sample the RGB texture through the nominal
   calibration. The production cloud is not built from clean renderer depth.

No hole filling, temporal persistence or smoothing is silently enabled. Once the actual
robot processing chain is known, use equivalent SDK filters on both real and sim streams.
A policy trained with undocumented hole filling can learn surfaces that were never observed.

## Validation and evidence limits

`test_model.py` tests the standalone mathematical surrogate on flat planes: at
0.5/1/2/4 m its measured RMS depth error follows the chosen disparity-noise model,
not a measurement of a real D435i. `model-unit-tests.json` records these results.

`validate.py` independently receives ROS data, checks dimensions/units/intrinsics,
known center and off-axis scene geometry against separate ground truth, live frames,
matching RGB/depth timestamps, invalid bands, and exact cloud Z agreement with
observation depth. `results/validation.json` and `results/model-unit-tests.json` hold the recorded results of the published version.

The first documentation-informed live check measured about 3.49 mm RMS against the
1 m target patch. That is a property of THIS configured surrogate, NOT a claim that
your physical cameras have 3.49 mm accuracy. Approximately 97.7% of full-frame raw
samples and 98.3% of registered samples were valid in this particular scene.
Factory >=99% fill specifications are qualified central-ROI tests, not an arbitrary
scene-wide requirement including stereo occlusions.

Unmodeled: proprietary D4 matching/confidence, active IR dots/speckle, reflectivity and
transparency failures, RGB rolling shutter/ISP/exposure behavior, A-factor nonlinearity,
temperature/calibration drift, actual latency, IMU and IR streams, robot mount transforms.
Real calibration exports or recorded bags can be used later; the camera need not be
physically connected to the Spark. Without real reference data we cannot validate the
remaining sensor-domain or downstream task gap.

## Sources retrieved 2026-11-11

- D400 family datasheet rev021, October 2025: https://realsenseai.com/wp-content/uploads/2025/09/Intel-RealSense-D400-Series-Datasheet-October-2025.pdf
  pp.67 (baseline/FOV/tolerances), 75-76 (stream modes), 82-84 (FOV/invalid band),
  85 (MinZ), 86 (depth-quality conditions).
- https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/
  Optimal 848x480, 50 mm baseline, illustrative 0.08-pixel RMS curves, Z-squared scaling.
- https://dev.realsenseai.com/docs/white-paper-subpixel-linearity-improvement-for-intel-realsense-depth-cameras/
  1/32-pixel example, preset/A-factor nonlinearity and limitations of RMS-only judgments.
- https://dev.realsenseai.com/docs/post-processing-filters/
  SDK filters are independent optional blocks; do not confuse visual decimation with
  sensor calibration or enable unknown real-robot filter settings by default.
- librealsense src/proc/align.cpp (align_images / align_z_to_other):
  registration pixel footprints and nearest-depth overlap resolution.


## Requirements

- NVIDIA Isaac Sim 6.x with the ROS 2 bridge extension
- ROS 2 Jazzy (system install) for RViz and `validate.py`
- g++ (builds `libalignment.so` on first run)

## Files

| File | Purpose |
|---|---|
| `camera_sim.py` | Isaac scene, left/right/RGB render products, ROS 2 publishers |
| `d435i_model.py` | Observation-depth surrogate (stereo visibility, disparity noise, quantization) |
| `d435i_profile.json` | Camera profile with per-value provenance |
| `alignment.cpp` | RealSense-style depth-to-color registration |
| `validate.py` | Independent ROS subscriber checks |
| `test_model.py` | Standalone plane tests of the depth model |
| `rviz.sh`, `camera.rviz` | RViz view of RGB, depth and the colored cloud |
| `fastdds.xml` | Shared-memory transport sized for full-resolution RGB-D and clouds |

## License

This project's code is licensed under the [Apache License 2.0](LICENSE).
Anyone may use, modify and redistribute it for education, research and commercial
purposes, subject to the license's conditions, including retaining required notices.
This is not an education-only or non-commercial license.

See [NOTICE](NOTICE) for RealSense alignment attribution. NVIDIA Isaac Sim and other
external dependencies retain their own licenses; this repository does not relicense
or redistribute those products. The simulation is provided without warranty.
