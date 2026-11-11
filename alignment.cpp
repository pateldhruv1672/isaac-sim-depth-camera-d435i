// SPDX-License-Identifier: Apache-2.0
// Copyright(c) 2017 RealSense, Inc. All Rights Reserved.
// Modified 2026 by Dhruv Patel; see NOTICE for the scope of modifications.
// Zero-distortion, parallel-optical-axis specialization of librealsense align.cpp.
// Same corner projection, integer rounding, rectangular footprint and nearest-depth rule.
#include <algorithm>
#include <cstdint>
extern "C" void align_depth_mm(const uint16_t* src, uint16_t* dst, int w, int h,
                               const float* kd, const float* kc, float tx) {
 std::fill(dst,dst+w*h,uint16_t(0));
 for(int y=0;y<h;++y) for(int x=0;x<w;++x){
  const uint16_t d=src[y*w+x]; if(!d)continue;
  const float z=d*.001f;
  const int x0=int(kc[0]*((x-.5f-kd[2])/kd[0]-tx/z)+kc[2]+.5f);
  const int x1=int(kc[0]*((x+.5f-kd[2])/kd[0]-tx/z)+kc[2]+.5f);
  const int y0=int(kc[1]*(y-.5f-kd[3])/kd[1]+kc[3]+.5f);
  const int y1=int(kc[1]*(y+.5f-kd[3])/kd[1]+kc[3]+.5f);
  if(x0<0||y0<0||x1>=w||y1>=h)continue;
  for(int j=y0;j<=y1;++j)for(int i=x0;i<=x1;++i){
   uint16_t& o=dst[j*w+i]; if(!o||d<o)o=d;
  }
 }
}
