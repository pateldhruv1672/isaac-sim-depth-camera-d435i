"""Render README figures from captures made by tools/capture_frames.py (numpy + Pillow only).

Usage: python3 tools/make_readme_figures.py CAPTURE_DIR OUT_DIR
Expects documented.npz; clean.npz and noisy.npz are optional (variants figure).
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG, INK, MUTED, RULE = (250, 250, 249), (24, 24, 27), (113, 113, 122), (228, 228, 231)
def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    for d in ['/usr/share/fonts/truetype/dejavu/', '']:
        try: return ImageFont.truetype(d + name, size)
        except OSError: pass
    return ImageFont.load_default()
MONO = lambda s: next((ImageFont.truetype(p, s) for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'] if Path(p).exists()), font(s))

def turbo(x):
    """Polynomial approximation of Google's Turbo colormap, x in [0, 1]."""
    x = np.clip(x, 0, 1)[..., None]
    r = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
    g = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    b = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])
    p = np.concatenate([x ** i for i in range(6)], -1)
    return (np.clip(np.stack([p @ r, p @ g, p @ b], -1), 0, 1) * 255).astype(np.uint8)

def depth_vis(mm, lo=900, hi=2300):
    out = turbo((mm.astype(np.float32) - lo) / (hi - lo)); out[mm == 0] = (16, 16, 18); return out

def error_vis(obs_mm, truth_m, lim=8.0):
    e = obs_mm.astype(np.float32) - truth_m * 1000.0
    t = np.clip(e / lim, -1, 1)[..., None]
    neg, pos, white = np.array([37, 99, 235.]), np.array([220, 38, 38.]), np.array([250, 250, 249.])
    out = np.where(t < 0, white + (neg - white) * -t, white + (pos - white) * t).astype(np.uint8)
    out[obs_mm == 0] = (16, 16, 18); return out

def overlay(rgb, aligned_mm):
    v = depth_vis(aligned_mm).astype(np.float32); base = rgb.astype(np.float32) * 0.35
    out = np.where((aligned_mm > 0)[..., None], 0.55 * v + 0.45 * rgb, base); return out.astype(np.uint8)

def render_cloud(xyz, packed, size=(848, 480)):
    """Oblique view of the organized cloud; NaN (invalid) points are skipped, nearest point wins."""
    pts = xyz.reshape(-1, 3); ok = np.isfinite(pts).all(1); pts = pts[ok]; c = packed.reshape(-1)[ok]
    col = np.stack([(c >> 16) & 255, (c >> 8) & 255, c & 255], -1).astype(np.uint8)
    centre = np.array([0., 0., 1.45]); p = pts - centre
    a, b = np.radians(-30), np.radians(22)              # tilt about optical X, then yaw about Y
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    p = p @ (Ry @ Rx).T; p[:, 2] += 1.9
    W, H = size; f = 620.0
    u = (f * p[:, 0] / p[:, 2] + W / 2).astype(int); v = (f * p[:, 1] / p[:, 2] + H / 2).astype(int)
    k = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1); order = np.argsort(-p[k, 2])
    u, v, col = u[k][order], v[k][order], col[k][order]
    img = np.full((H, W, 3), 16, np.uint8)
    for du in (0, 1):
        for dv in (0, 1): img[v + dv, u + du] = col
    return img

def tile(img, title, sub, w=424):
    im = Image.fromarray(img).resize((w, int(w * img.shape[0] / img.shape[1])), Image.LANCZOS)
    card = Image.new('RGB', (w, im.height + 54), BG); card.paste(im, (0, 54)); d = ImageDraw.Draw(card)
    d.text((0, 6), title, font=font(16, True), fill=INK); d.text((0, 30), sub, font=MONO(12), fill=MUTED)
    return card

def colorbar(w, label_lo, label_hi, kind, caption):
    bar = Image.new('RGB', (w, 44), BG); x = np.linspace(0, 1, w - 160)
    if kind == 'turbo': strip = turbo(x)
    else: strip = error_vis(((x * 2 - 1) * 8).astype(np.float32), np.zeros_like(x))
    strip = np.repeat(strip[None], 10, 0); bar.paste(Image.fromarray(strip), (80, 6))
    d = ImageDraw.Draw(bar); d.text((0, 2), label_lo, font=MONO(12), fill=MUTED)
    d.text((w - 72, 2), label_hi, font=MONO(12), fill=MUTED); d.text((80, 22), caption, font=font(12), fill=MUTED)
    return bar

def grid(cards, cols, gap=24, pad=28, header=None, footer=None):
    rows = [cards[i:i + cols] for i in range(0, len(cards), cols)]
    cw = max(c.width for c in cards); rh = [max(c.height for c in r) for r in rows]
    W = pad * 2 + cols * cw + (cols - 1) * gap; hh = 0 if header is None else header.height + 16
    fh = 0 if footer is None else max(f.height for f in footer) + 12
    H = pad * 2 + hh + sum(rh) + gap * (len(rows) - 1) + fh
    out = Image.new('RGB', (W, H), BG); y = pad
    if header is not None: out.paste(header, (pad, y)); y += hh
    for r, h in zip(rows, rh):
        for i, c in enumerate(r): out.paste(c, (pad + i * (cw + gap), y))
        y += h + gap
    if footer is not None:
        y += 12 - gap
        x = pad
        for f in footer: out.paste(f, (x, y)); x += f.width + gap
    return out

def heading(w, title, sub):
    h = Image.new('RGB', (w, 52), BG); d = ImageDraw.Draw(h)
    d.text((0, 0), title, font=font(22, True), fill=INK); d.text((0, 30), sub, font=font(14), fill=MUTED); return h

def stats(c):
    dm, t = c['depth_mm'], c['truth_m']; H, W = dm.shape; v = dm > 0
    e = dm[v].astype(np.float32) - t[v] * 1000
    patch = dm[H // 2 - 15:H // 2 + 15, W // 2 - 15:W // 2 + 15].astype(np.float32); pv = patch[patch > 0]
    return {'valid_pct': 100 * v.mean(), 'rms_mm': float(np.sqrt(np.mean(e ** 2))),
            'centre_rms_mm': float(np.sqrt(np.mean((pv - 1000) ** 2))), 'centre_median_mm': float(np.median(pv))}

def main(src, dst):
    src, dst = Path(src), Path(dst); dst.mkdir(parents=True, exist_ok=True)
    caps = {n: dict(np.load(src / f'{n}.npz')) for n in ['documented', 'clean', 'noisy'] if (src / f'{n}.npz').exists()}
    c = caps['documented']
    cards = [tile(c['rgb'], 'RGB', '/camera/camera/color/image_raw · rgb8'),
             tile(depth_vis(c['depth_mm']), 'Depth (observation)', '/camera/camera/depth/image_rect_raw · 16UC1'),
             tile(overlay(c['rgb'], c['aligned_mm']), 'Depth aligned to RGB', '/camera/camera/aligned_depth_to_color/image_raw'),
             tile(depth_vis((np.nan_to_num(c['truth_m']) * 1000).astype(np.uint16)), 'Ground-truth depth', '/sim/ground_truth/depth/image_rect_raw · 32FC1'),
             tile(error_vis(c['depth_mm'], c['truth_m']), 'Observation − ground truth', 'per-pixel error; black = invalid (0)'),
             tile(render_cloud(c['cloud_xyz'], c['cloud_rgb']), 'Organized point cloud', '/camera/camera/depth/color/points · 848×480')]
    w = 3 * 424 + 2 * 24
    out = grid(cards, 3, header=heading(w, 'One timestamp-matched frame from every output stream',
               'documented mode · 848×480 · camera 2.2 m above the scene, orange target top exactly 1.0 m away'),
               footer=[colorbar((w - 24) // 2, '0.9 m', '2.3 m', 'turbo', 'depth colour scale'),
                       colorbar((w - 24) // 2, '−8 mm', '+8 mm', 'error', 'error scale (blue nearer, red farther than truth)')])
    out.save(dst / 'outputs.png', optimize=True)
    summary = {n: stats(v) for n, v in caps.items()}
    if len(caps) > 1:
        order = [(n, l) for n, l in [('clean', 'clean'), ('documented', 'documented (default)'), ('noisy', '--noise-disparity 0.25')] if n in caps]
        y0, y1, x0, x1 = 120, 380, 150, 600                     # crop around the blue box and its stereo shadow
        cards = []
        for n, label in order:
            v = caps[n]; s = summary[n]
            sub = f"valid {s['valid_pct']:.1f}% · RMS vs truth {s['rms_mm']:.1f} mm"
            cards += [tile(depth_vis(v['depth_mm']), label, sub, w=380),
                      tile(error_vis(v['depth_mm'], v['truth_m'])[y0:y1, x0:x1], 'error, zoomed', 'blue box, stereo shadow, ruler', w=380),
                      tile(render_cloud(v['cloud_xyz'], v['cloud_rgb']), 'point cloud', 'same viewpoint for every mode', w=380)]
        w = 3 * 380 + 2 * 24
        out = grid(cards, 3, header=heading(w, 'Observation-model variants', 'same scene and camera pose; only the depth observation model changes'),
                   footer=[colorbar((w - 24) // 2, '0.9 m', '2.3 m', 'turbo', 'depth colour scale'),
                           colorbar((w - 24) // 2, '−8 mm', '+8 mm', 'error', 'error scale; black = invalid (0)')])
        out.save(dst / 'variants.png', optimize=True)
    (dst / 'figure-stats.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
