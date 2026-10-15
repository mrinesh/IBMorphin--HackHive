"""Render the knock-event explainer animation (MP4, ~5 s) for the dashboard.

A stylised cylinder cross-section: the spark flame front spreads, the end gas at
the cylinder edges auto-ignites ahead of it, and the colliding pressure waves
ring through the chamber. A synced cylinder-pressure trace shows the ringing.
The visuals are illustrative, not simulated engine data.

Run from the repository root (requires ffmpeg on PATH):
    uv run python scripts/render_knock_animation.py [output.mp4]
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, SS, FPS, DURATION = 1280, 720, 2, 30, 5.0
N_FRAMES = int(FPS * DURATION)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "dashboard" / "assets" / "knock_animation.mp4"

BG = (11, 15, 23)
METAL = (38, 46, 60)
METAL_HI = (70, 82, 102)
OUTLINE = (104, 118, 140)
TEXT = (226, 232, 240)
MUTED = (120, 132, 150)
KNOCK = (255, 127, 14)  # dashboard FAULT_COLORS["knock_like"]
FLAME_FRONT = (255, 214, 120)
NORMAL = (110, 190, 255)

# Engine geometry (1x pixel coordinates).
CX, BORE_L, BORE_R, WALL = 380, 230, 530, 22
HEAD_Y = 150
PLUG = (CX, HEAD_Y + 2)
TDC_TOP, PIN_DROP, ROD, CRANK_R = 232, 40, 200, 70
CRANK_Y = TDC_TOP + PIN_DROP + ROD + CRANK_R
PISTON_H = 72

# Crank-angle schedule: slow motion around combustion, theta in degrees (0 = TDC).
THETA0, THETA1, MOTION_END, FADE_START = -120.0, 80.0, 4.5, 4.45
SPARK_DEG, KNOCK_DEG = -18.0, 6.0
KEYFRAMES = [(0.0, THETA0), (1.0, SPARK_DEG), (1.85, KNOCK_DEG), (3.2, 32.0), (MOTION_END, THETA1)]


def _theta_schedule():
    t = np.linspace(0, MOTION_END, 4000)
    kt, kth = zip(*KEYFRAMES)
    lin = np.interp(t, kt, kth)
    k = np.exp(-0.5 * (np.arange(-400, 401) * (t[1] - t[0]) / 0.12) ** 2)
    padded = np.concatenate([2 * lin[0] - lin[400:0:-1], lin, 2 * lin[-1] - lin[-2:-402:-1]])
    return t, np.convolve(padded, k / k.sum(), mode="valid")


_T_GRID, _THETA_GRID = _theta_schedule()
T_SPARK = float(np.interp(SPARK_DEG, _THETA_GRID, _T_GRID))
T_KNOCK = float(np.interp(KNOCK_DEG, _THETA_GRID, _T_GRID))


def theta_at(t):
    return float(np.interp(min(t, MOTION_END), _T_GRID, _THETA_GRID))


def piston_top(theta_deg):
    th = np.radians(theta_deg)
    pin = CRANK_Y - (CRANK_R * np.cos(th) + np.sqrt(ROD**2 - (CRANK_R * np.sin(th)) ** 2))
    return pin - PIN_DROP


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ramp(t, start, length):
    return float(smooth((t - start) / length))


# Pressure trace (normalised, illustrative).
def _pressure(theta, knock):
    h = piston_top(theta) - HEAD_Y
    p = (piston_top(180.0) - HEAD_Y) / h
    p = p**1.32
    xb = 1 - np.exp(-5 * np.clip((theta - SPARK_DEG) / 55, 0, None) ** 3)
    p = p * (1 + 2.2 * xb)
    if knock:
        dt = np.clip(theta - KNOCK_DEG, 0, None)
        ring = np.exp(-dt / 14) * np.sin(2 * np.pi * dt / 6.5) * (theta > KNOCK_DEG)
        p = p + 0.24 * p.max() * ring + 0.22 * p.max() * np.exp(-dt / 12) * (1 - np.exp(-dt / 1.5)) * (theta > KNOCK_DEG)
    return p


TRACE_THETA = np.linspace(THETA0, THETA1, 1600)
TRACE_KNOCK = _pressure(TRACE_THETA, True)
TRACE_NORMAL = _pressure(TRACE_THETA, False)
P_SCALE = TRACE_KNOCK.max() * 1.05


def font(size, weight="Regular"):
    idx = {"Bold": 0, "Demi": 2, "Medium": 5, "Regular": 7, "Heavy": 8}[weight]
    try:
        return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", size * SS, index=idx)
    except OSError:
        return ImageFont.load_default(size * SS)


F_TITLE = font(30, "Bold")
F_SUB = font(15, "Medium")
F_LABEL = font(14, "Demi")
F_SMALL = font(12, "Medium")
F_STEP = font(15, "Demi")
F_BADGE = font(17, "Bold")


def s(v):
    return v * SS


def mix(c1, c2, a):
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * a)) for i in range(3))


def fade(c, a, bg=BG):
    return mix(bg, c, max(0.0, min(1.0, a)))


def make_background():
    yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
    d = np.hypot((xx - s(380)) / s(700), (yy - s(330)) / s(520))
    glow = np.clip(1 - d, 0, 1) ** 2
    img = np.empty((H * SS, W * SS, 3), np.float32)
    for i, (a, b) in enumerate(zip(BG, (20, 28, 42))):
        img[..., i] = a + (b - a) * glow
    return Image.fromarray(img.astype(np.uint8))


BACKGROUND = make_background()


def chamber_layer(t, yp):
    """RGBA layer for the gas inside the chamber, clipped to the bore."""
    x0, x1, y0, y1 = BORE_L, BORE_R, HEAD_Y, yp
    hh = max(int(round(s(y1 - y0))), 1)
    ww = s(x1 - x0)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    X, Y = x0 + (xx + 0.5) / SS, y0 + (yy + 0.5) / SS

    compress = smooth((t - 0.2) / max(T_SPARK - 0.2, 1e-3))
    col = np.zeros((hh, ww, 3), np.float32)
    col[:] = mix((22, 40, 66), (36, 66, 104), compress)
    col += np.clip(1 - (Y - y0) / 160, 0, 1)[..., None] * 8  # subtle depth

    # Normal flame front from the spark plug.
    if t >= T_SPARK:
        dt = t - T_SPARK
        r_at_knock = 128.0
        if t < T_KNOCK:
            R = r_at_knock * ((dt / (T_KNOCK - T_SPARK)) ** 1.1)
        else:
            R = r_at_knock + 160 * (1 - np.exp(-(t - T_KNOCK) / 0.35))
        d = np.hypot(X - PLUG[0], Y - PLUG[1])
        burned = np.clip((R - d) / 3 + 0.5, 0, 1)
        cool = ramp(t, T_KNOCK + 0.8, 1.6)
        inner = np.clip(d / max(R, 1), 0, 1)[..., None]
        hot = np.array(mix((255, 140, 40), (150, 55, 30), cool), np.float32)
        warm = np.array(mix((200, 70, 25), (95, 35, 25), cool), np.float32)
        burn_col = warm + (hot - warm) * inner
        col = col + (burn_col - col) * burned[..., None]
        front = np.exp(-(((d - R) / 4.0) ** 2)) * (1 - ramp(t, T_KNOCK + 0.25, 0.4))
        col += front[..., None] * (np.array(FLAME_FRONT, np.float32) - col) * 0.9

    # End-gas hot spots, then auto-ignition kernels at the far corners.
    kernels = [(BORE_L + 14, yp - 14), (BORE_R - 14, yp - 14)]
    pre = ramp(t, T_KNOCK - 0.35, 0.35) * (1 - ramp(t, T_KNOCK + 0.6, 0.8))
    if pre > 0:
        for kx, ky in kernels:
            dk = np.hypot(X - kx, Y - ky)
            if t < T_KNOCK:
                spot = np.exp(-((dk / (6 + 6 * pre)) ** 2)) * pre
                col += spot[..., None] * (np.array((255, 70, 50), np.float32) - col)
            else:
                rk = 95 * (1 - np.exp(-(t - T_KNOCK) / 0.07))
                core = np.clip((rk - dk) / 4 + 0.5, 0, 1) * pre
                heat = np.clip(1 - dk / max(rk, 1), 0, 1)[..., None]
                kc = np.array((255, 60, 40), np.float32) + heat * np.array((0, 190, 180), np.float32)
                col += core[..., None] * (kc - col)

    # Pressure waves: rings from both kernels plus a chamber-wide flicker.
    if t >= T_KNOCK:
        dt = t - T_KNOCK
        decay = np.exp(-dt / 0.75)
        waves = np.zeros((hh, ww), np.float32)
        for kx, ky in kernels:
            dk = np.hypot(X - kx, Y - ky)
            for n in range(6):
                rr = 420 * (dt - n * 0.11)
                if rr > 0:
                    waves += np.exp(-(((dk - rr) / 3.2) ** 2)) * np.exp(-rr / 260)
        flicker = 0.18 * decay * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * dt))
        waves = np.clip(waves * decay, 0, 1)
        col += waves[..., None] * (np.array((255, 236, 210), np.float32) - col) * 0.85
        col += flicker * (np.array(KNOCK, np.float32) - col)

    rgb = np.clip(col, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb), (s(x0), s(y0))


def draw_engine(img, t, theta, shake):
    yp = piston_top(theta)
    ox = shake
    layer, (lx, ly) = chamber_layer(t, yp)
    img.paste(layer, (int(lx + s(ox)), int(ly)))
    d = ImageDraw.Draw(img)

    def R(x0, y0, x1, y1, **kw):
        d.rectangle([s(x0 + ox), s(y0), s(x1 + ox), s(y1)], **kw)

    def RR(x0, y0, x1, y1, r, **kw):
        d.rounded_rectangle([s(x0 + ox), s(y0), s(x1 + ox), s(y1)], radius=s(r), **kw)

    def L(pts, **kw):
        d.line([(s(x + ox), s(y)) for x, y in pts], **kw)

    wall_bot = 455
    # Cylinder walls and head.
    R(BORE_L - WALL, HEAD_Y, BORE_L, wall_bot, fill=METAL)
    R(BORE_R, HEAD_Y, BORE_R + WALL, wall_bot, fill=METAL)
    for x in (BORE_L - WALL + 6, BORE_R + 6):  # coolant channels
        RR(x, HEAD_Y + 30, x + 10, wall_bot - 40, 5, fill=(28, 36, 50))
    RR(BORE_L - WALL - 8, HEAD_Y - 62, BORE_R + WALL + 8, HEAD_Y, 8, fill=METAL)
    L([(BORE_L, HEAD_Y), (BORE_L, wall_bot)], fill=OUTLINE, width=s(2))
    L([(BORE_R, HEAD_Y), (BORE_R, wall_bot)], fill=OUTLINE, width=s(2))
    L([(BORE_L, HEAD_Y), (BORE_R, HEAD_Y)], fill=OUTLINE, width=s(2))

    # Closed valves (heads seated in the roof, stems up through the head).
    for vx in (CX - 85, CX + 85):
        R(vx - 3, HEAD_Y - 62, vx + 3, HEAD_Y - 4, fill=METAL_HI)
        d.polygon([(s(vx - 30 + ox), s(HEAD_Y)), (s(vx + 30 + ox), s(HEAD_Y)),
                   (s(vx + 8 + ox), s(HEAD_Y - 8)), (s(vx - 8 + ox), s(HEAD_Y - 8))], fill=METAL_HI)

    # Spark plug.
    RR(CX - 9, HEAD_Y - 112, CX + 9, HEAD_Y - 60, 4, fill=(225, 228, 235))
    RR(CX - 13, HEAD_Y - 66, CX + 13, HEAD_Y - 44, 3, fill=(150, 160, 175))
    R(CX - 2, HEAD_Y - 44, CX + 2, HEAD_Y + 6, fill=(180, 186, 196))
    R(CX - 6, HEAD_Y + 6, CX + 3, HEAD_Y + 9, fill=(180, 186, 196))

    # Piston, rings, rod, crank.
    th = np.radians(theta)
    pin = (CX + 0, yp + PIN_DROP)
    crank_pin = (CX + CRANK_R * np.sin(th), CRANK_Y - CRANK_R * np.cos(th))
    d.ellipse([s(CX - CRANK_R - 18 + ox), s(CRANK_Y - CRANK_R - 18), s(CX + CRANK_R + 18 + ox),
               s(CRANK_Y + CRANK_R + 18)], outline=(40, 48, 62), width=s(2))
    L([pin, crank_pin], fill=METAL_HI, width=s(16), joint="curve")
    start = np.degrees(th)  # PIL arcs run clockwise from +x; the pin sits at (theta - 90)
    d.pieslice([s(CX - 50 + ox), s(CRANK_Y - 50), s(CX + 50 + ox), s(CRANK_Y + 50)],
               start, start + 180, fill=METAL)
    L([(CX, CRANK_Y), crank_pin], fill=METAL_HI, width=s(20))
    for (px, py), r in ((crank_pin, 11), ((CX, CRANK_Y), 13), (pin, 9)):
        d.ellipse([s(px - r + ox), s(py - r), s(px + r + ox), s(py + r)], fill=OUTLINE)
    RR(BORE_L + 2, yp, BORE_R - 2, yp + PISTON_H, 6, fill=(78, 90, 110))
    R(BORE_L + 2, yp, BORE_R - 2, yp + 6, fill=(120, 132, 152))
    for k in range(3):
        R(BORE_L + 2, yp + 14 + 8 * k, BORE_R - 2, yp + 16 + 8 * k, fill=(52, 60, 76))
    d.ellipse([s(pin[0] - 9 + ox), s(pin[1] - 9), s(pin[0] + 9 + ox), s(pin[1] + 9)], fill=OUTLINE)

    # Knock sensor bolted to the block.
    sx, sy = BORE_R + WALL, 300
    RR(sx, sy - 22, sx + 30, sy + 22, 5, fill=(58, 66, 82))
    RR(sx + 30, sy - 8, sx + 40, sy + 8, 2, fill=(90, 100, 118))
    hit = 0.0 if t < T_KNOCK else np.exp(-(t - T_KNOCK) / 1.2)
    if hit > 0:
        dt = t - T_KNOCK
        for n in range(3):
            ph = (dt * 1.6 - n * 0.33) % 1.0
            a = hit * (1 - ph)
            r = 14 + 34 * ph
            d.arc([s(sx + 40 - r + ox), s(sy - r), s(sx + 40 + r + ox), s(sy + r)], -45, 45,
                  fill=fade(KNOCK, a), width=s(3))
        d.ellipse([s(sx + 9 + ox), s(sy - 6), s(sx + 21 + ox), s(sy + 6)], fill=fade(KNOCK, 0.4 + 0.6 * hit, (58, 66, 82)))
    else:
        d.ellipse([s(sx + 9 + ox), s(sy - 6), s(sx + 21 + ox), s(sy + 6)], fill=(80, 90, 106))
    return yp


def draw_spark(img, t):
    a = 1 - ramp(t, T_SPARK, 0.22)
    if t < T_SPARK or a <= 0:
        return
    glow = Image.new("L", img.size, 0)
    gd = ImageDraw.Draw(glow)
    r = 10 + 30 * (1 - a)
    gd.ellipse([s(PLUG[0] - r), s(PLUG[1] + 4 - r), s(PLUG[0] + r), s(PLUG[1] + 4 + r)], fill=int(255 * a))
    glow = glow.filter(ImageFilter.GaussianBlur(s(8)))
    img.paste(Image.new("RGB", img.size, (255, 248, 225)), mask=glow)
    d = ImageDraw.Draw(img)
    for ang in range(0, 360, 45):
        rad = np.radians(ang)
        L1, L2 = 4, 4 + 16 * a
        d.line([(s(PLUG[0] + L1 * np.cos(rad)), s(PLUG[1] + 4 + L1 * np.sin(rad))),
                (s(PLUG[0] + L2 * np.cos(rad)), s(PLUG[1] + 4 + L2 * np.sin(rad)))],
               fill=(255, 255, 255), width=s(2))


def callout(d, a, text, anchor, label_xy, color, align="ra"):
    if a <= 0:
        return
    ax, ay = anchor
    lx, ly = label_xy
    col = fade(color, a)
    end = (lx + 8, ly + 10) if align == "ra" else (lx - 6, ly + 10)
    d.line([(s(ax), s(ay)), (s(end[0]), s(end[1]))], fill=fade(color, 0.7 * a), width=s(1))
    d.ellipse([s(ax - 3), s(ay - 3), s(ax + 3), s(ay + 3)], fill=col)
    d.text((s(lx), s(ly)), text, font=F_LABEL, fill=col, anchor=align)


def draw_panel(img, t, theta):
    d = ImageDraw.Draw(img)
    px0, px1 = 690, 1220
    intro = ramp(t, 0.0, 0.5)
    d.text((s(px0), s(64)), "ENGINE KNOCK", font=F_TITLE, fill=fade(TEXT, intro))
    d.text((s(px0), s(104)), "Uncontrolled end-gas auto-ignition", font=F_SUB, fill=fade(MUTED, intro))

    # Pressure plot.
    gx0, gx1, gy0, gy1 = px0, px1, 160, 420
    d.rounded_rectangle([s(gx0), s(gy0), s(gx1), s(gy1)], radius=s(10), fill=(16, 22, 33), outline=(34, 42, 56), width=s(1))
    d.text((s(gx0 + 16), s(gy0 + 12)), "Cylinder pressure", font=F_SMALL, fill=MUTED)
    ix0, ix1, iy0, iy1 = gx0 + 20, gx1 - 20, gy0 + 40, gy1 - 34

    def to_xy(th, p):
        return (s(ix0 + (th - THETA0) / (THETA1 - THETA0) * (ix1 - ix0)), s(iy1 - p / P_SCALE * (iy1 - iy0)))

    tdc = to_xy(0, 0)[0]
    for yy in np.linspace(iy0, iy1, 5):
        d.line([(s(ix0), s(yy)), (s(ix1), s(yy))], fill=(26, 33, 46), width=s(1))
    d.line([(tdc, s(iy0)), (tdc, s(iy1))], fill=(44, 54, 70), width=s(1))
    d.text((tdc, s(iy1 + 8)), "TDC", font=F_SMALL, fill=MUTED, anchor="ma")
    d.text((s(ix1), s(iy1 + 8)), "crank angle", font=F_SMALL, fill=MUTED, anchor="ra")

    upto = TRACE_THETA <= theta
    pre = upto & (TRACE_THETA <= KNOCK_DEG)
    if pre.sum() > 1:
        d.line([to_xy(a, p) for a, p in zip(TRACE_THETA[pre], TRACE_KNOCK[pre])], fill=TEXT, width=s(3), joint="curve")
    post = upto & (TRACE_THETA >= KNOCK_DEG - 0.2)
    if post.sum() > 1:
        d.line([to_xy(a, p) for a, p in zip(TRACE_THETA[post], TRACE_KNOCK[post])], fill=KNOCK, width=s(3), joint="curve")
    # Normal-combustion reference (dashed), revealed once knock starts.
    ref_a = ramp(t, T_KNOCK + 0.3, 0.5)
    if ref_a > 0:
        ref = upto & (TRACE_THETA >= KNOCK_DEG - 4)
        pts = [to_xy(a, p) for a, p in zip(TRACE_THETA[ref], TRACE_NORMAL[ref])]
        for i in range(0, len(pts) - 1, 10):
            d.line(pts[i:i + 6], fill=fade(NORMAL, ref_a, (16, 22, 33)), width=s(2))
    if t < MOTION_END + 0.5:
        cur = to_xy(theta, float(np.interp(theta, TRACE_THETA, TRACE_KNOCK)))
        c = KNOCK if theta > KNOCK_DEG else TEXT
        d.ellipse([cur[0] - s(5), cur[1] - s(5), cur[0] + s(5), cur[1] + s(5)], fill=c)

    # Legend in the plot header.
    lg = [("knock ringing", KNOCK, ramp(t, T_KNOCK + 0.15, 0.3)), ("normal burn", NORMAL, ref_a)]
    x = gx1 - 16
    for label, colr, a in lg:
        if a <= 0:
            continue
        col = fade(colr, a, (16, 22, 33))
        d.text((s(x), s(gy0 + 20)), label, font=F_SMALL, fill=col, anchor="rm")
        x -= F_SMALL.getlength(label) / SS + 22
        d.line([(s(x), s(gy0 + 20)), (s(x + 14), s(gy0 + 20))], fill=col, width=s(3))
        x -= 18

    # Three-step explainer.
    steps = [("Spark lights a normal flame front", T_SPARK),
             ("Unburned end gas auto-ignites ahead of it", T_KNOCK - 0.35),
             ("Colliding pressure waves ring the cylinder", T_KNOCK + 0.1)]
    for i, (label, ts) in enumerate(steps):
        y = 452 + i * 40
        on = ramp(t, ts, 0.25)
        accent = KNOCK if i > 0 else FLAME_FRONT
        nxt = steps[i + 1][1] if i + 1 < len(steps) else 99
        done = ramp(t, nxt, 0.3)
        fill = on * (1 - done)
        d.ellipse([s(px0), s(y), s(px0 + 24), s(y + 24)], outline=fade(accent, 0.25 + 0.75 * on), width=s(2),
                  fill=fade(accent, fill) if fill > 0 else None)
        num = mix(fade(MUTED, 0.6), accent, on) if fill < 0.5 else BG
        d.text((s(px0 + 12), s(y + 12)), str(i + 1), font=F_SMALL, fill=num, anchor="mm")
        d.text((s(px0 + 38), s(y + 12)), label, font=F_STEP, fill=fade(TEXT, 0.3 + 0.7 * on * (1 - 0.35 * done)), anchor="lm")

    # Detection badge.
    b = ramp(t, T_KNOCK + 0.2, 0.18)
    if b > 0:
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.6 * (t - T_KNOCK))
        bx, by, bw, bh = px0, 590, 230, 44
        grow = 1 + 0.15 * (1 - b)
        cx_, cy_ = bx + bw / 2, by + bh / 2
        hw, hh = bw / 2 * grow, bh / 2 * grow
        d.rounded_rectangle([s(cx_ - hw - 4), s(cy_ - hh - 4), s(cx_ + hw + 4), s(cy_ + hh + 4)], radius=s(26),
                            outline=fade(KNOCK, b * (0.25 + 0.35 * pulse)), width=s(2))
        d.rounded_rectangle([s(cx_ - hw), s(cy_ - hh), s(cx_ + hw), s(cy_ + hh)], radius=s(22), fill=fade(KNOCK, b))
        d.ellipse([s(bx + 18), s(cy_ - 5), s(bx + 28), s(cy_ + 5)], fill=fade((255, 255, 255), b * (0.6 + 0.4 * pulse), KNOCK))
        d.text((s(bx + 40), s(cy_)), "KNOCK DETECTED", font=F_BADGE, fill=fade((255, 255, 255), b, KNOCK), anchor="lm")


def render(t):
    theta = theta_at(t)
    dt = t - T_KNOCK
    shake = 0.0 if dt < 0 else 3.2 * np.exp(-dt / 0.35) * np.sin(2 * np.pi * 13 * dt)
    img = BACKGROUND.copy()
    yp = draw_engine(img, t, theta, shake)
    draw_spark(img, t)
    d = ImageDraw.Draw(img)
    callout(d, ramp(t, T_SPARK, 0.25), "Spark", (PLUG[0] - 10, PLUG[1] + 10), (170, 92), FLAME_FRONT)
    callout(d, ramp(t, T_KNOCK - 0.3, 0.25), "End-gas", (BORE_L + 14, yp - 14), (150, 250), KNOCK)
    callout(d, ramp(t, T_KNOCK - 0.3, 0.25), "auto-ignition", (BORE_L + 14, yp - 14), (150, 268), KNOCK)
    callout(d, ramp(t, T_KNOCK + 0.25, 0.3), "Pressure", (BORE_L + 70, HEAD_Y + 30), (150, 160), (255, 236, 210))
    callout(d, ramp(t, T_KNOCK + 0.25, 0.3), "waves", (BORE_L + 70, HEAD_Y + 30), (150, 178), (255, 236, 210))
    callout(d, ramp(t, T_KNOCK + 0.1, 0.3), "Knock sensor", (BORE_R + WALL + 15, 322), (BORE_R + WALL + 40, 352), KNOCK, "la")
    draw_panel(img, t, theta)
    return img.resize((W, H), Image.LANCZOS)


def main(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(render(0.0), np.float32)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )
    for i in range(N_FRAMES):
        t = i / FPS
        frame = np.asarray(render(t), np.float32)
        a = ramp(t, FADE_START, DURATION - FADE_START - 1 / FPS)  # cross-fade to frame 0 for a seamless loop
        frame = frame + (first - frame) * a
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {out_path} ({N_FRAMES} frames, spark t={T_SPARK:.2f}s, knock t={T_KNOCK:.2f}s)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
