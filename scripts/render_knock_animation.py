"""Render the compact knock clip shown beside the dashboard's knock alert.

A 3 s seamless loop (480x480 MP4, intended to display at ~240 px): the piston
compresses the charge, the spark lights a flame front, the end gas at the bore
edges auto-ignites ahead of it, and shock rings plus a sensor-trace burst mark
the knock. Illustrative motion only, not simulated engine data.

Motion system: one signature curve (ease-in-out sine) for the piston and
ambient layers, ease-out for arrivals, ease-out-expo for the knock burst, and a
firm three-oscillation shake with no overshoot.

Run from the repository root (requires ffmpeg on PATH):
    uv run python scripts/render_knock_animation.py [output.mp4]
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE, OUT_PX, SS = 240, 480, 4  # design units, output pixels, supersampling per unit
FPS, LOOP = 30, 3.0
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "dashboard" / "assets" / "knock_animation.mp4"

BG = np.array((13, 18, 27), np.float32)
LINE = (86, 98, 120)
METAL = (58, 68, 86)
METAL_TOP = (112, 124, 146)
KNOCK = np.array((255, 127, 14), np.float32)  # dashboard FAULT_COLORS["knock_like"]
CHARGE_LO, CHARGE_HI = np.array((24, 42, 68), np.float32), np.array((38, 74, 118), np.float32)
FLAME_HOT, FLAME_WARM = np.array((255, 150, 48), np.float32), np.array((214, 78, 28), np.float32)
EMBER = np.array((92, 36, 26), np.float32)
FRONT = np.array((255, 222, 140), np.float32)
WHITE_HOT = np.array((255, 244, 222), np.float32)
END_GAS = np.array((255, 64, 44), np.float32)

# Geometry (design units).
BORE_L, BORE_R, HEAD_Y, WALL_BOT = 72, 168, 46, 168
CX = (BORE_L + BORE_R) / 2
TDC_Y, BDC_Y, PISTON_H = 80, 126, 30
TRACE_X0, TRACE_X1, TRACE_Y = 28, 212, 204

# Timeline (seconds within the loop).
T_SPARK, T_KNOCK = 0.92, 1.36


def clamp01(x):
    return float(min(1.0, max(0.0, x)))


def prog(t, start, dur):
    return clamp01((t - start) / dur)


def ease_in_out_sine(x):
    return 0.5 - 0.5 * np.cos(np.pi * x)


def ease_out_cubic(x):
    return 1 - (1 - x) ** 3


def ease_out_expo(x):
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def lerp(a, b, k):
    return a + (b - a) * k


def piston_y(t):
    if t < 0.9:
        return BDC_Y + (TDC_Y - BDC_Y) * ease_in_out_sine(t / 0.9)
    if t < 1.5:  # short dwell at TDC keeps the combustion legible
        return TDC_Y
    if t < 2.55:
        return TDC_Y + (BDC_Y - TDC_Y) * ease_in_out_sine((t - 1.5) / 1.05)
    return BDC_Y


def shake(t):
    x = t - T_KNOCK
    if x < 0 or x > 0.36:
        return 0.0
    return 2.6 * np.sin(2 * np.pi * x / 0.12) * (1 - x / 0.36) ** 2


def knock_level(t):
    """0..1 intensity of the knock moment, used by secondary and ambient layers."""
    x = t - T_KNOCK
    if x < 0:
        return 0.0
    return ease_out_expo(clamp01(x / 0.1)) * float(np.exp(-x / 0.45))


def chamber(t, yp):
    """RGB array of the gas between head and piston (supersampled)."""
    hh = max(int(round((yp - HEAD_Y) * SS)), 1)
    ww = int((BORE_R - BORE_L) * SS)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    X, Y = BORE_L + (xx + 0.5) / SS, HEAD_Y + (yy + 0.5) / SS
    shade = (1.08 - 0.12 * (Y - HEAD_Y) / 60)[..., None]

    comp = 1 - (yp - TDC_Y) / (BDC_Y - TDC_Y)
    fresh = lerp(CHARGE_LO, CHARGE_HI, comp) * shade
    col = fresh.copy()

    # Flame front from the plug, then full burn after the knock.
    if t >= T_SPARK:
        r_knock = 44.0
        if t < T_KNOCK:
            R = r_knock * ease_out_cubic(prog(t, T_SPARK, T_KNOCK - T_SPARK)) ** 0.9
        else:
            R = r_knock + 70 * ease_out_cubic(prog(t, T_KNOCK, 0.4))
        d = np.hypot(X - CX, Y - HEAD_Y)
        burned = np.clip((R - d) * SS / 3 + 0.5, 0, 1)[..., None]
        cool = ease_in_out_sine(prog(t, 1.6, 0.9))
        inner = np.clip(d / max(R, 1), 0, 1)[..., None]
        gas = lerp(lerp(FLAME_HOT, EMBER, cool), lerp(FLAME_WARM, EMBER * 0.85, cool), inner)
        col = lerp(col, gas, burned)
        edge = np.exp(-(((d - R) / 1.6) ** 2))[..., None] * (1 - prog(t, T_KNOCK + 0.1, 0.3))
        col = lerp(col, FRONT, edge * 0.9)

    # End-gas hot spots, then auto-ignition kernels in the far corners.
    kernels = [(BORE_L + 6, yp - 6), (BORE_R - 6, yp - 6)]
    hot = ease_out_cubic(prog(t, T_KNOCK - 0.2, 0.2)) * (1 - ease_in_out_sine(prog(t, 1.7, 0.6)))
    if hot > 0:
        pop = ease_out_expo(prog(t, T_KNOCK, 0.16))
        for kx, ky in kernels:
            dk = np.hypot(X - kx, Y - ky)
            if t < T_KNOCK:
                spot = np.exp(-((dk / (2.5 + 2.5 * hot)) ** 2))[..., None] * hot
                col = lerp(col, END_GAS, spot)
            else:
                rk = 4 + 22 * pop
                core = np.clip((rk - dk) * SS / 3 + 0.5, 0, 1)[..., None] * hot
                heat = np.clip(1 - dk / rk, 0, 1)[..., None]
                col = lerp(col, lerp(END_GAS, WHITE_HOT, heat), core)

    # Shock rings from both kernels.
    x = t - T_KNOCK
    if 0 <= x < 0.6:
        rings = np.zeros(X.shape, np.float32)
        for kx, ky in kernels:
            dk = np.hypot(X - kx, Y - ky)
            for n in range(3):
                rr = 170 * (x - n * 0.08)
                if rr > 0:
                    rings += np.exp(-(((dk - rr) / 1.1) ** 2))
        col = lerp(col, WHITE_HOT, np.clip(rings * (1 - x / 0.6) ** 2, 0, 1)[..., None] * 0.8)

    # Fresh charge for the next cycle closes the loop.
    col = lerp(col, fresh, ease_in_out_sine(prog(t, 2.45, 0.5)))
    return np.clip(col, 0, 255).astype(np.uint8)


def sensor_signal(tau):
    """LOOP-periodic vibration signal with a damped burst at the knock, plus its envelope."""
    tau = np.mod(tau, LOOP)
    w = 2 * np.pi / LOOP
    base = 0.10 * np.sin(w * 37 * tau) + 0.06 * np.sin(w * 61 * tau + 1.3) + 0.04 * np.sin(w * 89 * tau + 0.4)
    x = tau - T_KNOCK
    on = x >= 0
    burst = np.where(on, np.exp(-np.clip(x, 0, None) / 0.16) * np.sin(2 * np.pi * 15 * x), 0.0)
    return base + 0.95 * burst, np.where(on, np.exp(-np.clip(x, 0, None) / 0.18), 0.0)


_YY, _XX = np.mgrid[0:SIZE * SS, 0:SIZE * SS].astype(np.float32) / SS
_GLOW = np.clip(1 - np.hypot(_XX - CX, _YY - 100) / 150, 0, 1) ** 2


def make_background(t):
    breathe = 0.85 + 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * t / LOOP))
    tint = lerp(np.array((22, 32, 48), np.float32), np.array((70, 38, 18), np.float32), knock_level(t))
    img = BG + (tint - BG) * (_GLOW * breathe)[..., None]
    return Image.fromarray(img.astype(np.uint8))


def s(v):
    return v * SS


def rgb(c):
    return tuple(int(v) for v in c)


def render(t):
    img = make_background(t)
    k = knock_level(t)
    ox = shake(t)
    yp = piston_y(t)

    # Ambient knock bloom behind the cylinder.
    if k > 0.01:
        bloom = Image.new("L", img.size, 0)
        ImageDraw.Draw(bloom).rounded_rectangle(
            [s(BORE_L - 4), s(HEAD_Y), s(BORE_R + 4), s(yp)], radius=s(6), fill=int(150 * k))
        bloom = bloom.filter(ImageFilter.GaussianBlur(s(14)))
        img.paste(Image.new("RGB", img.size, rgb(KNOCK)), mask=bloom)

    img.paste(Image.fromarray(chamber(t, yp)), (int(s(BORE_L + ox)), int(s(HEAD_Y))))
    d = ImageDraw.Draw(img)

    def rr(x0, y0, x1, y1, r, **kw):
        d.rounded_rectangle([s(x0 + ox), s(y0), s(x1 + ox), s(y1)], radius=s(r), **kw)

    # Piston with a rod stub.
    rr(CX - 5, yp + PISTON_H - 4, CX + 5, WALL_BOT + 14, 3, fill=(40, 48, 62))
    rr(BORE_L + 1.5, yp, BORE_R - 1.5, yp + PISTON_H, 4, fill=METAL)
    rr(BORE_L + 1.5, yp, BORE_R - 1.5, yp + 2.5, 1.2, fill=METAL_TOP)
    for i in range(2):
        rr(BORE_L + 1.5, yp + 8 + 5 * i, BORE_R - 1.5, yp + 9.2 + 5 * i, 0.5, fill=(44, 52, 68))

    # Cylinder liner and head as clean strokes; the liner warms with the knock.
    wall = rgb(lerp(np.array(LINE, np.float32), KNOCK, 0.7 * k))
    d.line([(s(BORE_L + ox), s(WALL_BOT)), (s(BORE_L + ox), s(HEAD_Y)), (s(BORE_R + ox), s(HEAD_Y)),
            (s(BORE_R + ox), s(WALL_BOT))], fill=wall, width=round(s(1.6)), joint="curve")
    rr(BORE_L - 10, HEAD_Y - 14, BORE_R + 10, HEAD_Y - 2, 4, fill=(30, 37, 50))

    # Spark plug: anticipation glow, then a crisp flash.
    rr(CX - 4, HEAD_Y - 30, CX + 4, HEAD_Y - 14, 2.5, fill=(214, 220, 230))
    rr(CX - 1, HEAD_Y - 14, CX + 1, HEAD_Y + 2, 0.6, fill=(160, 168, 182))
    pre = ease_in_out_sine(prog(t, T_SPARK - 0.12, 0.12)) if t < T_SPARK else 0.0
    flash = (1 - ease_out_cubic(prog(t, T_SPARK, 0.22))) if t >= T_SPARK else 0.0
    a = max(0.35 * pre, flash)
    if a > 0:
        glow = Image.new("L", img.size, 0)
        r = 3 + 9 * flash
        ImageDraw.Draw(glow).ellipse([s(CX + ox - r), s(HEAD_Y + 2 - r), s(CX + ox + r), s(HEAD_Y + 2 + r)],
                                     fill=int(255 * a))
        glow = glow.filter(ImageFilter.GaussianBlur(s(3)))
        img.paste(Image.new("RGB", img.size, (255, 246, 220)), mask=glow)
        d = ImageDraw.Draw(img)

    # Vibration arcs outside the liner at the knock.
    x = t - T_KNOCK
    if 0 <= x < 0.7:
        for cx_, (a0, a1) in ((BORE_L - 6, (135, 225)), (BORE_R + 6, (-45, 45))):
            for n in range(2):
                ph = clamp01((x - n * 0.1) / 0.5)
                if 0 < ph < 1:
                    r = 6 + 16 * ease_out_cubic(ph)
                    d.arc([s(cx_ + ox - r), s(102 - r), s(cx_ + ox + r), s(102 + r)], a0, a1,
                          fill=rgb(lerp(BG, KNOCK, (1 - ph) ** 1.5)), width=round(s(1.4)))

    # Scrolling sensor trace: the window shows the last LOOP seconds, so the burst flows right to left.
    xs = np.linspace(TRACE_X0, TRACE_X1, 360)
    sig, env = sensor_signal(t - LOOP * (TRACE_X1 - xs) / (TRACE_X1 - TRACE_X0))
    ys = TRACE_Y - 14 * sig
    fade_in = np.clip((xs - TRACE_X0) / 30, 0, 1)
    idle = np.array((70, 84, 106), np.float32)
    for i in range(len(xs) - 1):
        c = lerp(BG, lerp(idle, KNOCK, float(np.clip(env[i] * 1.4, 0, 1))), float(fade_in[i]))
        d.line([(s(xs[i]), s(ys[i])), (s(xs[i + 1]), s(ys[i + 1]))], fill=rgb(c), width=round(s(1.3)))
    head_c = rgb(lerp(np.array((150, 162, 182), np.float32), KNOCK, float(np.clip(env[-1] * 1.4, 0, 1))))
    d.ellipse([s(TRACE_X1 - 2.2), s(ys[-1] - 2.2), s(TRACE_X1 + 2.2), s(ys[-1] + 2.2)], fill=head_c)

    # Tile edge flash: secondary emphasis that ties the clip to the alert.
    if k > 0.01:
        d.rounded_rectangle([s(3), s(3), s(SIZE - 3), s(SIZE - 3)], radius=s(18),
                            outline=rgb(lerp(BG, KNOCK, 0.85 * k)), width=round(s(1.5)))
    return img.resize((OUT_PX, OUT_PX), Image.LANCZOS)


def main(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_PX}x{OUT_PX}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )
    n = int(FPS * LOOP)
    for i in range(n):
        proc.stdin.write(np.asarray(render(i / FPS), np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {out_path} ({n} frames, {OUT_PX}x{OUT_PX}, {LOOP:.1f} s loop)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
